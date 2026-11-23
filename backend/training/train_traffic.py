"""Train MODEL 3: festival-time traffic congestion level (low / medium / high).

Run from the backend folder:
    python -m training.train_traffic
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline

from app.config import ACCIDENT_DATASET, RANDOM_STATE, SIGNIFICANT_DATES_DATASET, TRAFFIC_MODEL_PATH
from app.ml_models import traffic_model as tm
from app.utils.events import NO_EVENT
from training.common import chronological_split, describe_split, file_sha256, log, md_table, save_bundle, write_report

warnings.filterwarnings("ignore", message="Unknown solver options")

VAL_START, TEST_START = "2024-01-01", "2024-07-01"


def candidates():
    from app.utils.preprocessing import make_preprocessor

    pre = make_preprocessor(tm.CATEGORICAL_FEATURES, tm.NUMERIC_FEATURES)
    return {
        "LogisticRegression": Pipeline([("preprocess", clone(pre)), ("model", LogisticRegression(max_iter=3000))]),
        "RandomForest": Pipeline([("preprocess", clone(pre)), ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_STATE))]),
        "HistGradientBoosting": Pipeline([("preprocess", clone(pre)), ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE))]),
    }


def metrics(y, proba, classes) -> dict:
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    return {
        "macro_f1": f1_score(y, pred, average="macro", labels=tm.CLASSES, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "log_loss": log_loss(y, proba, labels=classes),
    }


def label_metrics(y, pred) -> dict:
    return {
        "macro_f1": f1_score(y, pred, average="macro", labels=tm.CLASSES, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
    }


def main() -> None:
    log.info("Loading %s + %s", ACCIDENT_DATASET.name, SIGNIFICANT_DATES_DATASET.name)
    raw = pd.read_csv(ACCIDENT_DATASET)
    frame = tm.build_training_frame(raw, SIGNIFICANT_DATES_DATASET)
    train, val, test = chronological_split(frame, "date", VAL_START, TEST_START)
    split = describe_split(train, val, test, "date")
    log.info("Split: %s", split)
    event_rows = int((frame["festival_event"] != NO_EVENT).sum())
    log.info("Rows on a festival / significant date: %d of %d", event_rows, len(frame))

    comparison = {}
    for name, pipe in candidates().items():
        pipe.fit(train[tm.FEATURES], train[tm.TARGET])
        comparison[name] = metrics(val[tm.TARGET], pipe.predict_proba(val[tm.FEATURES]), list(pipe.classes_))
        log.info("%s validation: %s", name, {k: round(v, 4) for k, v in comparison[name].items()})
    best_name = max(comparison, key=lambda n: comparison[n]["macro_f1"])
    log.info("Selected %s (highest validation macro-F1)", best_name)

    train_val = pd.concat([train, val])
    final = candidates()[best_name].fit(train_val[tm.FEATURES], train_val[tm.TARGET])
    classes = list(final.classes_)
    y_test = test[tm.TARGET].to_numpy()
    test_proba = final.predict_proba(test[tm.FEATURES])
    test_pred = np.asarray(classes)[test_proba.argmax(axis=1)]
    model_metrics = metrics(y_test, test_proba, classes)

    majority = train_val[tm.TARGET].mode()[0]
    peak_rule = np.where(test["is_peak_hour"] == 1, "high", "low")
    baselines = {
        f"majority class ('{majority}')": label_metrics(y_test, np.full(len(test), majority)),
        "peak-hour rule (peak=high, else low)": label_metrics(y_test, peak_rule),
    }

    # Does the festival/event signal change anything? Compare share of 'high' on event vs normal days.
    event_effect = (
        frame.assign(is_high=(frame[tm.TARGET] == "high").astype(int))
        .groupby("festival_event")["is_high"].agg(["mean", "count"]).round(3).to_dict(orient="index")
    )
    importance = permutation_importance(final, test[tm.FEATURES], y_test, n_repeats=5, random_state=RANDOM_STATE, scoring="f1_macro")
    importances = sorted(zip(tm.FEATURES, importance.importances_mean), key=lambda item: -item[1])

    report = {
        "model": "traffic",
        "task": "Traffic congestion level (traffic_density proxy) with festival / significant-date features",
        "dataset": [ACCIDENT_DATASET.name, SIGNIFICANT_DATES_DATASET.name],
        "split": split,
        "selection_metric": "validation macro-F1 (higher is better)",
        "validation_comparison": comparison,
        "selected_model": best_name,
        "test_metrics": model_metrics,
        "test_baselines": baselines,
        "test_confusion_matrix": {"labels": tm.CLASSES, "matrix": confusion_matrix(y_test, test_pred, labels=tm.CLASSES).tolist()},
        "test_classification_report": classification_report(y_test, test_pred, labels=tm.CLASSES, output_dict=True, zero_division=0),
        "share_high_by_festival_event": event_effect,
        "rows_on_festival_or_event": event_rows,
        "permutation_importance_macro_f1": dict(importances),
        "limitations": [
            "No uploaded dataset measures traffic congestion; traffic_density from accident records is used as a proxy.",
            "traffic_density is observed only where and when accidents happened, so it is a biased sample of traffic.",
            f"Only {event_rows} of {len(frame)} rows fall on a festival or significant date, so festival effects are estimated from little data.",
            "pandals.csv and reviews.csv contain no traffic, crowd-count or date fields and are not used.",
        ],
    }
    md = "\n\n".join([
        "# Model 3 - Festival-time traffic congestion: evaluation report",
        f"Datasets: `{ACCIDENT_DATASET.name}` (target `traffic_density`) + `{SIGNIFICANT_DATES_DATASET.name}` (event dates). "
        f"Split: train {split['train']['from']}..{split['train']['to']}, validation {split['validation']['from']}..{split['validation']['to']}, test {split['test']['from']}..{split['test']['to']}",
        "## Algorithm comparison (validation)",
        md_table([[n, f"{m['macro_f1']:.3f}", f"{m['accuracy']:.3f}", f"{m['log_loss']:.3f}"] for n, m in comparison.items()], ["Model", "Macro-F1", "Accuracy", "Log loss"]),
        f"Selected: **{best_name}** (highest validation macro-F1).",
        "## Test period - model vs baselines",
        md_table([[best_name, f"{model_metrics['macro_f1']:.3f}", f"{model_metrics['accuracy']:.3f}", f"{model_metrics['balanced_accuracy']:.3f}"]]
                 + [[n, f"{m['macro_f1']:.3f}", f"{m['accuracy']:.3f}", f"{m['balanced_accuracy']:.3f}"] for n, m in baselines.items()],
                 ["Model", "Macro-F1", "Accuracy", "Balanced accuracy"]),
        "## Share of 'high' congestion by festival / event (all rows)",
        md_table([[k, v["mean"], v["count"]] for k, v in event_effect.items()], ["Festival / event", "Share high", "Rows"]),
        "## Feature importance (permutation, test set, drop in macro-F1)",
        md_table([[f, f"{v:.4f}"] for f, v in importances], ["Feature", "Importance"]),
        "## Limitations\n" + "\n".join(f"- {item}" for item in report["limitations"]),
    ])
    write_report("traffic", report, md)

    save_bundle(
        {
            "pipeline": final,
            "model_name": best_name,
            "features": tm.FEATURES,
            "classes": classes,
            "metrics": {"test": model_metrics, "baselines": baselines, "split": split},
            "feature_importance": dict(importances),
            "input_options": {
                "city": sorted(raw["city"].unique().tolist()),
                "road_type": sorted(raw["road_type"].unique().tolist()),
                "weather": sorted(raw["weather"].unique().tolist()),
                "festival": sorted(frame["festival_event"].unique().tolist(), key=lambda v: (v != NO_EVENT, v)),
                "lanes": [int(raw["lanes"].min()), int(raw["lanes"].max())],
            },
            "dataset": [ACCIDENT_DATASET.name, SIGNIFICANT_DATES_DATASET.name],
            "dataset_sha256": file_sha256(ACCIDENT_DATASET),
        },
        TRAFFIC_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
