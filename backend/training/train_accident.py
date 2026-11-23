"""Train MODEL 2: pre-accident road risk score (regression, 0-1).

Run from the backend folder:
    python -m training.train_accident
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from app.config import ACCIDENT_DATASET, ACCIDENT_MODEL_PATH, RANDOM_STATE
from app.ml_models import accident_model as am
from app.utils.preprocessing import make_preprocessor
from training.common import chronological_split, describe_split, file_sha256, log, md_table, save_bundle, write_report

VAL_START, TEST_START = "2024-01-01", "2024-07-01"


def candidates():
    pre = make_preprocessor(am.CATEGORICAL_FEATURES, am.NUMERIC_FEATURES)
    return {
        "LinearRegression (Ridge)": Pipeline([("preprocess", clone(pre)), ("model", Ridge(alpha=1e-3))]),
        "RandomForest": Pipeline([("preprocess", clone(pre)), ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE))]),
        "HistGradientBoosting": Pipeline([("preprocess", clone(pre)), ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE))]),
    }


def metrics(y, pred) -> dict:
    pred = np.clip(pred, 0, 1)
    bands_true = [am.risk_band(v) for v in y]
    bands_pred = [am.risk_band(v) for v in pred]
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": r2_score(y, pred),
        "risk_band_accuracy": accuracy_score(bands_true, bands_pred),
    }


def main() -> None:
    log.info("Loading %s", ACCIDENT_DATASET.name)
    raw = pd.read_csv(ACCIDENT_DATASET)
    frame = am.build_training_frame(raw)
    train, val, test = chronological_split(frame, "date", VAL_START, TEST_START)
    split = describe_split(train, val, test, "date")
    log.info("Split: %s", split)

    comparison = {}
    for name, pipe in candidates().items():
        pipe.fit(train[am.FEATURES], train[am.TARGET])
        comparison[name] = metrics(val[am.TARGET].to_numpy(), pipe.predict(val[am.FEATURES]))
        log.info("%s validation: %s", name, {k: round(v, 4) for k, v in comparison[name].items()})
    best_name = min(comparison, key=lambda n: comparison[n]["mae"])
    log.info("Selected %s (lowest validation MAE)", best_name)

    # Uncertainty: 90th percentile of absolute validation error of the winner.
    winner_on_train = candidates()[best_name].fit(train[am.FEATURES], train[am.TARGET])
    val_abs_err = np.abs(val[am.TARGET].to_numpy() - np.clip(winner_on_train.predict(val[am.FEATURES]), 0, 1))
    error_p90 = float(np.quantile(val_abs_err, 0.9))

    train_val = pd.concat([train, val])
    final = candidates()[best_name].fit(train_val[am.FEATURES], train_val[am.TARGET])
    y_test = test[am.TARGET].to_numpy()
    model_metrics = metrics(y_test, final.predict(test[am.FEATURES]))
    baseline_metrics = metrics(y_test, np.full(len(test), train_val[am.TARGET].mean()))

    importance = permutation_importance(final, test[am.FEATURES], y_test, n_repeats=5, random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    importances = sorted(zip(am.FEATURES, importance.importances_mean), key=lambda item: -item[1])

    # For linear models the API can explain each prediction exactly:
    # effect of an input = coef x (encoded value - average encoded training value).
    preprocess = final.named_steps["preprocess"]
    onehot = preprocess.named_transformers_["categorical"].named_steps["onehot"]
    feature_widths = [(name, len(cats)) for name, cats in zip(am.CATEGORICAL_FEATURES, onehot.categories_)]
    feature_widths += [(name, 1) for name in am.NUMERIC_FEATURES]
    linear_reference = preprocess.transform(train_val[am.FEATURES]).mean(axis=0)

    # Weather/visibility pairs actually present in the data (e.g. clear weather is always high visibility).
    valid_pairs = sorted({f"{w}|{v}" for w, v in zip(raw["weather"], raw["visibility"])})

    report = {
        "model": "accident",
        "task": "Pre-accident road risk score (0-1) = risk_score minus the post-crash fatal adjustment",
        "dataset": ACCIDENT_DATASET.name,
        "split": split,
        "selection_metric": "validation MAE (lower is better)",
        "validation_comparison": comparison,
        "selected_model": best_name,
        "test_metrics": model_metrics,
        "test_baseline_mean": baseline_metrics,
        "prediction_error_p90": error_p90,
        "permutation_importance_mae": dict(importances),
        "limitations": [
            "Every record is an accident, so the model scores risky conditions; it does not estimate the probability that an accident happens.",
            "The dataset's risk_score is rule-based; the model learns that rule (very high R2 is expected, not a sign of real-world accuracy).",
            "accident_severity could not be predicted from the inputs (no signal), so severity is not offered.",
        ],
    }
    md = "\n\n".join([
        "# Model 2 - Accident risk: evaluation report",
        f"Dataset: `{ACCIDENT_DATASET.name}` · Split: train {split['train']['from']}..{split['train']['to']}, "
        f"validation {split['validation']['from']}..{split['validation']['to']}, test {split['test']['from']}..{split['test']['to']}",
        "Target: `pre_accident_risk = risk_score - 0.20 x (accident_severity == fatal)`",
        "## Algorithm comparison (validation)",
        md_table([[n, f"{m['mae']:.4f}", f"{m['rmse']:.4f}", f"{m['r2']:.4f}", f"{m['risk_band_accuracy']:.3f}"] for n, m in comparison.items()], ["Model", "MAE", "RMSE", "R2", "Risk-band accuracy"]),
        f"Selected: **{best_name}** (lowest validation MAE).",
        "## Test period - model vs baseline",
        md_table([[best_name, f"{model_metrics['mae']:.4f}", f"{model_metrics['rmse']:.4f}", f"{model_metrics['r2']:.4f}", f"{model_metrics['risk_band_accuracy']:.3f}"],
                  ["Baseline (training mean)", f"{baseline_metrics['mae']:.4f}", f"{baseline_metrics['rmse']:.4f}", f"{baseline_metrics['r2']:.4f}", f"{baseline_metrics['risk_band_accuracy']:.3f}"]],
                 ["Model", "MAE", "RMSE", "R2", "Risk-band accuracy"]),
        f"90% of validation predictions were within **±{error_p90:.3f}** of the true score.",
        "## Feature importance (permutation, test set, increase in MAE)",
        md_table([[f, f"{v:.4f}"] for f, v in importances], ["Feature", "Importance"]),
        "## Limitations\n" + "\n".join(f"- {item}" for item in report["limitations"]),
    ])
    write_report("accident", report, md)

    save_bundle(
        {
            "pipeline": final,
            "model_name": best_name,
            "features": am.FEATURES,
            "target": am.TARGET,
            "risk_bands": am.RISK_BANDS,
            "error_p90": error_p90,
            "feature_widths": feature_widths,
            "linear_reference": linear_reference,
            "metrics": {"test": model_metrics, "baseline": baseline_metrics, "split": split},
            "feature_importance": dict(importances),
            "input_options": {
                "city": sorted(raw["city"].unique().tolist()),
                "road_type": sorted(raw["road_type"].unique().tolist()),
                "weather": sorted(raw["weather"].unique().tolist()),
                "visibility": sorted(raw["visibility"].unique().tolist()),
                "traffic_density": ["low", "medium", "high"],
                "festival": [am.NO_FESTIVAL] + sorted(raw["festival"].dropna().unique().tolist()),
                "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                "lanes": [int(raw["lanes"].min()), int(raw["lanes"].max())],
                "temperature": [int(raw["temperature"].min()), int(raw["temperature"].max())],
                "weather_visibility_pairs": valid_pairs,
            },
            "dataset": ACCIDENT_DATASET.name,
            "dataset_sha256": file_sha256(ACCIDENT_DATASET),
        },
        ACCIDENT_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
