"""Train MODEL 1: next-day rainfall category + heavy-rain (flood-trigger) risk.

Run from the backend folder:
    python -m training.train_rainfall
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from app import config
from app.config import RAINFALL_MODEL_PATH, RANDOM_STATE
from app.ml_models import rainfall_model as rm
from app.utils.preprocessing import make_preprocessor
from training.common import chronological_split, describe_split, file_sha256, log, md_table, save_bundle, write_report

warnings.filterwarnings("ignore", message="Unknown solver options")

CLASSES = list(range(len(rm.CATEGORY_LABELS)))


def candidates(numeric=None):
    pre = make_preprocessor(rm.CATEGORICAL_FEATURES, numeric or rm.NUMERIC_FEATURES)
    return {
        "LogisticRegression": Pipeline([("preprocess", clone(pre)), ("model", LogisticRegression(max_iter=3000))]),
        "RandomForest": Pipeline([("preprocess", clone(pre)), ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE))]),
        "HistGradientBoosting": Pipeline([("preprocess", clone(pre)), ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE))]),
    }


def climatology_table(train: pd.DataFrame) -> pd.DataFrame:
    """Historical class frequencies per (city, month): the 'normal' expectation."""
    t = train.assign(month=train["date"].dt.month)
    return pd.crosstab([t["location"], t["month"]], t["target"], normalize="index").reindex(columns=CLASSES, fill_value=0.0)


def climatology_proba(table: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    keys = list(zip(frame["location"], frame["date"].dt.month))
    probs = table.reindex(keys).fillna(1.0 / len(CLASSES)).to_numpy()
    return probs / probs.sum(axis=1, keepdims=True)


def prob_metrics(y, proba) -> dict:
    proba = np.clip(proba, 1e-9, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    heavy = (y == rm.HEAVY_CLASS).astype(int)
    return {
        "log_loss": log_loss(y, proba, labels=CLASSES),
        "macro_f1": f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "heavy_pr_auc": average_precision_score(heavy, proba[:, rm.HEAVY_CLASS]),
        "heavy_roc_auc": roc_auc_score(heavy, proba[:, rm.HEAVY_CLASS]),
    }


def label_metrics(y, pred) -> dict:
    return {
        "macro_f1": f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
    }


def alert_metrics(y, flag) -> dict:
    heavy = (y == rm.HEAVY_CLASS).astype(int)
    return {
        "precision": precision_score(heavy, flag, zero_division=0),
        "recall": recall_score(heavy, flag, zero_division=0),
        "f1": f1_score(heavy, flag, zero_division=0),
        "alerts_raised": int(np.sum(flag)),
        "heavy_days": int(heavy.sum()),
    }


def best_threshold(y, p_heavy) -> float:
    precision, recall, thresholds = precision_recall_curve((y == rm.HEAVY_CLASS).astype(int), p_heavy)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def tune_combined_threshold(y, p_heavy, persist) -> float:
    """Threshold for "model OR persistence" with the best F1 (inf = the model adds nothing)."""
    best_t, best_f1 = float("inf"), alert_metrics(y, persist)["f1"]
    for t in np.unique(np.quantile(p_heavy, np.linspace(0.5, 1.0, 501))):
        f1 = alert_metrics(y, (p_heavy >= t) | persist)["f1"]
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t


def alert_flags(rule: str, p_heavy, persist, threshold: float, forecast=None):
    if rule == "model":
        return p_heavy >= threshold
    if rule == "persistence":
        return persist
    if rule == "forecast":
        return np.nan_to_num(np.asarray(forecast, dtype=float), nan=-1.0) >= rm.HEAVY_MM
    return (p_heavy >= threshold) | persist


def choose_alert_rule(y, p_heavy, lag_1, forecast=None) -> tuple:
    """Compare heavy-rain alert rules on the validation year; return (rule, thresholds, validation metrics).
    Ties go to the simpler rule (persistence, forecast, model, then the combination)."""
    persist = np.asarray(lag_1) >= rm.HEAVY_MM
    thresholds = {"model": best_threshold(y, p_heavy), "persistence": None}
    combined = tune_combined_threshold(y, p_heavy, persist)
    if np.isfinite(combined):
        thresholds["model_or_persistence"] = combined
    if forecast is not None:
        thresholds["forecast"] = None
    metrics = {rule: alert_metrics(y, alert_flags(rule, p_heavy, persist, t if t is not None else np.inf, forecast))
               for rule, t in thresholds.items()}
    order = ["persistence", "forecast", "model", "model_or_persistence"]
    best = max((r for r in order if r in metrics), key=lambda r: (round(metrics[r]["f1"], 9), -order.index(r)))
    return best, thresholds, metrics


def rule_metrics_on(y, p_heavy, lag_1, thresholds: dict, forecast=None) -> dict:
    persist = np.asarray(lag_1) >= rm.HEAVY_MM
    return {rule: alert_metrics(y, alert_flags(rule, p_heavy, persist, t if t is not None else np.inf, forecast)) for rule, t in thresholds.items()}


FAMILY_LABEL = {"past_rain": "past rain", "past_rain_forecast": "past rain + forecast"}
FAMILY_FEATURES = {"past_rain": (rm.FEATURES, rm.NUMERIC_FEATURES), "past_rain_forecast": (rm.FEATURES_FORECAST, rm.NUMERIC_FEATURES_FORECAST)}
MIN_FORECAST_TRAIN_ROWS = 5000      # about 5 months x 35 segments
MIN_FORECAST_VAL_SHARE = 0.8


def split_dates(frame: pd.DataFrame, with_forecast: bool = False) -> tuple:
    """Validation = the year before last, test = from 1 January of the last full year before the
    data ends onward. Uploaded city file (2016-2025): val 2023, test 2024-2025.
    Open-Meteo file (2020 - recent): val 2024, test 2025 - recent.
    With forecasts (archived from 2024): val = last year - 1, test = the last (current) year."""
    last_year = int(frame["date"].max().year)
    if with_forecast:
        return f"{last_year - 1}-01-01", f"{last_year}-01-01"
    return f"{last_year - 2}-01-01", f"{last_year - 1}-01-01"


CITY_LIMITATIONS = [
    "No flood labels exist in the data; heavy rain (>= 64.5 mm/day, IMD) is used as a flood trigger, not a flood observation.",
    "Daily city-level totals; the model cannot forecast hourly or segment-level rainfall.",
    "Heavy-rain days are rare (about 0.5% of days), so alerts have low precision.",
]
CORRIDOR_LIMITATIONS = [
    "No flood labels exist; heavy rain (>= 64.5 mm/day, IMD) is used as a flood trigger, not a flood observation.",
    "Rainfall is ERA5 / ERA5-Land reanalysis from Open-Meteo (a weather model fitted to observations, about 10-25 km grid), not rain-gauge readings. It smooths very local cloudbursts, so it records fewer heavy-rain days than a gauge would.",
    "Daily totals: the model forecasts the whole day, not individual hours.",
    "Heavy-rain days are rare, so alerts have low precision. The alert rule is chosen on the validation year; see the alert tables above.",
]
FORECAST_LIMITATIONS = [
    "Weather forecasts are archived only from January 2024, so the 'past rain + forecast' models learn from one year (2024) and are tested on the current year only; their comparison with past-rain-only models uses exactly the same test days.",
    "When no forecast made the day before exists for a date (e.g. dates far in the past or future), the app uses the past-rain-only fallback model, and says so.",
]


def dataset_description(path) -> str:
    if path.name == config.RAINFALL_CORRIDOR_DATASET.name:
        return (f"`{path.name}`: real daily rainfall at each of the 35 corridor segments, downloaded from the "
                "Open-Meteo Historical Weather API (ERA5 reanalysis). Weather data by Open-Meteo.com (CC BY 4.0).")
    return f"`{path.name}`: uploaded daily rainfall for 20 Indian cities."


def load_forecasts(corridor: bool):
    path = config.RAIN_FORECAST_DATASET
    if not corridor or not path.exists():
        return None
    df = pd.read_csv(path, usecols=["Date", "Location", "Forecast_mm"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.rename(columns={"Date": "date", "Location": "location", "Forecast_mm": rm.FORECAST_FEATURE})


def class_proba(pipeline, frame: pd.DataFrame) -> np.ndarray:
    raw = pipeline.predict_proba(frame)
    out = np.zeros((len(frame), len(CLASSES)))
    for i, c in enumerate(pipeline.classes_):
        out[:, int(c)] = raw[:, i]
    return out


def main() -> None:
    dataset = config.RAINFALL_DATASET
    corridor = dataset.name == config.RAINFALL_CORRIDOR_DATASET.name
    log.info("Loading %s", dataset.name)
    raw = pd.read_csv(dataset)
    frame = rm.build_training_frame(raw)

    # Optional forecast input: join "forecast made the day before" by (segment, day).
    forecasts = load_forecasts(corridor)
    frame[rm.FORECAST_FEATURE] = np.nan
    if forecasts is not None:
        frame = frame.drop(columns=[rm.FORECAST_FEATURE]).merge(forecasts, on=["location", "date"], how="left")
    has_fc = frame[rm.FORECAST_FEATURE].notna()
    use_forecast = False
    if forecasts is not None:
        val_start, test_start = split_dates(frame, with_forecast=True)
        in_train = has_fc & (frame["date"] < val_start)
        in_val = (frame["date"] >= val_start) & (frame["date"] < test_start)
        use_forecast = in_train.sum() >= MIN_FORECAST_TRAIN_ROWS and has_fc[in_val].mean() >= MIN_FORECAST_VAL_SHARE
        log.info("Forecast file found: %d training rows and %.0f%% of validation rows have a forecast -> %s",
                 in_train.sum(), 100 * has_fc[in_val].mean(), "compare with and without forecast" if use_forecast else "not enough, past rain only")
    val_start, test_start = split_dates(frame, with_forecast=use_forecast)
    train, val, test = chronological_split(frame, "date", val_start, test_start)
    split = describe_split(train, val, test, "date")
    log.info("Split: %s", split)

    # Every candidate is scored on the same validation days (those with a forecast, when forecasts are used).
    families = ["past_rain"] + (["past_rain_forecast"] if use_forecast else [])
    eval_val = val[val[rm.FORECAST_FEATURE].notna()] if use_forecast else val
    y_val = eval_val["target"].to_numpy()

    # 1) Compare algorithms (x input sets) on the validation year (trained on train years only).
    comparison, fitted = {}, {}
    for family in families:
        feats, numeric = FAMILY_FEATURES[family]
        part = train if family == "past_rain" else train[train[rm.FORECAST_FEATURE].notna()]
        for name, pipe in candidates(numeric).items():
            key = f"{name} · {FAMILY_LABEL[family]}" if use_forecast else name
            log.info("Training %s on %d rows ...", key, len(part))
            pipe.fit(part[feats], part["target"])
            comparison[key] = prob_metrics(y_val, class_proba(pipe, eval_val[feats]))
            fitted[key] = (family, name, pipe)
            log.info("  validation: %s", {k: round(v, 4) for k, v in comparison[key].items()})
    # Lowest log loss = most reliable probabilities (proper scoring rule).
    best_key = min(comparison, key=lambda k: comparison[k]["log_loss"])
    best_family, best_name, _ = fitted[best_key]
    fallback_key = min((k for k in comparison if fitted[k][0] == "past_rain"), key=lambda k: comparison[k]["log_loss"])
    _, fallback_name, _ = fitted[fallback_key]

    # 2) Alert rule per profile, chosen on validation.
    def profile_rules(key, rows, allow_forecast_rule):
        family, _, pipe = fitted[key]
        p = class_proba(pipe, rows[FAMILY_FEATURES[family][0]])[:, rm.HEAVY_CLASS]
        fc = rows[rm.FORECAST_FEATURE].to_numpy() if allow_forecast_rule else None
        return choose_alert_rule(rows["target"].to_numpy(), p, rows["lag_1"].to_numpy(), fc)

    alert_rule, rule_thresholds, val_alert = profile_rules(best_key, eval_val, use_forecast)
    model_threshold = rule_thresholds["model"]
    threshold = rule_thresholds.get(alert_rule) or model_threshold   # rules without a threshold: model threshold only scales the rain component
    primary_needs_forecast = use_forecast and (best_family == "past_rain_forecast" or alert_rule == "forecast")
    log.info("Selected %s; heavy-rain alert rule chosen on validation: %s (F1 %s)", best_key, alert_rule,
             {r: round(m["f1"], 3) for r, m in val_alert.items()})
    if primary_needs_forecast:
        fb_rule, fb_thresholds, fb_val_alert = profile_rules(fallback_key, val, False)
        fb_threshold = fb_thresholds.get(fb_rule) or fb_thresholds["model"]
        log.info("Fallback when no forecast exists: %s, alert rule %s", fallback_key, fb_rule)

    # 3) Refit on train + validation, evaluate once on the untouched test period.
    train_val = pd.concat([train, val])

    def refit(family, name):
        feats, numeric = FAMILY_FEATURES[family]
        rows = train_val if family == "past_rain" else train_val[train_val[rm.FORECAST_FEATURE].notna()]
        return candidates(numeric)[name].fit(rows[feats], rows["target"])

    final = refit(best_family, best_name)
    feats = FAMILY_FEATURES[best_family][0]
    eval_test = test[test[rm.FORECAST_FEATURE].notna()] if use_forecast else test
    y_test = eval_test["target"].to_numpy()
    test_proba = class_proba(final, eval_test[feats])
    test_pred = test_proba.argmax(axis=1)
    model_metrics = prob_metrics(y_test, test_proba)
    test_fc = eval_test[rm.FORECAST_FEATURE].to_numpy() if use_forecast else None
    test_alerts = rule_metrics_on(y_test, test_proba[:, rm.HEAVY_CLASS], eval_test["lag_1"].to_numpy(), rule_thresholds, test_fc)
    model_alert = test_alerts[alert_rule]
    log.info("Test heavy-rain alert (%s): precision %.3f, recall %.3f", alert_rule, model_alert["precision"], model_alert["recall"])

    fallback_final = None
    fallback_report = None
    if primary_needs_forecast:
        fallback_final = refit("past_rain", fallback_name)
        fb_same = class_proba(fallback_final, eval_test[rm.FEATURES])
        fb_all = class_proba(fallback_final, test[rm.FEATURES])
        fallback_report = {
            "model": fallback_key, "alert_rule": fb_rule, "alert_threshold": fb_threshold, "alert_rule_thresholds": fb_thresholds,
            "validation_alert_rules": fb_val_alert,
            "test_metrics_same_days": prob_metrics(y_test, fb_same),
            "test_metrics_all_days": prob_metrics(test["target"].to_numpy(), fb_all),
            "test_alert_rules_all_days": rule_metrics_on(test["target"].to_numpy(), fb_all[:, rm.HEAVY_CLASS], test["lag_1"].to_numpy(), fb_thresholds),
        }
    elif use_forecast and best_family == "past_rain":
        log.info("Past rain only won on validation; forecasts are not used by the model.")

    # 4) Baselines on the same test days.
    clim = climatology_table(train_val)
    baselines = {
        "climatology (location x month frequencies)": prob_metrics(y_test, climatology_proba(clim, eval_test)),
        "persistence (tomorrow = today)": label_metrics(y_test, rm.rainfall_to_category(eval_test["lag_1"])),
    }
    if use_forecast:
        baselines["forecast alone (category of the forecast made the day before)"] = label_metrics(y_test, rm.rainfall_to_category(eval_test[rm.FORECAST_FEATURE]))
        if fallback_report is not None:
            baselines[f"{fallback_key} (fallback, same days)"] = fallback_report["test_metrics_same_days"]
        elif best_family == "past_rain_forecast":
            pass
    persistence_alert = test_alerts["persistence"]

    limitations = (CORRIDOR_LIMITATIONS + (FORECAST_LIMITATIONS if use_forecast else [])) if corridor else CITY_LIMITATIONS
    report = {
        "model": "rainfall",
        "task": "Next-day IMD rainfall category; heavy-rain probability as flood-trigger risk",
        "dataset": dataset.name if not use_forecast else [dataset.name, config.RAIN_FORECAST_DATASET.name],
        "dataset_description": dataset_description(dataset) + (
            f" Plus `{config.RAIN_FORECAST_DATASET.name}`: the rain forecast for each day made the day before (Open-Meteo Previous Runs API)." if use_forecast else ""),
        "locations": int(raw["Location"].nunique()),
        "split": split,
        "uses_forecast": bool(primary_needs_forecast),
        "forecast_rows": {"train": int(train[rm.FORECAST_FEATURE].notna().sum()), "validation": int(len(eval_val)), "test": int(len(eval_test))} if use_forecast else None,
        "selection_metric": "validation log loss (lower is better)",
        "validation_comparison": comparison,
        "selected_model": best_key,
        "heavy_alert_threshold": threshold,
        "heavy_alert_rule": alert_rule,
        "heavy_alert_rule_text": rm.ALERT_RULE_TEXT[alert_rule],
        "alert_rule_thresholds": rule_thresholds,
        "validation_alert_rules": val_alert,
        "test_alert_rules": test_alerts,
        "test_metrics": model_metrics,
        "test_heavy_alert": model_alert,
        "test_baselines": baselines,
        "test_baseline_persistence_heavy_alert": persistence_alert,
        "fallback": fallback_report,
        "test_confusion_matrix": {"labels": rm.CATEGORY_LABELS, "matrix": confusion_matrix(y_test, test_pred, labels=CLASSES).tolist()},
        "test_classification_report": classification_report(y_test, test_pred, labels=CLASSES, target_names=rm.CATEGORY_LABELS, output_dict=True, zero_division=0),
        "limitations": limitations,
    }

    rows = [[n, f"{m['log_loss']:.3f}", f"{m['macro_f1']:.3f}", f"{m['heavy_pr_auc']:.3f}"] for n, m in comparison.items()]
    trow = lambda n, m: [n, f"{m.get('log_loss', float('nan')):.3f}" if "log_loss" in m else "-", f"{m['macro_f1']:.3f}", f"{m['accuracy']:.3f}", f"{m['heavy_pr_auc']:.3f}" if "heavy_pr_auc" in m else "-"]
    rule_rows = lambda metrics, thresholds, chosen: [[f"{rule}{' (selected)' if rule == chosen else ''}", "-" if thresholds[rule] is None else f"{thresholds[rule]:.4f}",
                                                       f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", m['alerts_raised'], m['heavy_days']] for rule, m in metrics.items()]
    rule_head = ["Rule", "Threshold", "Precision", "Recall", "F1", "Alerts raised", "Actual heavy days"]
    parts = [
        "# Model 1 - Rainfall / heavy-rain risk: evaluation report",
        f"Dataset: {report['dataset_description']}",
        f"Split: train {split['train']['from']}..{split['train']['to']}, "
        f"validation {split['validation']['from']}..{split['validation']['to']}, test {split['test']['from']}..{split['test']['to']}",
    ]
    if use_forecast:
        parts.append(f"Forecast input available for {report['forecast_rows']['train']:,} training rows (2024 only); all models are scored on the same "
                     f"{len(eval_val):,} validation and {len(eval_test):,} test segment-days that have a forecast made the day before.")
    parts += [
        "## Algorithm comparison (validation year)",
        md_table(rows, ["Model", "Log loss", "Macro-F1", "Heavy-rain PR-AUC"]),
        f"Selected: **{best_key}** (lowest validation log loss).",
        "## Test period - model vs baselines (same test days)",
        md_table([trow(best_key, model_metrics)] + [trow(n, m) for n, m in baselines.items()], ["Model", "Log loss", "Macro-F1", "Accuracy", "Heavy-rain PR-AUC"]),
        "## Heavy-rain alert rule (chosen on the validation year by F1)",
        md_table(rule_rows(val_alert, rule_thresholds, alert_rule), rule_head),
        f"Selected alert rule: **{alert_rule}** = {rm.ALERT_RULE_TEXT[alert_rule]}.",
        "## Heavy-rain alert on the test period (rules fixed on validation, test seen once)",
        md_table(rule_rows(test_alerts, rule_thresholds, alert_rule), rule_head),
    ]
    if fallback_report is not None:
        parts += [
            "## Fallback when no forecast exists for a date",
            f"Model **{fallback_key}**, alert rule **{fb_rule}** (chosen on validation). On all {len(test):,} test segment-days:",
            md_table([trow(fallback_key, fallback_report["test_metrics_all_days"])], ["Model", "Log loss", "Macro-F1", "Accuracy", "Heavy-rain PR-AUC"]),
            md_table(rule_rows(fallback_report["test_alert_rules_all_days"], fb_thresholds, fb_rule), rule_head),
        ]
    parts.append("## Limitations\n" + "\n".join(f"- {item}" for item in limitations))
    write_report("rainfall", report, "\n\n".join(parts))

    bundle = {
        "pipeline": final,
        "model_name": best_key,
        "features": feats,
        "classes": rm.CATEGORY_LABELS,
        "heavy_class": rm.HEAVY_CLASS,
        "heavy_alert_threshold": threshold,
        "heavy_alert_rule": alert_rule,
        "heavy_alert_model_threshold": model_threshold,
        "uses_forecast": bool(primary_needs_forecast),
        "climatology_heavy": {f"{loc}|{month}": float(p) for (loc, month), p in clim[rm.HEAVY_CLASS].items()},
        "metrics": {"test": model_metrics, "test_heavy_alert": model_alert, "baselines": baselines, "split": split},
        "input_options": {"location": rm.known_locations(raw)},
        "dataset": dataset.name,
        "dataset_description": report["dataset_description"],
        "dataset_sha256": file_sha256(dataset),
    }
    if fallback_final is not None:
        bundle["fallback"] = {"pipeline": fallback_final, "model_name": fallback_key, "features": rm.FEATURES,
                              "heavy_alert_threshold": fb_threshold, "heavy_alert_rule": fb_rule}
    if use_forecast:
        bundle["forecast_dataset"] = config.RAIN_FORECAST_DATASET.name
    save_bundle(bundle, RAINFALL_MODEL_PATH)


if __name__ == "__main__":
    main()
