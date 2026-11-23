"""Helpers shared by the three training scripts."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn

from app.config import REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("training")


def chronological_split(df: pd.DataFrame, date_col: str, val_start: str, test_start: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train < val_start <= validation < test_start <= test. No shuffling: the model is
    always evaluated on a period that comes after everything it was trained on."""
    dates = pd.to_datetime(df[date_col])
    train = df[dates < val_start]
    val = df[(dates >= val_start) & (dates < test_start)]
    test = df[dates >= test_start]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("A split is empty - check the split dates against the dataset range.")
    return train, val, test


def describe_split(train, val, test, date_col: str) -> Dict[str, Any]:
    def span(part):
        d = pd.to_datetime(part[date_col])
        return {"rows": int(len(part)), "from": d.min().strftime("%Y-%m-%d"), "to": d.max().strftime("%Y-%m-%d")}

    return {"train": span(train), "validation": span(val), "test": span(test)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 4)
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    return value


def save_bundle(bundle: Dict[str, Any], path: Path) -> None:
    bundle = dict(bundle)
    bundle.setdefault("trained_at", now_iso())
    bundle.setdefault("sklearn_version", sklearn.__version__)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    log.info("Saved model bundle -> %s (%.1f KB)", path, path.stat().st_size / 1024)


def write_report(name: str, report: Dict[str, Any], markdown: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{name}_report.json").write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    (REPORTS_DIR / f"{name}_report.md").write_text(markdown, encoding="utf-8")
    log.info("Wrote reports/%s_report.json and .md", name)


def md_table(rows, headers) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)
