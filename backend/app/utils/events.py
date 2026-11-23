"""Festival / significant-date lookup shared by training and serving (traffic model)."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

NO_EVENT = "None"
OTHER_EVENT = "Other event"
# Festival names used in indian_roads_dataset.csv, and words that identify them
# in significant_dates.csv. Order matters: first match wins.
FESTIVAL_KEYWORDS = [
    ("Diwali", ("diwali", "deepavali")),
    ("Holi", ("holi",)),
    ("Eid", ("eid", "bakrid")),
    ("New Year", ("new year",)),
]


def normalise_event(text: Optional[str]) -> str:
    """Map a free-text event name to one of the festival categories."""
    if text is None or (isinstance(text, float) and pd.isna(text)) or not str(text).strip():
        return NO_EVENT
    lowered = str(text).lower()
    for label, words in FESTIVAL_KEYWORDS:
        if any(word in lowered for word in words):
            return label
    return OTHER_EVENT


@lru_cache(maxsize=4)
def load_significant_dates(path: Path) -> Dict[str, str]:
    """Returns {'YYYY-MM-DD': 'event text'}; empty dict if the file is missing."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df.groupby("Date")["Event"].apply(lambda s: "; ".join(s.astype(str))).to_dict()


def event_on(day: date, path: Path) -> Optional[str]:
    return load_significant_dates(path).get(day.strftime("%Y-%m-%d"))
