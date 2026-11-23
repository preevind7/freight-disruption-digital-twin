"""Reusable preprocessing blocks for the three pipelines."""
from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(categorical: List[str], numeric: List[str], scale_numeric: bool = True) -> ColumnTransformer:
    """One-hot encode categoricals (unknown values ignored) and impute/scale numerics.

    Scaling is needed for linear models; tree models are unaffected by it, so the
    same preprocessor can be used for every algorithm we compare.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            ("numeric", Pipeline(numeric_steps), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
