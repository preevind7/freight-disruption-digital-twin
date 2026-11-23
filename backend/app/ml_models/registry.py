"""Loads trained model bundles once and hands them to the services.

A bundle is a dict saved with joblib by the training scripts. It always holds:
    pipeline        fitted scikit-learn Pipeline (preprocessing + estimator)
    model_name      algorithm that won on the validation set
    features        ordered list of input feature columns
    metrics         evaluation results on the held-out test period
    trained_at      ISO timestamp
    sklearn_version version used for training
plus model-specific extras (classes, thresholds, input options...).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import sklearn

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when a model file is missing or could not be loaded."""


class ModelRegistry:
    def __init__(self) -> None:
        self._bundles: Dict[str, Dict[str, Any]] = {}
        self._errors: Dict[str, str] = {}
        self._paths: Dict[str, Path] = {}

    def load(self, name: str, path: Path) -> None:
        self._paths[name] = path
        self._bundles.pop(name, None)
        self._errors.pop(name, None)
        if not path.exists():
            self._errors[name] = (
                f"Model file not found: {path.name}. "
                f"Train it with: python -m training.train_{name}"
            )
            logger.warning(self._errors[name])
            return
        try:
            bundle = joblib.load(path)
            if not isinstance(bundle, dict) or "pipeline" not in bundle:
                raise ValueError("file is not a valid model bundle")
        except Exception as exc:  # corrupted file, version mismatch, ...
            self._errors[name] = (
                f"Model file {path.name} could not be loaded ({exc}). "
                f"Re-train it with: python -m training.train_{name}"
            )
            logger.error(self._errors[name])
            return
        trained_version = bundle.get("sklearn_version")
        if trained_version and trained_version != sklearn.__version__:
            logger.warning(
                "%s was trained with scikit-learn %s but %s is installed; re-train to be safe.",
                name, trained_version, sklearn.__version__,
            )
        self._bundles[name] = bundle
        logger.info("Loaded %s model (%s)", name, bundle.get("model_name"))

    def get(self, name: str) -> Dict[str, Any]:
        if name in self._bundles:
            return self._bundles[name]
        raise ModelNotAvailableError(self._errors.get(name, f"Model '{name}' is not loaded."))

    def status(self, name: str) -> Dict[str, Optional[str]]:
        bundle = self._bundles.get(name)
        return {
            "loaded": bundle is not None,
            "model_name": bundle.get("model_name") if bundle else None,
            "trained_at": bundle.get("trained_at") if bundle else None,
            "error": self._errors.get(name),
        }

    def names(self):
        return list(self._paths.keys())


registry = ModelRegistry()
