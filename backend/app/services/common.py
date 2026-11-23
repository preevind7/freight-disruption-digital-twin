from __future__ import annotations

from typing import Any, Dict

from app.schemas.common import ModelInfo


def model_info(name: str, bundle: Dict[str, Any]) -> ModelInfo:
    return ModelInfo(name=name, algorithm=str(bundle.get("model_name")), trained_at=bundle.get("trained_at"))
