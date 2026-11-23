import json

from fastapi import APIRouter, HTTPException

from app import config
from app.ml_models.registry import registry

public_router = APIRouter(tags=["System"])
router = APIRouter(tags=["System"])

MODEL_PATHS = {
    "rainfall": config.RAINFALL_MODEL_PATH,
    "accident": config.ACCIDENT_MODEL_PATH,
    "traffic": config.TRAFFIC_MODEL_PATH,
}


def load_all_models() -> None:
    for name, path in MODEL_PATHS.items():
        registry.load(name, path)


@public_router.get("/api/health")
def health():
    """Server status and whether each of the three models is loaded."""
    models = {name: registry.status(name) for name in MODEL_PATHS}
    all_loaded = all(m["loaded"] for m in models.values())
    return {"status": "ok" if all_loaded else "degraded", "version": config.APP_VERSION, "models": models}


@router.get("/api/models")
def models_overview():
    """Metrics, algorithm and allowed input values for every model (used by the frontend)."""
    out = {}
    for name in MODEL_PATHS:
        status = registry.status(name)
        entry = {"status": status}
        if status["loaded"]:
            bundle = registry.get(name)
            entry.update(
                algorithm=bundle.get("model_name"),
                dataset=bundle.get("dataset"),
                input_options=bundle.get("input_options"),
                metrics=bundle.get("metrics"),
            )
        report_path = config.REPORTS_DIR / f"{name}_report.json"
        if report_path.exists():
            entry["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        out[name] = entry
    return out


@router.post("/api/models/reload")
def reload_models():
    """Reload model files from disk after re-training, without restarting the server."""
    load_all_models()
    from app.services import corridor_service
    corridor_service.clear_cache()
    return health()


@router.get("/api/models/{name}")
def model_detail(name: str):
    if name not in MODEL_PATHS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'. Use one of: {', '.join(MODEL_PATHS)}.")
    return models_overview()[name]
