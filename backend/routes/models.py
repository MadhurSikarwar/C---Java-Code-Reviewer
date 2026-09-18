"""
IntelliReview — /api/models
Serves what the site says about the models: their descriptions plus the *measured* evaluation written by
ml/retrain_all.py (ml/model_report.json). The UI never hard-codes accuracy claims.
"""
import json
import os

from fastapi import APIRouter

router = APIRouter()

REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "model_report.json")

MODELS = [
    {"id": "v3", "short": "V3", "name": "Path-sensitive forest", "kind": "Random forest",
     "features": 34, "inputs": "classic + path-sensitive (pointer state, data flow, CFG)",
     "note": "Sees what the pointer and data-flow analysis found, so it is the most direct read of a memory bug."},
    {"id": "dl", "short": "DL V1", "name": "Neural net, classic features", "kind": "Small neural network",
     "features": 21, "inputs": "structure, size, complexity, code-quality signals",
     "note": "Only sees coarse metrics (no pointer analysis), so it is a useful second opinion."},
    {"id": "ensemble", "short": "Ensemble", "name": "Boosted ensemble", "kind": "LightGBM + XGBoost + extra trees",
     "features": 21, "inputs": "the same 21 classic features",
     "note": "Three tree models voting; stable on unusual inputs."},
    {"id": "v4_dl", "short": "V4", "name": "Neural net, all features", "kind": "Small neural network",
     "features": 36, "inputs": "everything above plus how many distinct paths reach each finding and how severe they are",
     "note": "Knows the most, and is the one most inclined to trust the rule engine's evidence."},
]


@router.get("/models")
def models():
    report = None
    try:
        with open(REPORT_PATH, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        pass
    return {"models": MODELS, "report": report}
