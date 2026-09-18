"""
IntelliReview — ML Predictor
Loads the saved model and runs inference to produce a risk label and score.
"""
import os
import sys
import threading
import joblib
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, ENCODER_PATH, RISK_CLEAN_MAX, RISK_MODERATE_MAX
from ml.synthetic_data import FEATURE_NAMES   # single source of truth

DL_MODEL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_dl_v1_19feat.keras")
DL_SCALER_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_scaler_v1_19feat.pkl")
DL_V4_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_v4_dedup_massive.keras")
DL_V4_SCALER_PATH= os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4_scaler.pkl")

_model     = None  # sklearn stacking ensemble (19 features)
_dl_model  = None  # Keras DL V1 (19 features)
_dl_scaler = None
_v3_model  = None  # RF path-sensitive (34 features)
_v4_model  = None  # Keras DL V4 dedup (36 features)
_v4_scaler = None
_label_names = ["Clean", "Moderate Risk", "High Risk"]
_lock = threading.RLock()   # Keras / sklearn model objects are shared between request threads

from analyzers.issue_taxonomy import NON_SECURITY_TYPES as _NON_SECURITY_TYPES, QUALITY_TYPES as _QUALITY_TYPES


def _load_gate() -> str:
    """Which evidence-gate variant retrain_all.py found best ('high' or 'full'); see model_report.json."""
    try:
        import json
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_report.json"), encoding="utf-8") as f:
            g = json.load(f).get("gate", "full")
        return g if g in ("high", "full", "calm") else "full"
    except Exception:
        return "full"


_GATE = _load_gate()


def _as_frame(model, X):
    """sklearn warns when a model fitted on a DataFrame is called with a bare ndarray."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) == X.shape[1]:
        import pandas as pd
        return pd.DataFrame(X, columns=list(names))
    return X


def load_model():
    """Load the ML model from disk (called once on startup). Builds the models first on a fresh checkout."""
    global _model
    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print("⚠️  No trained models found (they are not stored in git) — building them now...")
        _train_model()

    _model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")


def _train_model():
    """Build the models with the honest pipeline (ml/bootstrap.py) — never the old size-only synthetic trainer."""
    from ml.bootstrap import bootstrap
    bootstrap()


def load_dl_model():
    """Load the deep learning model and its scaler from disk."""
    global _dl_model, _dl_scaler
    if _dl_model is not None and _dl_scaler is not None:
        return

    if not os.path.exists(DL_MODEL_PATH):
        print("⚠️  DL Model not found — please run `train_dl_model.py` first.")
        return

    import logging
    # Suppress verbose TF warnings
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        # Python 3.8+ Windows workaround for CUDA DLLs
        if os.name == 'nt':
            try:
                os.add_dll_directory(r"C:\CUDA_Manual\bin")
            except Exception:
                pass

        from tensorflow.keras.models import load_model as keras_load
        _dl_model = keras_load(DL_MODEL_PATH)
        _dl_scaler = joblib.load(DL_SCALER_PATH)
        print(f"✅ Deep Learning Model loaded from {DL_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load DL model: {e}")


def load_v3_model():
    """Load the V3 Path Sensitive ML model from disk."""
    global _v3_model
    if _v3_model is not None:
        return

    v3_path = os.path.join(os.path.dirname(__file__), "model_v3_path_sensitive.pkl")
    if not os.path.exists(v3_path):
        print("⚠️  V3 model.pkl not found. Please train it.")
        return

    _v3_model = joblib.load(v3_path)
    print(f"✅ V3 Model loaded from {v3_path}")


def load_v4_dl_model():
    """Load the V4 36-feature Dedup DL model and its scaler."""
    global _v4_model, _v4_scaler
    if _v4_model is not None:
        return

    if not os.path.exists(DL_V4_MODEL_PATH):
        print(f"⚠️  V4 DL model not found at {DL_V4_MODEL_PATH}. Run v4_train_dedup_dl.py first.")
        return

    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        if os.name == 'nt':
            try:
                os.add_dll_directory(r"C:\CUDA_Manual\bin")
            except Exception:
                pass
        from tensorflow.keras.models import load_model as keras_load
        _v4_model = keras_load(DL_V4_MODEL_PATH)
        if os.path.exists(DL_V4_SCALER_PATH):
            _v4_scaler = joblib.load(DL_V4_SCALER_PATH)
        else:
            print("⚠️  V4 scaler not found — will use unscaled input.")
        print(f"✅ V4 DL model loaded from {DL_V4_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load V4 DL model: {e}")


# Features that are *findings* (as opposed to the shape of the code). Zeroing them answers "what would this model say
# from the structure of the code alone?" — a counterfactual that shows how much the verdict leans on the evidence.
EVIDENCE_IDX = [7, 8, 13, 14, 21, 22, 23, 25, 26, 34, 35]
_MODEL_TITLE = {"v3": "V3 forest", "dl": "DL V1 net", "ensemble": "Boosted ensemble", "v4_dl": "V4 net"}


def _pct(p):
    return {"Clean": round(p[0] * 100), "Moderate": round(p[1] * 100), "High": round(p[2] * 100)}


def predict_risk(feature_vector: List[float], model_type: str = "v3", issues: List[Dict[str, Any]] = None,
                 trace=None) -> Dict[str, Any]:
    """Thread-safe entry point (see `_predict_risk`). `trace(event_dict)` receives the model's reasoning as it happens."""
    with _lock:
        return _predict_risk(feature_vector, model_type, issues, trace)


def _pad(X, n):
    if X.shape[1] < n:
        return np.hstack((X, np.zeros((1, n - X.shape[1]))))
    return X[:, :n]


def _raw_proba(m_type: str, X_full):
    """Class probabilities [clean, moderate, high] straight from one model (no gate), or None if it isn't available."""
    global _dl_model, _dl_scaler, _v3_model, _v4_model, _v4_scaler, _model
    if m_type == "dl":
        if _dl_model is None:
            load_dl_model()
        if _dl_model is None:
            return None
        n = len(_dl_scaler.mean_) if hasattr(_dl_scaler, "mean_") else 19
        return _dl_model.predict(_dl_scaler.transform(_pad(X_full, n)), verbose=0)[0].tolist()
    if m_type == "v4_dl":
        if _v4_model is None:
            load_v4_dl_model()
        if _v4_model is None:
            return None
        n = len(_v4_scaler.mean_) if (_v4_scaler is not None and hasattr(_v4_scaler, "mean_")) else 36
        X = _pad(X_full, n)
        if _v4_scaler is not None:
            X = _v4_scaler.transform(X)
        return _v4_model.predict(X, verbose=0)[0].tolist()
    if m_type == "v3":
        if _v3_model is None:
            load_v3_model()
        if _v3_model is None:
            return None
        X = _pad(X_full, getattr(_v3_model, "n_features_in_", 34))
        return _v3_model.predict_proba(_as_frame(_v3_model, X))[0].tolist()
    if m_type == "ensemble":
        if _model is None:
            load_model()
        if _model is None:
            return None
        return _model.predict_proba(_as_frame(_model, _pad(X_full, 21)))[0].tolist()
    return None


def _predict_risk(feature_vector: List[float], model_type: str = "v3", issues: List[Dict[str, Any]] = None,
                  trace=None) -> Dict[str, Any]:
    """
    Given a feature vector, return risk label, score 0–100, and per-class probabilities.
    model_type: 'dl', 'v3', 'ensemble', 'v4_dl', or 'all'.

    The rule engine's evidence gates the model: a model alone may not call code "High Risk" unless a
    critical/high-severity *security* finding backs it up, and such a finding always makes the verdict High. With the
    "full" gate variant (chosen by ml/retrain_all.py when it measures better) medium findings also rule out "Clean". (Models trained on code-size statistics used to call every large, correct program High Risk.)
    """
    ALL_MODELS = ["ensemble", "dl", "v3", "v4_dl"]
    models_to_run = ALL_MODELS if model_type == "all" else [model_type]
    results = {}

    X_full = np.array(feature_vector).reshape(1, -1)
    
    emit = trace or (lambda e: None)

    for m_type in models_to_run:
        proba = _raw_proba(m_type, X_full)
        if proba is None:
            continue
        label_idx = int(np.argmax(proba))
        raw_label_idx = label_idx
        n_inputs = {"dl": 21, "ensemble": 21, "v3": 34, "v4_dl": 36}.get(m_type, len(feature_vector))
        emit({"stage": "models", "kind": "model", "model": m_type,
              "text": f"{_MODEL_TITLE.get(m_type, m_type)} reads {n_inputs} numbers and leans {_label_names[label_idx]}.",
              "probs": _pct(proba)})
        # counterfactual: the same model with every finding-derived feature set to zero
        if any(float(feature_vector[i]) != 0.0 for i in EVIDENCE_IDX if i < len(feature_vector)):
            X_alt = np.array(feature_vector, dtype=float).copy()
            for i in EVIDENCE_IDX:
                if i < len(X_alt):
                    X_alt[i] = 0.0
            alt = _raw_proba(m_type, X_alt.reshape(1, -1))
            if alt is not None:
                emit({"stage": "models", "kind": "counterfactual", "model": m_type,
                      "text": f"From the shape of the code alone, with the findings removed, it would say "
                              f"{_label_names[int(np.argmax(alt))]}.", "probs": _pct(alt)})

        # --- EVIDENCE GATING & CONFIDENCE CALIBRATION ---
        confidence = float(np.max(proba) * 100)
        explanations = []
        is_hard_override = False

        if len(feature_vector) > 29 and feature_vector[29] > 20:  # V3 cyclomatic complexity
            confidence -= 10.0
            explanations.append("Very high cyclomatic complexity limits how reliably branch behaviour can be predicted.")

        issues = issues or []
        high_evidence = [i for i in issues
                         if i.get("severity") in ("CRITICAL", "HIGH") and i.get("type") not in _NON_SECURITY_TYPES]
        any_evidence = [i for i in issues
                        if i.get("severity") in ("CRITICAL", "HIGH", "MEDIUM") and i.get("type") not in _QUALITY_TYPES]

        if high_evidence and label_idx < 2:
            label_idx = 2
            proba = [0.0, 0.1, 0.9]
            confidence = 100.0
            is_hard_override = True
            explanations.append("Raised to HIGH RISK: the rule engine found critical/high-severity defects "
                                "(" + ", ".join(sorted({i.get("type", "?") for i in high_evidence})) + ").")
        elif not high_evidence and label_idx == 2:
            label_idx = 1
            proba = [proba[0], proba[1] + proba[2], 0.0]
            explanations.append("Capped at MODERATE RISK: the model reacted to size/complexity statistics, but no "
                                "critical or high-severity defect was found.")
        if _GATE == "calm" and not any_evidence and not high_evidence and label_idx == 1:
            label_idx = 0
            proba = [proba[0] + proba[1], 0.0, proba[2]]
            explanations.append("Kept Clean: the model leaned Moderate, but nothing concrete was found to point at.")
        if _GATE != "full" or (not any_evidence and not high_evidence):
            pass
        elif label_idx == 0:
            label_idx = 1
            proba = [0.0, max(proba[1], 0.5), proba[2]]
            explanations.append("Raised to MODERATE RISK: medium-severity findings were reported.")

        gate_notes = [x for x in explanations if not x.startswith("Very high cyclomatic")]
        for note in gate_notes:
            note = note.replace("HIGH RISK", "High risk").replace("MODERATE RISK", "Moderate risk")
            emit({"stage": "gate", "kind": "gate", "model": m_type, "text": note, "to": _label_names[label_idx]})
        if not gate_notes:
            emit({"stage": "gate", "kind": "gate", "model": m_type, "to": _label_names[label_idx],
                  "text": f"The evidence gate checked the findings against the model: no change, still {_label_names[label_idx]}."})

        label = _label_names[label_idx]
        risk_score = _compute_risk_score(label_idx, proba, feature_vector[:21], m_type)
        if is_hard_override:
            risk_score = max(risk_score, 95)
        elif not explanations:
            explanations.append("Prediction bounds aligned successfully with nominal variance.")
            
        results[m_type] = {
            "risk_label": label,
            "risk_score": risk_score,
            "confidence": round(confidence, 1),
            "explanations": explanations,
            "probabilities": {
                "Clean": round(proba[0] * 100, 1),
                "Moderate Risk": round(proba[1] * 100, 1),
                "High Risk": round(proba[2] * 100, 1),
            },
        }

    if model_type != "all" and model_type in results:
        return results[model_type]
        
    # Return bundle for 'all' mode
    primary_key = "ensemble" if "ensemble" in results else list(results.keys())[0] if results else None
    if not primary_key:
        return {"risk_label": "Clean", "risk_score": 0, "confidence": 0, "explanations": [], "probabilities": {}, "comparisons": {}}
        
    primary = results[primary_key]
    return {
        "risk_label": primary["risk_label"],
        "risk_score": primary["risk_score"],
        "confidence": primary["confidence"],
        "explanations": primary["explanations"],
        "probabilities": primary["probabilities"],
        "comparisons": results
    }


def _compute_risk_score(label_idx: int, proba: List[float], features: List[float], model_type: str = "ensemble") -> int:
    """
    Map class probabilities + label to a 0-100 risk score.
    Uses weighted combination of class-specific probability bands.
    """
    # Base range per class
    ranges = [
        (0, 35),   # Clean
        (36, 65),  # Moderate
        (66, 100), # High Risk
    ]

    # Primary band from predicted class
    lo, hi = ranges[label_idx]

    # Fine-grained position within band using probability of this class
    p = proba[label_idx]
    base_score = lo + int((hi - lo) * p)

    # Boost score for specific high-weight features safely
    limit = min(len(FEATURE_NAMES), len(features))
    feat = dict(zip(FEATURE_NAMES[:limit], features[:limit]))
    
    adjustments = 0
    adjustments += min(feat.get("memory_leak_count", 0) * 4, 12)
    adjustments += min(feat.get("unsafe_function_count", 0) * 3, 9)
    adjustments += min(feat.get("max_nesting_depth", 0) * 2, 8)
    adjustments += min(max(0, feat.get("cyclomatic_complexity", 1) - 10) * 1, 6)

    score = base_score + adjustments
    return max(0, min(100, score))
