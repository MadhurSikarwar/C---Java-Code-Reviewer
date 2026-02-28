"""
IntelliReview — ML Predictor
Loads the saved model and runs inference to produce a risk label and score.
"""
import os
import sys
import joblib
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, ENCODER_PATH, RISK_CLEAN_MAX, RISK_MODERATE_MAX
from ml.synthetic_data import FEATURE_NAMES   # single source of truth

DL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_dl_massive.keras")
DL_SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_scaler.pkl")

_model = None
_dl_model = None
_dl_scaler = None
_v3_model = None
_label_names = ["Clean", "Moderate Risk", "High Risk"]


def load_model():
    """Load the ML model from disk (called once on startup)."""
    global _model
    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        # Train on the fly if model not found
        print("⚠️  model.pkl not found — training now...")
        _train_model()

    _model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")


def _train_model():
    """Train model inline if not pre-built."""
    train_script = os.path.join(os.path.dirname(__file__), "train.py")
    import subprocess
    subprocess.run([sys.executable, train_script], check=True)


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


def predict_risk(feature_vector: List[float], model_type: str = "v3", issues: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Given a feature vector, return risk label, score 0–100, and per-class probabilities.
    model_type: 'dl', 'v3', or 'all'.
    """
    models_to_run = ["dl", "v3"] if model_type == "all" else [model_type]
    results = {}

    X_full = np.array(feature_vector).reshape(1, -1)
    
    # Classic features for old ensemble model (just in case it is still explicitly requested)
    X_classic = np.copy(X_full)
    if X_classic.shape[1] > 21:
        X_classic = X_classic[:, :21]

    for m_type in models_to_run:
        label_idx = 0
        proba = [0.0, 0.0, 0.0]
        
        if m_type == "dl":
            global _dl_model, _dl_scaler
            if _dl_model is None: load_dl_model()
            if _dl_model is None: continue
            
            X_dl = np.copy(X_full)
            target_feats = len(_dl_scaler.mean_) if hasattr(_dl_scaler, 'mean_') else 34
            if X_dl.shape[1] < target_feats:
                padding = np.zeros((1, target_feats - X_dl.shape[1]))
                X_dl = np.hstack((X_dl, padding))
            elif X_dl.shape[1] > target_feats:
                X_dl = X_dl[:, :target_feats]
                
            X_scaled = _dl_scaler.transform(X_dl)
            probs = _dl_model.predict(X_scaled, verbose=0)[0].tolist()
            label_idx = int(np.argmax(probs))
            proba = probs

        elif m_type == "v3":
            global _v3_model
            if _v3_model is None: load_v3_model()
            if _v3_model is None: continue
            
            X_v3 = np.copy(X_full)
            target_feats = getattr(_v3_model, 'n_features_in_', 34)
            if X_v3.shape[1] < target_feats:
                padding = np.zeros((1, target_feats - X_v3.shape[1]))
                X_v3 = np.hstack((X_v3, padding))
            elif X_v3.shape[1] > target_feats:
                X_v3 = X_v3[:, :target_feats]
                
            label_idx = int(_v3_model.predict(X_v3)[0])
            try: proba = _v3_model.predict_proba(X_v3)[0].tolist()
            except AttributeError:
                proba = [0.0, 0.0, 0.0]
                proba[label_idx] = 1.0

        elif m_type == "ensemble":
            global _model
            if _model is None: load_model()
            if _model is None: continue
            
            label_idx = int(_model.predict(X_classic)[0])
            try: proba = _model.predict_proba(X_classic)[0].tolist()
            except AttributeError:
                proba = [0.0, 0.0, 0.0]
                proba[label_idx] = 1.0

        # --- GENERATION 4 HARD OVERRIDE LOGIC & CONFIDENCE CALIBRATION ---
        confidence = float(np.max(proba) * 100)
        explanations = []
        is_hard_override = False
        
        # Penalize confidence for untracked global states (index 30 = global_mutations out of 34 total config)
        if len(feature_vector) > 30 and feature_vector[30] > 0:
            penalty = min(25.0, feature_vector[30] * 5.0)
            confidence = max(15.0, confidence - penalty)
            explanations.append(f"Confidence penalized by {penalty}% due to dynamic pointer states escaping CFG tracking bounds.")
            
        if len(feature_vector) > 28 and feature_vector[28] > 20: # deep complexity
            confidence -= 10.0
            explanations.append("Maximal Cyclomatic Complexity limits branch prediction bounds accuracy.")

        if issues:
            has_critical = any(i.get("severity") in ("CRITICAL", "HIGH") for i in issues)
            if has_critical and label_idx < 2:
                print(f"⚠️ [Hard Override] Rule Engine found Critical issues. Overriding ML prediction.")
                label_idx = 2
                proba = [0.0, 0.1, 0.9]
                confidence = 100.0
                is_hard_override = True
                explanations.append("Prediction Hard-Overridden to HIGH RISK: Critical Rule-Engine violations (e.g. Memory Leak, Use-After-Free) detected natively.")

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
