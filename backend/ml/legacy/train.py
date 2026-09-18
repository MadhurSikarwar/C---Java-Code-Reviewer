"""
IntelliReview — ML Model Trainer  (ULTRA ACCURACY: LightGBM + Optuna + SMOTE + Stacking)

Pipeline:
  1. 5000 samples × 19 features from upgraded synthetic_data.py
  2. SMOTE oversampling on Moderate Risk (the hardest boundary)
  3. Optuna Bayesian hyperparameter search for LightGBM (50 trials)
  4. Optuna Bayesian search for XGBoost (30 trials)
  5. StackingClassifier (LightGBM + XGBoost + ExtraTrees) with LR meta-learner
  6. 5-fold Stratified CV on final model
  7. Feature importance plot + confusion matrix printed
"""
import os, sys, warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from ml.synthetic_data import generate_synthetic_data, FEATURE_NAMES

MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

CV_FOLDS      = 5
OPTUNA_TRIALS_LGB = 50   # increase for more tuning (each trial ~3 sec)
OPTUNA_TRIALS_XGB = 30


# ── Helpers ────────────────────────────────────────────────────────────
def _eval(model, X_test, y_test, name):
    pred = model.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    f1   = f1_score(y_test, pred, average="macro")
    print(f"   [{name}]  Accuracy={acc:.4f}  F1-macro={f1:.4f}")
    return f1, pred


def _print_importances(model, top_n=10):
    """Print top feature importances from the best base estimator."""
    try:
        # StackingClassifier: get first estimator
        base = (model.estimators_[0][1]
                if hasattr(model, "estimators_") else model)
        if hasattr(base, "feature_importances_"):
            fi = sorted(zip(FEATURE_NAMES, base.feature_importances_),
                        key=lambda x: x[1], reverse=True)
            print(f"\n🔑 Top-{top_n} Feature Importances:")
            for i, (fname, imp) in enumerate(fi[:top_n]):
                bar = "█" * int(imp * 60)
                print(f"   {i+1:2}. {fname:<28} {imp:.4f}  {bar}")
    except Exception:
        pass


# ── SMOTE ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
def _apply_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        counts = np.bincount(y_res)
        print(f"   SMOTE applied → {dict(enumerate(counts))}")
        return X_res, y_res
    except ImportError:
        print("   ⚠️  imbalanced-learn not installed → skipping SMOTE.")
        print("      Run: pip install imbalanced-learn")
        return X_train, y_train


# ── Optuna objective factories ─────────────────────────────────────────
def _lgb_objective(X_tr, y_tr, cv):
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1000),
            "max_depth":        trial.suggest_int("max_depth", 4, 14),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 120),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 40),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight":     "balanced",
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        -1,
        }
        clf = lgb.LGBMClassifier(**params)
        scores = cross_val_score(clf, X_tr, y_tr, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        return scores.mean()
    return objective


def _xgb_objective(X_tr, y_tr, cv):
    from xgboost import XGBClassifier

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "use_label_encoder": False,
            "eval_metric":      "mlogloss",
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        clf = XGBClassifier(**params)
        scores = cross_val_score(clf, X_tr, y_tr, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        return scores.mean()
    return objective


# ── Main train function ────────────────────────────────────────────────
def train():
    print("🔧 Generating 5000-sample dataset (19 features)...")
    df = generate_synthetic_data(n=5000)

    X = df[FEATURE_NAMES].values.astype(float)
    y = df["label"].values.astype(int)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # ── SMOTE ────────────────────────────────────────────────────────
    print("\n⚖️  Applying SMOTE to balance training set...")
    X_train_bal, y_train_bal = _apply_smote(X_train, y_train)

    models = {}

    # ── 1. LightGBM + Optuna ─────────────────────────────────────────
    lgb_model = None
    try:
        import lightgbm as lgb
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n🚀 LightGBM — Optuna search ({OPTUNA_TRIALS_LGB} trials)...")
        study_lgb = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
        study_lgb.optimize(
            _lgb_objective(X_train_bal, y_train_bal, cv),
            n_trials=OPTUNA_TRIALS_LGB,
            show_progress_bar=False,
        )
        best_lgb_params = study_lgb.best_params
        best_lgb_params.update({
            "class_weight": "balanced", "random_state": 42,
            "n_jobs": -1, "verbosity": -1,
        })
        lgb_model = lgb.LGBMClassifier(**best_lgb_params)
        lgb_model.fit(X_train_bal, y_train_bal)
        lgb_f1, _ = _eval(lgb_model, X_test, y_test, "LightGBM")
        models["LightGBM"] = (lgb_f1, lgb_model)
        print(f"   Best Optuna params: {study_lgb.best_params}")
    except ImportError:
        print("   ⚠️  lightgbm/optuna not installed.")
        print("      Run: pip install lightgbm optuna")

    # ── 2. XGBoost + Optuna ──────────────────────────────────────────
    xgb_model = None
    try:
        from xgboost import XGBClassifier
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n⚡ XGBoost — Optuna search ({OPTUNA_TRIALS_XGB} trials)...")
        study_xgb = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
        study_xgb.optimize(
            _xgb_objective(X_train_bal, y_train_bal, cv),
            n_trials=OPTUNA_TRIALS_XGB,
            show_progress_bar=False,
        )
        best_xgb_params = study_xgb.best_params
        best_xgb_params.update({
            "use_label_encoder": False, "eval_metric": "mlogloss",
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
        })
        xgb_model = XGBClassifier(**best_xgb_params)
        xgb_model.fit(X_train_bal, y_train_bal)
        xgb_f1, _ = _eval(xgb_model, X_test, y_test, "XGBoost")
        models["XGBoost"] = (xgb_f1, xgb_model)
    except ImportError:
        print("   ⚠️  xgboost/optuna not installed.")

    # ── 3. ExtraTrees (fast strong base learner, no tuning needed) ───
    print("\n🌳 Training ExtraTrees...")
    et = ExtraTreesClassifier(
        n_estimators=400, max_depth=None,
        min_samples_split=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    et.fit(X_train_bal, y_train_bal)
    et_f1, _ = _eval(et, X_test, y_test, "ExtraTrees")
    models["ExtraTrees"] = (et_f1, et)

    # ── 4. Stacking Ensemble ─────────────────────────────────────────
    base_estimators = []
    if lgb_model:  base_estimators.append(("lgb", lgb_model))
    if xgb_model:  base_estimators.append(("xgb", xgb_model))
    base_estimators.append(("et", et))

    final_model = None
    if len(base_estimators) >= 2:
        print(f"\n🔀 Building StackingClassifier ({len(base_estimators)} base learners + LR meta)...")
        meta = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=5.0, max_iter=2000,
                                      solver="lbfgs", random_state=42)),
        ])
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta,
            cv=CV_FOLDS,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        stack.fit(X_train_bal, y_train_bal)
        stack_f1, stack_pred = _eval(stack, X_test, y_test, "Stacking")
        models["Stacking"] = (stack_f1, stack)

    # ── 5. Pick best ─────────────────────────────────────────────────
    best_name, (best_f1, best_model) = max(models.items(), key=lambda x: x[1][0])
    best_pred = best_model.predict(X_test)

    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_name}  |  F1-macro = {best_f1:.4f}")
    print(f"   Test Accuracy = {accuracy_score(y_test, best_pred):.4f}")
    print(f"{'='*60}")

    print("\n📋 Classification Report:")
    print(classification_report(
        y_test, best_pred,
        target_names=["Clean", "Moderate Risk", "High Risk"],
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))

    # ── 6. Full CV on entire dataset ─────────────────────────────────
    print(f"\n📈 {CV_FOLDS}-Fold CV accuracy (full dataset)...")
    cv_acc = cross_val_score(best_model, X, y_enc, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    print(f"   Mean = {cv_acc.mean():.4f}  ±  {cv_acc.std():.4f}")

    # ── 7. Feature Importances ───────────────────────────────────────
    _print_importances(best_model)

    # ── 8. Save ──────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n✅ Saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
