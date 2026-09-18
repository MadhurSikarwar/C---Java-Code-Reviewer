"""
IntelliReview — Hybrid Trainer (Synthetic + Real + Juliet Data)
================================================================
Pure sklearn — NO lightgbm, NO optuna, NO imbalanced-learn required.
Uses only packages already installed with scikit-learn.

What makes this high accuracy:
  1. Combines synthetic (10,000) + Juliet Test Suite (real vulnerabilities)
     + optional GitHub real_data.csv
  2. HistGradientBoostingClassifier — sklearn's native fast boosting (LightGBM-like)
  3. GradientBoostingClassifier     — sklearn's standard boosting (very strong)
  4. ExtraTreesClassifier           — low-variance ensemble
  5. RandomForestClassifier         — tuned with RandomizedSearchCV
  6. StackingClassifier with Logistic Regression meta-learner
  7. 5-fold Stratified CV for reliable evaluation
  8. 19 features from full IntelliReview pipeline
"""
import os, sys, warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline

from ml.synthetic_data import generate_synthetic_data, FEATURE_NAMES

MODEL_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH     = os.path.join(MODEL_DIR, "label_encoder.pkl")
REAL_DATA_PATH        = os.path.join(MODEL_DIR, "real_data.csv")
JULIET_DATA_PATH      = os.path.join(MODEL_DIR, "juliet_data.csv")
JULIET_JAVA_DATA_PATH = os.path.join(MODEL_DIR, "juliet_java_data.csv")

CV_FOLDS    = 5
N_SYNTH     = 10000  # synthetic samples — large enough base for 3-class coverage
N_ITER_RF   = 40     # RandomizedSearchCV iterations for Random Forest
N_ITER_GB   = 30     # RandomizedSearchCV iterations for GradientBoosting


# ── Data loading ───────────────────────────────────────────────────────
def load_data():
    """Load synthetic + C/Java Juliet + optional GitHub real data, merge and return X, y."""
    print("🔧 Generating synthetic data ({:,} samples × {} features)...".format(
        N_SYNTH, len(FEATURE_NAMES)))
    df_synth = generate_synthetic_data(n=N_SYNTH)

    frames       = [df_synth]
    n_real        = 0
    n_juliet_c    = 0
    n_juliet_java = 0

    # ── Load Juliet C/C++ Test Suite data ─────────────────────────────────
    if os.path.exists(JULIET_DATA_PATH):
        try:
            df_j = pd.read_csv(JULIET_DATA_PATH)
            missing = [f for f in FEATURE_NAMES if f not in df_j.columns]
            if missing:
                print(f"   ⚠️  juliet_data.csv missing columns {missing} — skipping.")
            else:
                frames.append(df_j[FEATURE_NAMES + ["label"]])
                n_juliet_c = len(df_j)
                dist = dict(zip(*np.unique(df_j["label"].values, return_counts=True)))
                print(f"   ✅ Juliet C/C++ data  : {n_juliet_c:>6,} samples | dist={dist}")
        except Exception as e:
            print(f"   ⚠️  Could not load juliet_data.csv: {e}")
    else:
        print("   ℹ️  juliet_data.csv not found — run: python ml\\juliet_data_collector.py")

    # ── Load Juliet Java Test Suite data ──────────────────────────────────
    if os.path.exists(JULIET_JAVA_DATA_PATH):
        try:
            df_jj = pd.read_csv(JULIET_JAVA_DATA_PATH)
            missing = [f for f in FEATURE_NAMES if f not in df_jj.columns]
            if missing:
                print(f"   ⚠️  juliet_java_data.csv missing columns {missing} — skipping.")
            else:
                frames.append(df_jj[FEATURE_NAMES + ["label"]])
                n_juliet_java = len(df_jj)
                dist = dict(zip(*np.unique(df_jj["label"].values, return_counts=True)))
                print(f"   ✅ Juliet Java data   : {n_juliet_java:>6,} samples | dist={dist}")
        except Exception as e:
            print(f"   ⚠️  Could not load juliet_java_data.csv: {e}")
    else:
        print("   ℹ️  juliet_java_data.csv not found — run: python ml\\juliet_java_collector.py")

    # ── Load optional GitHub real data ────────────────────────────────────
    if os.path.exists(REAL_DATA_PATH):
        try:
            df_real = pd.read_csv(REAL_DATA_PATH)
            missing = [f for f in FEATURE_NAMES if f not in df_real.columns]
            if missing:
                print(f"   ⚠️  real_data.csv missing columns {missing} — skipping.")
            else:
                frames.append(df_real[FEATURE_NAMES + ["label"]])
                n_real = len(df_real)
                print(f"   ✅ GitHub real data   : {n_real:>6,} samples")
        except Exception as e:
            print(f"   ⚠️  Could not load real_data.csv: {e}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[FEATURE_NAMES].values.astype(float)
    y = df["label"].values.astype(int)

    print(f"\n   📊 Total dataset : {len(df):,} samples")
    print(f"      {N_SYNTH:,} synthetic + {n_juliet_c:,} Juliet-C + "
          f"{n_juliet_java:,} Juliet-Java + {n_real} GitHub")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# ── Model definitions ──────────────────────────────────────────────────
def build_models(X_train, y_train, cv):
    models = {}

    # ──── 1. HistGradientBoosting (fast, LightGBM-like, great for large data) ─
    print("\n🚀 Training HistGradientBoostingClassifier (fast boosting)...")
    hgb_params = {
        "max_iter":        [200, 300, 400],
        "max_depth":       [3, 4, 5, 6, None],
        "learning_rate":   [0.05, 0.08, 0.1, 0.15],
        "min_samples_leaf":[10, 20, 30],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }
    hgb_search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
        hgb_params, n_iter=30, cv=cv,
        scoring="f1_macro", n_jobs=-1, random_state=42, verbose=0,
    )
    hgb_search.fit(X_train, y_train)
    best_hgb = hgb_search.best_estimator_
    models["HistGradBoosting"] = (hgb_search.best_score_, best_hgb)
    print(f"   CV F1={hgb_search.best_score_:.4f}  | params={hgb_search.best_params_}")

    # ──── 2. Gradient Boosting (sklearn built-in, very powerful) ─────
    print(f"\n⚡ Tuning GradientBoosting ({N_ITER_GB} random search iters)...")
    gb_params = {
        "n_estimators":  [200, 300, 400, 500],
        "max_depth":     [3, 4, 5, 6],
        "learning_rate": [0.05, 0.08, 0.1, 0.15],
        "subsample":     [0.7, 0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
        "max_features":  ["sqrt", "log2", 0.5, 0.7],
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params, n_iter=N_ITER_GB, cv=cv,
        scoring="f1_macro", n_jobs=-1, random_state=42, verbose=0,
    )
    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    models["GradientBoosting"] = (gb_search.best_score_, best_gb)
    print(f"   CV F1={gb_search.best_score_:.4f}  | params={gb_search.best_params_}")

    # ──── 3. Random Forest (tuned) ────────────────────────────────────
    print(f"\n🌲 Tuning RandomForest ({N_ITER_RF} random search iters)...")
    rf_params = {
        "n_estimators":      [300, 400, 500, 600],
        "max_depth":         [8, 10, 12, 15, None],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf":  [1, 2],
        "max_features":      ["sqrt", "log2", 0.4, 0.6],
        "class_weight":      ["balanced", "balanced_subsample"],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_params, n_iter=N_ITER_RF, cv=cv,
        scoring="f1_macro", n_jobs=-1, random_state=42, verbose=0,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    models["RandomForest"] = (rf_search.best_score_, best_rf)
    print(f"   CV F1={rf_search.best_score_:.4f}  | params={rf_search.best_params_}")

    # ──── 4. ExtraTrees (fast and robust) ────────────────────────────
    print("\n🌳 Training ExtraTrees (fast, no tuning)...")
    et = ExtraTreesClassifier(
        n_estimators=500, max_depth=None,
        min_samples_split=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    et.fit(X_train, y_train)
    et_cv = cross_val_score(et, X_train, y_train, cv=cv,
                            scoring="f1_macro", n_jobs=-1).mean()
    models["ExtraTrees"] = (et_cv, et)
    print(f"   CV F1={et_cv:.4f}")

    return models


# ── Stacking ensemble ──────────────────────────────────────────────────
def build_stack(base_models, X_train, y_train, cv):
    print(f"\n🔀 Building StackingClassifier ({len(base_models)} base learners + LR meta)...")
    base_estimators = [(name, model) for name, (_, model) in base_models.items()]
    meta = Pipeline([
        ("qt",  QuantileTransformer(output_distribution="normal", random_state=42)),
        ("lr",  LogisticRegression(C=10.0, max_iter=3000, solver="lbfgs",
                                   multi_class="multinomial", random_state=42)),
    ])
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)
    return stack


# ── Main ───────────────────────────────────────────────────────────────
def train():
    X, y = load_data()
    le   = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.12, random_state=42, stratify=y_enc,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # ── Try SMOTE (optional) ─────────────────────────────────────────
    try:
        from imblearn.over_sampling import SMOTE
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        print(f"   ✅ SMOTE applied: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    except ImportError:
        print("   ℹ️  SMOTE skipped (install imbalanced-learn for it)")

    # ── Build and tune base models ────────────────────────────────────
    base_models = build_models(X_train, y_train, cv)

    # ── Build stacking ensemble ───────────────────────────────────────
    stack = build_stack(base_models, X_train, y_train, cv)
    stack_pred = stack.predict(X_test)
    stack_f1   = f1_score(y_test, stack_pred, average="macro")
    stack_acc  = accuracy_score(y_test, stack_pred)

    # Add to candidates
    all_models = dict(base_models)
    all_models["Stacking"] = (stack_f1, stack)

    # ── Pick best ────────────────────────────────────────────────────
    best_name, (best_score, best_model) = max(
        all_models.items(), key=lambda x: x[1][0]
    )
    best_pred = best_model.predict(X_test)
    best_acc  = accuracy_score(y_test, best_pred)
    best_f1   = f1_score(y_test, best_pred, average="macro")

    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL : {best_name}")
    print(f"   Test Accuracy  : {best_acc:.4f}  ({best_acc*100:.1f}%)")
    print(f"   F1 Macro       : {best_f1:.4f}")
    print(f"{'='*60}")

    print("\n📋 Classification Report:")
    print(classification_report(
        y_test, best_pred,
        target_names=["Clean", "Moderate Risk", "High Risk"],
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, best_pred))

    # ── Full CV ───────────────────────────────────────────────────────
    print(f"\n📈 {CV_FOLDS}-Fold CV on full dataset...")
    cv_acc = cross_val_score(best_model, X, y_enc, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    print(f"   Mean = {cv_acc.mean():.4f}  ±  {cv_acc.std():.4f}")

    # ── Feature importances ───────────────────────────────────────────
    try:
        # For stacking, get first base estimator
        base = (best_model.estimators_[0][1]
                if hasattr(best_model, "estimators_") else best_model)
        if hasattr(base, "feature_importances_"):
            fi = sorted(zip(FEATURE_NAMES, base.feature_importances_),
                        key=lambda x: x[1], reverse=True)
            print(f"\n🔑 Top-10 Feature Importances:")
            for i, (fname, imp) in enumerate(fi[:10]):
                bar = "█" * int(imp * 60)
                print(f"   {i+1:2}. {fname:<28} {imp:.4f}  {bar}")
    except Exception:
        pass

    # ── Save ─────────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
