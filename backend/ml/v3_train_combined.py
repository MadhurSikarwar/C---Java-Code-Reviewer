import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

CLASSIC_FEATURES = [
    "num_functions", "num_loops", "max_nesting_depth", "num_conditionals",
    "num_pointer_uses", "num_mallocs", "num_frees", "memory_leak_count",
    "unsafe_function_count", "cyclomatic_complexity", "recursion_count",
    "lines_of_code", "function_call_count", "leak_ratio", "unsafe_density",
    "comment_ratio", "magic_number_count", "exception_handling",
    "halstead_volume", "duplicate_block_score", "dead_code_estimate"
]

V3_FEATURES = [
    "use_after_free_count", "double_free_count", "path_leak_probability",
    "pointer_state_transitions", "infinite_loop_risks", "uninitialized_vars_used",
    "cfg_node_count", "branch_density", "v3_cyclomatic_complexity", 
    "global_mutation_count", "v3_loop_count", "v3_max_loop_depth", "v3_recursion_count"
]

def augment_v3_features_robustly(df: pd.DataFrame, seed: int = 1337) -> pd.DataFrame:
    """
    Imputes the 13 V3 features onto the historical Juliet dataset WITHOUT TARGET LEAKAGE.
    We base the V3 features heavily on the existing code metrics rather than the label,
    with added statistical noise, so the model has to actually 'learn' the correlations
    rather than memorizing a perfectly separated synthetic distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    augmented = df.copy()
    
    def iclamp(arr, lo, hi): return np.clip(arr, lo, hi).astype(int)
    def clamp(arr, lo, hi): return np.clip(arr, lo, hi)

    # Base the new metrics semi-realistically off existing classic metrics
    loc = df["lines_of_code"].values
    ptrs = df["num_pointer_uses"].values
    cc = df["cyclomatic_complexity"].values
    leaks = df["memory_leak_count"].values
    loops = df["num_loops"].values
    
    # ── 1. Pointer & Path Metrics ──
    # UAF is rare, mostly happens when ptrs > 0 and LOC is somewhat high
    uaf_base = (ptrs > 5) * (rng.random(n) > 0.85).astype(int)
    use_after_free = iclamp(uaf_base + rng.binomial(2, 0.05, n), 0, 5)

    # Double free follows a similar logic but even rarer
    df_base = (leaks > 0) * (rng.random(n) > 0.90).astype(int)
    double_free = iclamp(df_base + rng.binomial(2, 0.02, n), 0, 3)

    # Path leak probability should correlate slightly with CC and existing leaks
    path_leak = clamp((leaks * 0.15) + (cc * 0.02) + rng.normal(0, 0.1, n), 0.0, 1.0)
    
    transitions = iclamp((ptrs * 1.5) + rng.normal(0, ptrs * 0.2 + 1, n), 0, 500)

    # ── 2. Data Flow Metrics ──
    inf_loop = (loops > 0) * (rng.random(n) > 0.85).astype(int)
    inf_loop = iclamp(inf_loop + rng.binomial(1, 0.05, n), 0, 5)
    
    uninit = iclamp(rng.binomial(cc.astype(int), 0.05), 0, 10)

    # ── 3. Structural Smell Metrics ──
    cfg_nodes = iclamp((loc * 0.8) + rng.normal(0, loc * 0.1 + 1, n), 5, 2000)
    branch_den = clamp(1.0 + (cc / (cfg_nodes + 1)) + rng.normal(0, 0.05, n), 1.0, 2.5)
    
    # Exact CC from V3 is usually very close to original CC
    v3_cc = iclamp(cc + rng.normal(0, 1, n), 1, 100)
    global_mut = iclamp(rng.binomial((loc/10).astype(int) + 1, 0.05), 0, 15)

    # ── 4. Complexity Engine Metrics ──
    v3_lc = iclamp(loops + rng.normal(0, 0.5, n), 0, 50)
    v3_ld = iclamp((v3_lc > 0).astype(int) + (v3_lc > 3).astype(int) + rng.binomial(1, 0.1, n), 0, 10)
    v3_rec = df["recursion_count"].values

    # Merge
    augmented["use_after_free_count"] = use_after_free
    augmented["double_free_count"] = double_free
    augmented["path_leak_probability"] = path_leak
    augmented["pointer_state_transitions"] = transitions
    augmented["infinite_loop_risks"] = inf_loop
    augmented["uninitialized_vars_used"] = uninit
    augmented["cfg_node_count"] = cfg_nodes
    augmented["branch_density"] = branch_den
    augmented["v3_cyclomatic_complexity"] = v3_cc
    augmented["global_mutation_count"] = global_mut
    augmented["v3_loop_count"] = v3_lc
    augmented["v3_max_loop_depth"] = v3_ld
    augmented["v3_recursion_count"] = v3_rec
    
    return augmented

def train():
    base_dir = os.path.dirname(__file__)
    juliet_path = os.path.join(base_dir, "juliet_data.csv")
    
    if not os.path.exists(juliet_path):
        print(f"❌ Could not find {juliet_path}")
        return
        
    print(f"📂 Loading Juliet dataset from {juliet_path}...")
    df = pd.read_csv(juliet_path)
    
    # Fill any NaNs in classic features
    for col in CLASSIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"✨ Robustly Augmenting {len(df)} rows with 13 Path-Sensitive V3 features (Preventing Target Leakage)...")
    df_augmented = augment_v3_features_robustly(df)
    
    # The final feature set is CLASSIC (21) + V3 (13) = 34 features
    final_features = CLASSIC_FEATURES + V3_FEATURES
    X = df_augmented[final_features]
    y = df_augmented["label"]
    
    # Add validation split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n🧠 Training Ultimate V3 Path-Sensitive Model on {len(X_train)} samples, {len(final_features)} features...")
    clf = RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Training Complete. Validation Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=["Clean", "Moderate Risk", "High Risk"]))
    
    out_path = os.path.join(base_dir, "model_v3_path_sensitive.pkl")
    joblib.dump(clf, out_path)
    print(f"💾 V3 Model saved to: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")
    
    # Optional: Feature importance check to verify no single feature dominates
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    print("\n🔍 Top 5 Most Important Features:")
    for i in top_indices:
        print(f"   - {final_features[i]}: {importances[i]:.4f}")

    # Save the feature names so the API knows what order to feed them in
    feature_list_path = os.path.join(base_dir, "v3_feature_list.pkl")
    joblib.dump(final_features, feature_list_path)

if __name__ == "__main__":
    train()
