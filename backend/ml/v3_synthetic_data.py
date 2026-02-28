"""
IntelliReview V3 — Synthetic Training Data Generator
=====================================================
Generates data explicitly for the Path-Sensitive engine.
Produces `v3_training_data.csv`.
"""
import numpy as np
import pandas as pd
import os

SEED = 42
NUM_SAMPLES = 5000

V3_FEATURE_NAMES = [
    "use_after_free_count",
    "double_free_count",
    "path_leak_probability",
    "pointer_state_transitions",
    "infinite_loop_risks",
    "uninitialized_vars_used",
    "cfg_node_count",
    "branch_density",
    "cyclomatic_complexity",
    "global_mutation_count",
    "loop_count",
    "max_loop_depth",
    "recursion_count"
]

LABEL_MAP = {0: "Clean", 1: "Moderate Risk", 2: "High Risk"}

def generate_v3_synthetic_data(n: int = NUM_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_clean    = int(n * 0.40)
    n_moderate = int(n * 0.35)
    n_high     = n - n_clean - n_moderate

    def clamp(arr, lo, hi):
        return np.clip(arr, lo, hi)

    def iclamp(arr, lo, hi):
        return np.clip(arr, lo, hi).astype(int)

    # ── CLEAN ──────────────────────────────────────────────────────────
    clean = {
        "use_after_free_count":  np.zeros(n_clean, dtype=int),
        "double_free_count":     np.zeros(n_clean, dtype=int),
        "path_leak_probability": np.zeros(n_clean, dtype=float),
        "pointer_state_transitions": iclamp(rng.normal(15, 5, n_clean), 5, 40),
        "infinite_loop_risks":   np.zeros(n_clean, dtype=int),
        "uninitialized_vars_used": np.zeros(n_clean, dtype=int),
        "cfg_node_count":        iclamp(rng.normal(25, 10, n_clean), 10, 60),
        "branch_density":        clamp(rng.normal(1.05, 0.05, n_clean), 1.0, 1.2),
        "cyclomatic_complexity": iclamp(rng.normal(3, 1.5, n_clean), 1, 7),
        "global_mutation_count": np.zeros(n_clean, dtype=int),
        "loop_count":            iclamp(rng.normal(1, 1, n_clean), 0, 3),
        "max_loop_depth":        iclamp(rng.normal(1, 0.3, n_clean), 0, 1),
        "recursion_count":       np.zeros(n_clean, dtype=int),
        "label": [0] * n_clean,
    }

    # ── MODERATE ───────────────────────────────────────────────────────
    moderate = {
        "use_after_free_count":  np.zeros(n_moderate, dtype=int),
        "double_free_count":     np.zeros(n_moderate, dtype=int),
        "path_leak_probability": clamp(rng.normal(0.2, 0.1, n_moderate), 0.0, 0.5),
        "pointer_state_transitions": iclamp(rng.normal(45, 15, n_moderate), 20, 100),
        "infinite_loop_risks":   iclamp(rng.normal(0.5, 0.5, n_moderate), 0, 2),
        "uninitialized_vars_used": iclamp(rng.normal(1, 0.8, n_moderate), 0, 3),
        "cfg_node_count":        iclamp(rng.normal(80, 25, n_moderate), 30, 200),
        "branch_density":        clamp(rng.normal(1.2, 0.1, n_moderate), 1.05, 1.4),
        "cyclomatic_complexity": iclamp(rng.normal(11, 2, n_moderate), 8, 16),
        "global_mutation_count": iclamp(rng.normal(1, 1, n_moderate), 0, 4),
        "loop_count":            iclamp(rng.normal(4, 1.5, n_moderate), 2, 8),
        "max_loop_depth":        iclamp(rng.normal(2, 0.6, n_moderate), 1, 3),
        "recursion_count":       iclamp(rng.normal(0.5, 0.7, n_moderate), 0, 3),
        "label": [1] * n_moderate,
    }

    # ── HIGH RISK ─────────────────────────────────────────────────────
    high = {
        "use_after_free_count":  iclamp(rng.normal(1.5, 0.8, n_high), 1, 4),
        "double_free_count":     iclamp(rng.normal(1.2, 0.6, n_high), 0, 3),
        "path_leak_probability": clamp(rng.normal(0.7, 0.2, n_high), 0.4, 1.0),
        "pointer_state_transitions": iclamp(rng.normal(120, 30, n_high), 60, 300),
        "infinite_loop_risks":   iclamp(rng.normal(2, 1, n_high), 1, 5),
        "uninitialized_vars_used": iclamp(rng.normal(3, 1.5, n_high), 1, 8),
        "cfg_node_count":        iclamp(rng.normal(200, 60, n_high), 100, 500),
        "branch_density":        clamp(rng.normal(1.4, 0.15, n_high), 1.25, 2.0),
        "cyclomatic_complexity": iclamp(rng.normal(24, 4, n_high), 18, 50),
        "global_mutation_count": iclamp(rng.normal(4, 2, n_high), 2, 12),
        "loop_count":            iclamp(rng.normal(12, 3, n_high), 6, 25),
        "max_loop_depth":        iclamp(rng.normal(4, 0.8, n_high), 3, 8),
        "recursion_count":       iclamp(rng.normal(3, 1.5, n_high), 1, 10),
        "label": [2] * n_high,
    }

    frames = [pd.DataFrame(p) for p in (clean, moderate, high)]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["label_name"] = df["label"].map(LABEL_MAP)
    return df

if __name__ == "__main__":
    df = generate_v3_synthetic_data()
    out_path = os.path.join(os.path.dirname(__file__), "v3_training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df)} V3 samples, {len(V3_FEATURE_NAMES)} path features → {out_path}")
    print(df["label_name"].value_counts())
