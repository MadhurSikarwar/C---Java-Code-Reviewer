"""NOTE: only FEATURE_NAMES / LABEL_MAP from this module are still used (single source of truth for the feature layout).
The data generator below is legacy: it produced size-only blobs and must not be used for training (see ml/legacy/README.md).
"""
"""
IntelliReview — Synthetic Training Data Generator  (ULTRA-ACCURACY VERSION)
19 total features: 13 original + 2 engineered ratios + 6 new code-quality signals.

NEW FEATURES:
  comment_ratio          → well-commented code = cleaner
  magic_number_count     → hardcoded literals = bad practice
  exception_handling     → proper error handling = better quality (Java)
  halstead_volume        → approximated complexity measure
  duplicate_block_score  → copy-paste likelihood
  dead_code_estimate     → unused variable / unreachable code count
"""
import numpy as np
import pandas as pd
import os

SEED = 42
NUM_SAMPLES = 5000

FEATURE_NAMES = [
    # ── Original 13 ─────────────────────────────────────────────────
    "num_functions",
    "num_loops",
    "max_nesting_depth",
    "num_conditionals",
    "num_pointer_uses",
    "num_mallocs",
    "num_frees",
    "memory_leak_count",
    "unsafe_function_count",
    "cyclomatic_complexity",
    "recursion_count",
    "lines_of_code",
    "function_call_count",
    # ── 2 Engineered ratios ──────────────────────────────────────────
    "leak_ratio",           # memory_leak_count / max(num_mallocs, 1)
    "unsafe_density",       # unsafe_function_count / max(num_functions, 1)
    # ── 6 New quality signals ────────────────────────────────────────
    "comment_ratio",        # comment lines / total lines  (0.0–1.0)
    "magic_number_count",   # hardcoded numeric literals
    "exception_handling",   # try/catch/throw/finally blocks
    "halstead_volume",      # approx: (operators + operands) * log2(vocabulary)
    "duplicate_block_score",# 0–10: copy-paste pattern score
    "dead_code_estimate",   # unused vars / unreachable returns
]

LABEL_MAP = {0: "Clean", 1: "Moderate Risk", 2: "High Risk"}


def _add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df["leak_ratio"] = (
        df["memory_leak_count"] / df["num_mallocs"].clip(lower=1)
    ).clip(0, 1).round(2)
    df["unsafe_density"] = (
        df["unsafe_function_count"] / df["num_functions"].clip(lower=1)
    ).clip(0, 5).round(2)
    return df


def generate_synthetic_data(n: int = NUM_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate n labeled synthetic samples with 19 features."""
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
        "num_functions":         iclamp(rng.normal(3,   1.2, n_clean),  1,  8),
        "num_loops":             iclamp(rng.normal(2,   1.0, n_clean),  0,  5),
        "max_nesting_depth":     iclamp(rng.normal(1,   0.4, n_clean),  0,  1),
        "num_conditionals":      iclamp(rng.normal(3,   1.5, n_clean),  0,  7),
        "num_pointer_uses":      iclamp(rng.normal(1,   0.8, n_clean),  0,  4),
        "num_mallocs":           iclamp(rng.normal(1,   0.8, n_clean),  0,  3),
        "num_frees":             iclamp(rng.normal(1,   0.8, n_clean),  0,  3),
        "memory_leak_count":     np.zeros(n_clean, dtype=int),
        "unsafe_function_count": np.zeros(n_clean, dtype=int),
        "cyclomatic_complexity": iclamp(rng.normal(3,   1.5, n_clean),  1,  7),
        "recursion_count":       np.zeros(n_clean, dtype=int),
        "lines_of_code":         iclamp(rng.normal(45,  15,  n_clean), 10, 90),
        "function_call_count":   iclamp(rng.normal(6,   3,   n_clean),  1, 15),
        # New features — clean code has good comments, few magic numbers
        "comment_ratio":         clamp(rng.normal(0.22, 0.05, n_clean), 0.10, 0.45).round(2),
        "magic_number_count":    iclamp(rng.normal(1,   1,   n_clean),  0,  4),
        "exception_handling":    iclamp(rng.normal(2,   1,   n_clean),  0,  5),
        "halstead_volume":       iclamp(rng.normal(200, 60,  n_clean), 50, 400),
        "duplicate_block_score": iclamp(rng.normal(1,   0.5, n_clean),  0,  2),
        "dead_code_estimate":    np.zeros(n_clean, dtype=int),
        "label": [0] * n_clean,
    }

    # ── MODERATE ───────────────────────────────────────────────────────
    moderate = {
        "num_functions":         iclamp(rng.normal(6,   2,   n_moderate),  2, 20),
        "num_loops":             iclamp(rng.normal(5,   1.5, n_moderate),  2, 12),
        "max_nesting_depth":     iclamp(rng.normal(2,   0.5, n_moderate),  1,  3),
        "num_conditionals":      iclamp(rng.normal(8,   2.5, n_moderate),  3, 16),
        "num_pointer_uses":      iclamp(rng.normal(5,   2,   n_moderate),  1, 12),
        "num_mallocs":           iclamp(rng.normal(3,   1.5, n_moderate),  1,  7),
        "num_frees":             iclamp(rng.normal(2,   1,   n_moderate),  0,  6),
        "memory_leak_count":     iclamp(rng.normal(1,   0.5, n_moderate),  1,  2),
        "unsafe_function_count": iclamp(rng.normal(1,   0.5, n_moderate),  1,  2),
        "cyclomatic_complexity": iclamp(rng.normal(11,  2,   n_moderate),  8, 16),
        "recursion_count":       iclamp(rng.normal(0.5, 0.5, n_moderate),  0,  2),
        "lines_of_code":         iclamp(rng.normal(160, 50,  n_moderate), 60,300),
        "function_call_count":   iclamp(rng.normal(15,  5,   n_moderate),  6, 35),
        # New features — moderate: sparse comments, some magic numbers
        "comment_ratio":         clamp(rng.normal(0.10, 0.04, n_moderate), 0.02, 0.20).round(2),
        "magic_number_count":    iclamp(rng.normal(6,   2,   n_moderate),  3, 12),
        "exception_handling":    iclamp(rng.normal(1,   0.7, n_moderate),  0,  3),
        "halstead_volume":       iclamp(rng.normal(600, 120, n_moderate),200,1000),
        "duplicate_block_score": iclamp(rng.normal(3,   1,   n_moderate),  1,  5),
        "dead_code_estimate":    iclamp(rng.normal(1,   0.5, n_moderate),  0,  3),
        "label": [1] * n_moderate,
    }

    # ── HIGH RISK ─────────────────────────────────────────────────────
    high = {
        "num_functions":         iclamp(rng.normal(12,  4,   n_high),   5, 40),
        "num_loops":             iclamp(rng.normal(12,  3,   n_high),   5, 30),
        "max_nesting_depth":     iclamp(rng.normal(4,   0.8, n_high),   3,  8),
        "num_conditionals":      iclamp(rng.normal(18,  4,   n_high),   8, 40),
        "num_pointer_uses":      iclamp(rng.normal(18,  5,   n_high),   8, 40),
        "num_mallocs":           iclamp(rng.normal(9,   2.5, n_high),   5, 20),
        "num_frees":             iclamp(rng.normal(3,   1.5, n_high),   0,  8),
        "memory_leak_count":     iclamp(rng.normal(6,   1.5, n_high),   3, 12),
        "unsafe_function_count": iclamp(rng.normal(5,   1.5, n_high),   2, 10),
        "cyclomatic_complexity": iclamp(rng.normal(24,  4,   n_high),  18, 50),
        "recursion_count":       iclamp(rng.normal(2,   0.8, n_high),   1,  6),
        "lines_of_code":         iclamp(rng.normal(450, 100, n_high), 200,1000),
        "function_call_count":   iclamp(rng.normal(35,  8,   n_high),  15, 80),
        # New features — high risk: no comments, many magic numbers, no error handling
        "comment_ratio":         clamp(rng.normal(0.02, 0.01, n_high), 0.0, 0.06).round(2),
        "magic_number_count":    iclamp(rng.normal(18,  4,   n_high),  10, 40),
        "exception_handling":    iclamp(rng.normal(0.2, 0.3, n_high),   0,  1),
        "halstead_volume":       iclamp(rng.normal(1800,300, n_high), 900,4000),
        "duplicate_block_score": iclamp(rng.normal(7,   1.5, n_high),   4, 10),
        "dead_code_estimate":    iclamp(rng.normal(5,   1.5, n_high),   2, 12),
        "label": [2] * n_high,
    }

    frames = [pd.DataFrame(p) for p in (clean, moderate, high)]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = _add_engineered(df)
    df["label_name"] = df["label"].map(LABEL_MAP)
    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    out_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df)} samples, {len(FEATURE_NAMES)} features → {out_path}")
    print(df["label_name"].value_counts())
