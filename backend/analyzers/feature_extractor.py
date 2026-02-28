"""
IntelliReview — Feature Extractor  (19-feature version)
Merges all analysis outputs into a flat numerical vector for ML.

Features:
  13 original structural features
   2 engineered ratios (leak_ratio, unsafe_density)
   6 new code quality signals (comment_ratio, magic_number_count, ...)
"""
from typing import Dict, Any, List
from ml.synthetic_data import FEATURE_NAMES          # single source of truth
from analyzers.code_quality import analyze_code_quality


def extract_features(
    parse_result: Dict[str, Any],
    memory_result: Dict[str, Any],
    unsafe_result: Dict[str, Any],
    complexity_result: Dict[str, Any],
    recursion_result: Dict[str, Any],
    source: str = "",
) -> Dict[str, Any]:
    """
    Combine all analysis outputs into a single feature dict ordered
    to match FEATURE_NAMES from synthetic_data.py.
    """
    # ── 13 original structural features ────────────────────────────────
    features = {
        "num_functions":         parse_result.get("num_functions", 0),
        "num_loops":             parse_result.get("num_loops", 0),
        "max_nesting_depth":     parse_result.get("max_nesting_depth", 0),
        "num_conditionals":      parse_result.get("num_conditionals", 0),
        "num_pointer_uses":      parse_result.get("num_pointer_uses", 0),
        "num_mallocs":           parse_result.get("num_mallocs", 0),
        "num_frees":             parse_result.get("num_frees", 0),
        "memory_leak_count":     memory_result.get("memory_leak_count", 0),
        "unsafe_function_count": unsafe_result.get("unsafe_function_count", 0),
        "cyclomatic_complexity": complexity_result.get("cyclomatic_complexity", 1),
        "recursion_count":       recursion_result.get("recursion_count", 0),
        "lines_of_code":         parse_result.get("lines_of_code", 0),
        "function_call_count":   parse_result.get("function_call_count", 0),
    }

    # ── 2 engineered ratio features ─────────────────────────────────────
    n_mallocs   = max(features["num_mallocs"], 1)
    n_functions = max(features["num_functions"], 1)
    features["leak_ratio"]     = round(min(features["memory_leak_count"] / n_mallocs, 1.0), 2)
    features["unsafe_density"] = round(min(features["unsafe_function_count"] / n_functions, 5.0), 2)

    # ── 6 new code quality signal features ──────────────────────────────
    language = parse_result.get("language", "C")
    if source:
        quality = analyze_code_quality(source, language)
        features.update(quality)
    else:
        # Fallback zeros if no source provided
        features.update({
            "comment_ratio": 0.0,
            "magic_number_count": 0,
            "exception_handling": 0,
            "halstead_volume": 0,
            "duplicate_block_score": 0,
            "dead_code_estimate": 0,
        })

    # ── Ensure all expected features present (fill any gaps) ───────────
    for fname in FEATURE_NAMES:
        features.setdefault(fname, 0)

    # ── Build ordered vector for ML model ───────────────────────────────
    vector = [float(features[f]) for f in FEATURE_NAMES]

    return {
        "feature_vector": vector,
        "feature_names":  FEATURE_NAMES,
        "features":       features,
    }
