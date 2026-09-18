"""
IntelliReview — static analysis pipeline (everything except the ML models).

`static_analysis(source, language)` is the single place where source code becomes
  * a list of aggregated issues, and
  * the 36-dimensional feature vector every model consumes:
        21 classic features (analyzers.feature_extractor)
      + 13 path-sensitive V3 features (pointer / data-flow / CFG)
      +  2 V4 evidence features (dedup_path_occurrences, scaled_severity_score)

It is used by the API route AND by the training scripts, so the models are trained on exactly the
features they will see at inference time (the old code computed the last two features nowhere, so
the V4 model always received zeros for them).
"""
import sys
from typing import Dict, Any, List

from parsers.c_parser import parse_c_code
from parsers.java_parser import parse_java_code
from analyzers.memory_leak import detect_memory_leaks
from analyzers.unsafe_functions import detect_unsafe_functions
from analyzers.java_security import detect_java_security
from analyzers.complexity import analyze_complexity
from analyzers.recursion import detect_recursion
from analyzers.feature_extractor import extract_features
from analyzers.v3_cfg_builder import build_cfg_for_file
from analyzers.v3_pointer_state import analyze_pointers
from analyzers.v3_data_flow import analyze_data_flow
from analyzers.v3_code_smells import analyze_smells
from analyzers.v3_complexity import analyze_time_complexity
from analyzers.v3_feature_extractor import extract_v3_features
from analyzers.v4_issue_aggregator import aggregate_issues
from analyzers.suppress import apply_suppressions
from analyzers.fixes import attach_fixes

V3_ORDER = [
    "use_after_free_count", "double_free_count", "path_leak_probability",
    "pointer_state_transitions", "infinite_loop_risks", "uninitialized_vars_used",
    "cfg_node_count", "branch_density", "cyclomatic_complexity",
    "global_mutation_count", "loop_count", "max_loop_depth", "recursion_count",
]
V4_EXTRA = ["dedup_path_occurrences", "scaled_severity_score"]
ALL_FEATURE_NAMES: List[str] = []   # filled lazily (needs ml.synthetic_data)

SEVERITY_WEIGHT = {"LOW": 0.25, "MEDIUM": 0.5, "HIGH": 0.75, "CRITICAL": 1.0}

from analyzers.issue_taxonomy import NON_SECURITY_TYPES, QUALITY_TYPES, risk_level_from_issues  # noqa: E402,F401


def all_feature_names() -> List[str]:
    from ml.synthetic_data import FEATURE_NAMES
    return list(FEATURE_NAMES) + ["v3_" + n for n in V3_ORDER] + V4_EXTRA


def v4_extra_features(issues: List[Dict[str, Any]]) -> List[float]:
    if not issues:
        return [0.0, 0.0]
    occ = sum(int(i.get("path_occurrences", 1)) for i in issues)
    sev = sum(SEVERITY_WEIGHT.get(str(i.get("severity", "LOW")).upper(), 0.25) for i in issues) / len(issues)
    return [float(occ), round(sev, 3)]


def static_analysis(source: str, language: str) -> Dict[str, Any]:
    """Run parsing + every analyzer. `language` is 'C' or 'JAVA' (already normalised)."""
    language = language.upper()
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 6000))    # deeply nested (but valid) code
    try:
        return _static_analysis(source, language)
    finally:
        sys.setrecursionlimit(old_limit)


def _static_analysis(source: str, language: str) -> Dict[str, Any]:
    is_c = language == "C"
    parse_result = parse_c_code(source) if is_c else parse_java_code(source)
    notes: List[str] = []

    complexity_result = analyze_complexity(parse_result, source)

    cfgs: List[Dict[str, Any]] = []
    v3_issues: List[Dict[str, Any]] = []
    v3_features = {k: 0.0 for k in V3_ORDER}
    p_res = {"metrics": {}, "warnings": []}

    if is_c and parse_result.get("analysis_mode") == "ast" and parse_result.get("ast") is not None:
        try:
            cfgs = build_cfg_for_file(parse_result["ast"])
            p_res = analyze_pointers(cfgs)
            d_res = analyze_data_flow(source, cfgs)
            s_res = analyze_smells(source, cfgs)
            c_res = analyze_time_complexity(cfgs)
            v3_features.update(extract_v3_features(p_res, d_res, s_res, c_res))
            v3_issues = list(p_res["warnings"]) + list(d_res["warnings"]) + list(s_res["warnings"])
            if p_res.get("truncated"):
                notes.append("Pointer analysis hit its state budget on a very branchy function; "
                             "results for that function may be incomplete.")
        except RecursionError:
            notes.append("Code is nested too deeply for path-sensitive analysis; used structural metrics only.")
            cfgs, v3_issues = [], []
        except Exception as e:  # analyzer bug must not take the request down; surface it instead of hiding it
            notes.append(f"Path-sensitive analysis failed ({type(e).__name__}); used structural metrics only.")
            cfgs, v3_issues = [], []

    # Recursion: precise call graph if we have CFGs, text heuristic otherwise
    recursion_result = detect_recursion(parse_result, source, cfgs=cfgs if cfgs else None)

    # Memory: the CFG analysis is authoritative when available; otherwise fall back to name pairing
    if cfgs:
        memory_result = {"memory_leak_count": int(p_res["metrics"].get("leak_count", 0)),
                         "double_free_count": int(p_res["metrics"].get("double_free_count", 0)), "issues": []}
    elif is_c:
        memory_result = detect_memory_leaks(parse_result, source)
    else:
        memory_result = {"memory_leak_count": 0, "double_free_count": 0, "issues": []}

    if is_c:
        unsafe_result = detect_unsafe_functions(parse_result)
    else:
        unsafe_result = detect_java_security(source)

    feature_data = extract_features(parse_result, memory_result, unsafe_result, complexity_result,
                                    recursion_result, source=source)

    raw_issues = (memory_result.get("issues", []) + unsafe_result.get("issues", [])
                  + complexity_result.get("issues", []) + recursion_result.get("issues", []) + v3_issues)
    all_issues = aggregate_issues(raw_issues)
    all_issues, suppressed = apply_suppressions(source, all_issues)   # `// intellireview: ignore`
    attach_fixes(source, language, all_issues)                          # one-line repairs the UI can apply
    if suppressed:
        notes.append(f"{len(suppressed)} finding(s) were suppressed by intellireview: ignore comments.")

    vector = list(feature_data["feature_vector"])            # 21
    vector += [float(v3_features.get(k, 0.0)) for k in V3_ORDER]   # +13 = 34
    vector += v4_extra_features(all_issues)                  # +2 = 36

    return {
        "parse_result": parse_result,
        "memory_result": memory_result,
        "unsafe_result": unsafe_result,
        "complexity_result": complexity_result,
        "recursion_result": recursion_result,
        "feature_data": feature_data,
        "feature_vector": vector,
        "v3_issues": v3_issues,
        "cfgs": cfgs,
        "all_issues": all_issues,
        "suppressed": suppressed,
        "notes": notes,
        "v3_leak_count": int(p_res["metrics"].get("leak_count", 0)),
    }
