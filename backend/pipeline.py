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
import re
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


# ------------------------------------------------------------------------------------------------ trace helpers
_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
_TYPE_WORDS = {
    "UNSAFE_FUNCTION": "unsafe library call", "USE_AFTER_FREE": "use after free", "DOUBLE_FREE": "double free",
    "NULL_DEREF": "null dereference", "BUFFER_OVERFLOW": "buffer overflow", "FORMAT_STRING": "format-string bug",
    "MEMORY_LEAK": "memory leak", "RESOURCE_LEAK": "resource leak", "UNINITIALIZED_VARIABLE": "uninitialized variable",
    "SQL_INJECTION": "SQL injection", "COMMAND_INJECTION": "command injection", "UNCHECKED_ALLOC": "unchecked allocation",
}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else ('es' if word.endswith('s') else 's')}"


def _emit_parse(emit, pr: Dict[str, Any], is_c: bool) -> None:
    fns, loops = pr.get("num_functions", 0), pr.get("num_loops", 0)
    learned = pr.get("learned_types") or []
    if pr.get("analysis_mode") == "ast":
        first = (pr.get("functions") or [{}])[0].get("line")
        emit({"stage": "parse", "kind": "step", "line": first,
              "text": f"Parsed it into a syntax tree: {_plural(fns, 'function')}, {_plural(loops, 'loop')}."})
        if learned:
            emit({"stage": "parse", "kind": "step",
                  "text": f"It uses types from headers we can't see ({', '.join(learned[:4])}{'…' if len(learned) > 4 else ''}); "
                          "learned them and parsed again."})
    else:
        why = "it looks like C++" if pr.get("cpp_like") else (pr.get("parse_errors") or ["the parser gave up"])[0][:90]
        emit({"stage": "parse", "kind": "step",
              "text": f"Couldn't build a full syntax tree ({why}). Falling back to pattern-based reading, which is less precise."})


def _emit_graph_and_pointers(emit, cfgs, p_res, d_res) -> None:
    by_fn: Dict[str, list] = {}
    for w in p_res.get("warnings", []):
        by_fn.setdefault(w.get("function"), []).append(w)
    shown = cfgs[:10]
    for c in shown:
        name, blocks = c.get("function"), len(c.get("nodes", {}))
        found = by_fn.get(name, [])
        tail = ""
        if found:
            kinds = sorted({_TYPE_WORDS.get(w["type"], w["type"].lower().replace("_", " ")) for w in found})
            tail = f" Something to report: {', '.join(kinds)}."
        emit({"stage": "graph", "kind": "step", "line": c.get("line"),
              "text": f"`{name}`: mapped {_plural(blocks, 'basic block')} and followed every pointer through each branch.{tail}"})
    if len(cfgs) > len(shown):
        emit({"stage": "graph", "kind": "step", "text": f"…and {len(cfgs) - len(shown)} more functions the same way."})
    tracked = sum(len(c.get("tracked_scalars", [])) for c in cfgs)
    m = d_res.get("metrics", {})
    emit({"stage": "dataflow", "kind": "step",
          "text": f"Checked {_plural(tracked, 'variable')} for use before assignment: "
                  f"{_plural(m.get('uninitialized_vars_used', 0), 'possibly uninitialized read')}."})


def _emit_rules(emit, is_c, pr, unsafe_result, complexity_result, recursion_result) -> None:
    issues = unsafe_result.get("issues", [])
    if is_c:
        calls = pr.get("unsafe_calls", [])
        proven = sum(1 for c in calls if c.get("function") == "buffer_overflow")
        emit({"stage": "bounds", "kind": "step",
              "text": f"Measured buffer sizes and index ranges against every copy and array access: "
                      f"{_plural(proven, 'access')} provably out of bounds."})
        emit({"stage": "taint", "kind": "step",
              "text": f"Looked at each risky library call and whether attacker-influenced data can reach it: "
                      f"{_plural(len(issues) - proven, 'call')} judged unsafe."})
    else:
        emit({"stage": "taint", "kind": "step",
              "text": f"Followed untrusted input through {_plural(pr.get('num_functions', 0), 'method')} to SQL, shell, "
                      f"file, HTML and deserialization sinks, and checked crypto and credentials: {_plural(len(issues), 'finding')}."})
    fc = complexity_result.get("function_complexity") or []
    worst = max(fc, key=lambda f: f.get("cyclomatic_complexity", 0)) if fc else None
    if worst and worst.get("cyclomatic_complexity", 0) >= 6:
        emit({"stage": "graph", "kind": "step", "line": worst.get("line"),
              "text": f"Most complex function: `{worst.get('function')}` with cyclomatic complexity {worst.get('cyclomatic_complexity')}."})
    if recursion_result.get("recursion_count"):
        emit({"stage": "graph", "kind": "step",
              "text": f"Found {_plural(recursion_result['recursion_count'], 'recursive function')}."})


def _emit_findings_and_features(emit, all_issues, suppressed, vector) -> None:
    ordered = sorted(all_issues, key=lambda i: (_SEV_ORDER.get(i.get("severity"), 9), i.get("line") or 0))
    if not ordered:
        emit({"stage": "findings", "kind": "step", "text": "No findings."})
    for i in ordered[:25]:
        msg = re.sub(r"\s+at line \d+", "", (i.get("message") or "").replace("`", ""))
        emit({"stage": "findings", "kind": "finding", "severity": i.get("severity"), "line": i.get("line") or 0,
              "type": i.get("type"), "text": msg[:150]})
    if len(ordered) > 25:
        emit({"stage": "findings", "kind": "step", "text": f"…and {len(ordered) - 25} more."})
    if suppressed:
        emit({"stage": "findings", "kind": "step", "text": f"{_plural(len(suppressed), 'finding')} silenced by intellireview: ignore comments."})
    fixable = sum(1 for i in all_issues if i.get("fix"))
    if fixable:
        emit({"stage": "findings", "kind": "step", "text": f"{_plural(fixable, 'finding')} can be repaired with a one-line edit."})

    names = all_feature_names()
    pri = ["unsafe_function_count", "memory_leak_count", "v3_use_after_free_count", "v3_double_free_count",
           "v3_uninitialized_vars_used", "v3_path_leak_probability", "dedup_path_occurrences", "scaled_severity_score",
           "lines_of_code", "num_functions", "cyclomatic_complexity", "num_mallocs", "num_frees", "num_loops"]
    shown = [[n, round(vector[names.index(n)], 2)] for n in pri if n in names and vector[names.index(n)] != 0][:9]
    emit({"stage": "numbers", "kind": "features", "features": shown,
          "text": "Condensed all of that into 36 numbers for the models. The ones that matter most here:"})


def all_feature_names() -> List[str]:
    from ml.synthetic_data import FEATURE_NAMES
    return list(FEATURE_NAMES) + ["v3_" + n for n in V3_ORDER] + V4_EXTRA


def v4_extra_features(issues: List[Dict[str, Any]]) -> List[float]:
    if not issues:
        return [0.0, 0.0]
    occ = sum(int(i.get("path_occurrences", 1)) for i in issues)
    sev = sum(SEVERITY_WEIGHT.get(str(i.get("severity", "LOW")).upper(), 0.25) for i in issues) / len(issues)
    return [float(occ), round(sev, 3)]


def static_analysis(source: str, language: str, trace=None) -> Dict[str, Any]:
    """Run parsing + every analyzer. `language` is 'C' or 'JAVA' (already normalised).
    `trace(event_dict)`, if given, is called as each stage finishes so a UI can show the engine thinking."""
    language = language.upper()
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 6000))    # deeply nested (but valid) code
    try:
        return _static_analysis(source, language, trace or (lambda e: None))
    finally:
        sys.setrecursionlimit(old_limit)


def _static_analysis(source: str, language: str, emit) -> Dict[str, Any]:
    is_c = language == "C"
    n_lines = len(source.splitlines())
    emit({"stage": "parse", "kind": "step", "text": f"Read {n_lines} line{'s' if n_lines != 1 else ''} of {'C/C++' if is_c else 'Java'}."})
    parse_result = parse_c_code(source) if is_c else parse_java_code(source)
    notes: List[str] = []
    _emit_parse(emit, parse_result, is_c)

    complexity_result = analyze_complexity(parse_result, source)

    cfgs: List[Dict[str, Any]] = []
    v3_issues: List[Dict[str, Any]] = []
    v3_features = {k: 0.0 for k in V3_ORDER}
    p_res = {"metrics": {}, "warnings": []}

    if is_c and parse_result.get("analysis_mode") == "ast" and parse_result.get("ast") is not None:
        try:
            cfgs = build_cfg_for_file(parse_result["ast"])
            # pointer/resource states are followed through same-file calls on the analysis copy (parsers/c_inline.py)
            p_res = analyze_pointers(build_cfg_for_file(parse_result.get("ast_analysis") or parse_result["ast"]))
            d_res = analyze_data_flow(source, cfgs)
            s_res = analyze_smells(source, cfgs)
            c_res = analyze_time_complexity(cfgs)
            v3_features.update(extract_v3_features(p_res, d_res, s_res, c_res))
            v3_issues = list(p_res["warnings"]) + list(d_res["warnings"]) + list(s_res["warnings"])
            _emit_graph_and_pointers(emit, cfgs, p_res, d_res)
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
    _emit_rules(emit, is_c, parse_result, unsafe_result, complexity_result, recursion_result)

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
    _emit_findings_and_features(emit, all_issues, suppressed, vector)

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
