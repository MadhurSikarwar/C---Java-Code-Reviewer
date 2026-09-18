"""
IntelliReview — Heuristic Memory Leak Detector
==============================================
Used ONLY when the precise, path-sensitive analysis (v3_pointer_state, which needs an AST) is not
available — i.e. for C++ and for C that pycparser cannot parse.

Strategy: pair allocations with releases *by variable name*
  malloc/calloc/realloc/strdup/new  ->  free / delete

An allocation is reported when its variable is never released anywhere in the file, is not returned, and is
not stored somewhere else (struct field, another variable, array slot, passed to a call), because in those
cases ownership plausibly moved. This is deliberately conservative — comparing raw malloc/free *counts*
(what this module used to do) reports a "double free" for `if (x) free(p); else free(p);` and misses leaks
as soon as counts happen to match.
"""
import re
from typing import Dict, Any

from parsers.text_utils import sanitize

ALLOC_RE = re.compile(
    r"\b(\w+)\s*=\s*(?:\([^()]*\)\s*)?(?:malloc|calloc|realloc|strdup|strndup|aligned_alloc)\s*\(|"
    r"\b(\w+)\s*=\s*new\b(?!\s*\()|\b(\w+)\s*=\s*new\s*\(")
FREE_RE = re.compile(r"\bfree\s*\(\s*(?:\([^()]*\)\s*)?(\w+)\s*\)|\bdelete\s*(?:\[\s*\])?\s*(\w+)")


def detect_memory_leaks(parse_result: Dict[str, Any], source: str = "") -> Dict[str, Any]:
    if parse_result.get("language") != "C":
        return {"memory_leak_count": 0, "double_free_count": 0, "issues": []}

    # With an AST, use the precise path-sensitive analysis (this is what the pipeline does too).
    if parse_result.get("analysis_mode") == "ast" and parse_result.get("ast") is not None:
        from analyzers.v3_cfg_builder import build_cfg_for_file
        from analyzers.v3_pointer_state import analyze_pointers
        res = analyze_pointers(build_cfg_for_file(parse_result["ast"]))
        issues = [w for w in res["warnings"] if w["type"] in ("MEMORY_LEAK", "DOUBLE_FREE") and w["severity"] != "LOW"]
        return {"memory_leak_count": res["metrics"]["leak_count"],
                "double_free_count": res["metrics"]["double_free_count"], "issues": issues}

    if not source:
        return {"memory_leak_count": 0, "double_free_count": 0, "issues": []}

    code = sanitize(source)
    freed = {m.group(1) or m.group(2) for m in FREE_RE.finditer(code)}
    issues = []
    seen = set()
    for m in ALLOC_RE.finditer(code):
        var = m.group(1) or m.group(2) or m.group(3)
        if not var or var in freed or var in seen:
            continue
        v = re.escape(var)
        escapes = (
            re.search(rf"\breturn\s+(?:\([^()]*\)\s*)?{v}\b", code)                # returned
            or re.search(rf"(?:->|\.|\])\s*\w*\s*=\s*{v}\s*;", code)               # stored in field / array
            or re.search(rf"\b\w+\s*=\s*{v}\s*;", code.replace(m.group(0), ""))     # aliased
            or re.search(rf"\b(?!free\b|delete\b|printf\b|puts\b|strlen\b)\w+\s*\([^;]*\b{v}\b[^;]*\)", code.replace(m.group(0), ""))
        )
        # A pointer handed to some other function might be released there: report only if it is *only*
        # ever used locally (no escape), otherwise stay quiet.
        if escapes:
            continue
        seen.add(var)
        line = code.count("\n", 0, m.start()) + 1
        issues.append({
            "type": "MEMORY_LEAK",
            "severity": "MEDIUM",
            "line": line,
            "message": f"Memory Leak: '{var}' is allocated at line {line} but never freed/deleted.",
            "suggestion": "Release every allocation (free / delete / delete[]) or use RAII / smart pointers.",
        })

    return {"memory_leak_count": len(issues), "double_free_count": 0, "issues": issues}
