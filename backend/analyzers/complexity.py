"""
IntelliReview — Cyclomatic Complexity & Time Complexity Analyzer
Computes per-function and overall complexity metrics.
"""
import re
from typing import Dict, Any, List


# Decision keywords that increase cyclomatic complexity
DECISION_KEYWORDS_C = re.compile(
    r'\b(if|else\s+if|for|while|do|case|catch|&&|\|\|)\b'
)
DECISION_KEYWORDS_JAVA = re.compile(
    r'\b(if|else\s+if|for|while|do|case|catch|&&|\|\|)\b'
)

TIME_COMPLEXITY_MAP = {
    0: "O(1)",
    1: "O(n)",
    2: "O(n²)",
    3: "O(n³)",
    4: "O(n⁴) — consider algorithm redesign",
}


def analyze_complexity(parse_result: Dict[str, Any], source: str) -> Dict[str, Any]:
    """
    Compute:
      1. Cyclomatic complexity per function and overall
      2. Estimated time complexity from max loop nesting depth
      3. Per-function complexity breakdown
    """
    language = parse_result.get("language", "C")
    functions = parse_result.get("functions", [])
    lines = source.splitlines()

    decision_pattern = (
        DECISION_KEYWORDS_C if language == "C" else DECISION_KEYWORDS_JAVA
    )

    func_complexity = []
    issues = []

    if functions:
        func_list = sorted(functions, key=lambda f: f.get("line", 0))
        for idx, func in enumerate(func_list):
            func_start = func.get("line", 0)
            func_end = (
                func_list[idx + 1].get("line", len(lines))
                if idx + 1 < len(func_list)
                else len(lines)
            )

            func_body = "\n".join(lines[func_start - 1 : func_end - 1])
            # CC = number of decision points + 1
            decisions = len(decision_pattern.findall(func_body))
            cc = decisions + 1

            func_complexity.append({
                "function": func["name"],
                "line": func_start,
                "cyclomatic_complexity": cc,
                "risk": _cc_risk(cc),
            })

            if cc > 10:
                issues.append({
                    "type": "HIGH_COMPLEXITY",
                    "severity": "HIGH" if cc > 15 else "MEDIUM",
                    "function": func["name"],
                    "line": func_start,
                    "cc": cc,
                    "message": (
                        f"Function `{func['name']}` has cyclomatic complexity of {cc}, "
                        f"which is {'very high' if cc > 15 else 'high'}. "
                        "Complex functions are harder to test and maintain."
                    ),
                    "suggestion": (
                        "Refactor into smaller, single-purpose functions. "
                        "Aim for CC ≤ 10 per function."
                    ),
                })
    else:
        # No function info — compute overall CC from source
        decisions = len(decision_pattern.findall(source))
        cc = decisions + 1
        func_complexity.append({
            "function": "__global__",
            "line": 1,
            "cyclomatic_complexity": cc,
            "risk": _cc_risk(cc),
        })

    overall_cc = max((f["cyclomatic_complexity"] for f in func_complexity), default=1)

    # Nesting depth → estimated time complexity
    depth = min(parse_result.get("max_nesting_depth", 0), 4)
    time_complexity = TIME_COMPLEXITY_MAP.get(depth, f"O(n^{depth})")

    # Nested loop depth issues
    if depth >= 3:
        issues.append({
            "type": "DEEP_NESTING",
            "severity": "HIGH" if depth >= 4 else "MEDIUM",
            "line": 0,
            "message": (
                f"Maximum loop nesting depth of {depth} detected. "
                f"Estimated time complexity: {time_complexity}."
            ),
            "suggestion": (
                "Reduce nesting depth by extracting inner loops into functions "
                "or rethinking the algorithm (e.g., use hash maps to avoid O(n²) lookups)."
            ),
        })

    return {
        "cyclomatic_complexity": overall_cc,
        "time_complexity": time_complexity,
        "max_nesting_depth": depth,
        "function_complexity": func_complexity,
        "issues": issues,
    }


def _cc_risk(cc: int) -> str:
    if cc <= 5:
        return "LOW"
    elif cc <= 10:
        return "MODERATE"
    elif cc <= 15:
        return "HIGH"
    else:
        return "VERY HIGH"
