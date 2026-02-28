"""
IntelliReview — Suggestion Generator
Aggregates all issue lists into prioritized, actionable suggestions
with severity levels and concrete fix examples.
"""
from typing import Dict, Any, List


SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

# General good-practice suggestions based on feature thresholds
GENERAL_TIPS = [
    {
        "condition": lambda f: f.get("max_nesting_depth", 0) >= 3,
        "severity": "HIGH",
        "category": "Code Structure",
        "message": "Deep nesting (depth ≥ 3) detected.",
        "suggestion": "Extract inner logic into named helper functions to reduce nesting. Consider guard clauses (early returns) to flatten nested conditionals.",
        "example": "// Instead of deep nesting:\nif (a) { if (b) { ... } }\n// Use guard clause:\nif (!a) return;\nif (!b) return;\n// ... main logic here",
    },
    {
        "condition": lambda f: f.get("cyclomatic_complexity", 1) > 15,
        "severity": "HIGH",
        "category": "Complexity",
        "message": "Very high cyclomatic complexity (> 15).",
        "suggestion": "Break complex functions into smaller, testable units. Each function should do one thing. Aim for CC ≤ 10.",
        "example": None,
    },
    {
        "condition": lambda f: f.get("cyclomatic_complexity", 1) > 10,
        "severity": "MEDIUM",
        "category": "Complexity",
        "message": "Cyclomatic complexity > 10 — function is hard to test.",
        "suggestion": "Consider splitting this function. Use polymorphism or strategy pattern instead of long if-else chains.",
        "example": None,
    },
    {
        "condition": lambda f: f.get("num_functions", 1) == 0,
        "severity": "MEDIUM",
        "category": "Code Structure",
        "message": "No functions detected — all code is in global scope.",
        "suggestion": "Organize code into functions/methods. This improves reusability, testability, and readability.",
        "example": None,
    },
    {
        "condition": lambda f: f.get("recursion_count", 0) > 0,
        "severity": "MEDIUM",
        "category": "Performance",
        "message": "Recursive function(s) detected.",
        "suggestion": "Ensure all recursive functions have a clear base case. For large inputs or performance-critical paths, consider an iterative approach with an explicit stack.",
        "example": None,
    },
    {
        "condition": lambda f: f.get("num_mallocs", 0) > 0 and f.get("num_frees", 0) == 0,
        "severity": "HIGH",
        "category": "Memory Management",
        "message": "Memory is allocated but never freed.",
        "suggestion": "Every malloc/calloc must have a corresponding free(). Use a cleanup label pattern for C error handling to ensure memory is always released.",
        "example": "int *p = malloc(n * sizeof(int));\nif (!p) goto cleanup;\n// ... use p ...\ncleanup:\nfree(p);\np = NULL;",
    },
    {
        "condition": lambda f: f.get("lines_of_code", 0) > 300,
        "severity": "LOW",
        "category": "Maintainability",
        "message": "Large file (> 300 lines of code).",
        "suggestion": "Consider splitting into multiple modules. Large files are hard to navigate, review, and test.",
        "example": None,
    },
]


def generate_suggestions(
    features: Dict[str, Any],
    memory_issues: List[Dict],
    unsafe_issues: List[Dict],
    complexity_issues: List[Dict],
    recursion_issues: List[Dict],
) -> Dict[str, Any]:
    """
    Combine all detected issues into a unified, prioritized suggestion list.
    """
    all_suggestions = []

    # Pull suggestions from each analyzer's issues
    for issue in memory_issues + unsafe_issues + complexity_issues + recursion_issues:
        if "suggestion" in issue:
            all_suggestions.append({
                "severity": issue.get("severity", "MEDIUM"),
                "category": _categorize(issue.get("type", "")),
                "message": issue.get("message", ""),
                "suggestion": issue.get("suggestion", ""),
                "line": issue.get("line", 0),
                "example": None,
            })

    # Add general tips based on feature values
    for tip in GENERAL_TIPS:
        try:
            if tip["condition"](features):
                all_suggestions.append({
                    "severity": tip["severity"],
                    "category": tip["category"],
                    "message": tip["message"],
                    "suggestion": tip["suggestion"],
                    "line": 0,
                    "example": tip.get("example"),
                })
        except Exception:
            pass

    # Deduplicate by message prefix
    seen = set()
    unique = []
    for s in all_suggestions:
        key = s["message"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    # Sort by severity
    unique.sort(key=lambda s: SEVERITY_ORDER.get(s["severity"], 99))

    return {
        "suggestions": unique,
        "total_suggestions": len(unique),
        "critical_count": sum(1 for s in unique if s["severity"] == "CRITICAL"),
        "high_count": sum(1 for s in unique if s["severity"] == "HIGH"),
        "medium_count": sum(1 for s in unique if s["severity"] == "MEDIUM"),
        "low_count": sum(1 for s in unique if s["severity"] == "LOW"),
    }


def _categorize(issue_type: str) -> str:
    return {
        "MEMORY_LEAK": "Memory Management",
        "DOUBLE_FREE": "Memory Management",
        "UNSAFE_FUNCTION": "Security",
        "HIGH_COMPLEXITY": "Complexity",
        "DEEP_NESTING": "Code Structure",
        "RECURSION": "Performance",
    }.get(issue_type, "General")
