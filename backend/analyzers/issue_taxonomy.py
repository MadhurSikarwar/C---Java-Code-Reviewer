"""
IntelliReview — which issue types may influence the RISK verdict.

QUALITY_TYPES        style / maintainability observations (nesting, complexity, dead code, recursion). They are
                     reported to the user but never change the risk label: a deeply nested function is not "risky".
NON_SECURITY_TYPES   everything that must never justify "High Risk" on its own (quality smells plus reliability
                     issues such as an uninitialised read or a leak, which cap at Moderate).
Anything else that reaches CRITICAL/HIGH severity (memory corruption, injection, ...) is High Risk evidence.
"""
QUALITY_TYPES = {"HIGH_COMPLEXITY", "DEEP_NESTING", "DEAD_CODE", "RECURSION"}

NON_SECURITY_TYPES = QUALITY_TYPES | {
    "INFINITE_LOOP", "UNINITIALIZED_VARIABLE", "EMPTY_CATCH", "RESOURCE_LEAK", "MEMORY_LEAK", "UNCHECKED_ALLOC",
}


def risk_level_from_issues(issues) -> int:
    """0 = Clean, 1 = Moderate, 2 = High — what the rule engine alone concludes."""
    if any(i.get("severity") in ("CRITICAL", "HIGH") and i.get("type") not in NON_SECURITY_TYPES for i in issues):
        return 2
    if any(i.get("severity") in ("CRITICAL", "HIGH", "MEDIUM") and i.get("type") not in QUALITY_TYPES for i in issues):
        return 1
    return 0
