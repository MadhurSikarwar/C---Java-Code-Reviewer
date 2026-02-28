"""
IntelliReview — Memory Leak Detector (C-specific)
Identifies malloc/calloc/realloc calls without corresponding free().
"""
from typing import Dict, Any, List


def detect_memory_leaks(parse_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze malloc/free pairings to find potential memory leaks.
    
    Strategy:
      - Count malloc vs free calls
      - Flag any malloc line that doesn't have a corresponding free
      - Detect double-free (more frees than mallocs)
    
    Returns dict with issues and counts.
    """
    issues = []

    if parse_result.get("language") != "C":
        return {"memory_leak_count": 0, "double_free_count": 0, "issues": []}

    malloc_lines = parse_result.get("malloc_lines", [])
    free_lines = parse_result.get("free_lines", [])
    num_mallocs = parse_result.get("num_mallocs", 0)
    num_frees = parse_result.get("num_frees", 0)

    # Simple heuristic: if more mallocs than frees → potential leak
    leak_count = max(0, num_mallocs - num_frees)

    if leak_count > 0:
        # Flag the unpaired malloc lines
        paired = num_frees
        unmatched_mallocs = malloc_lines[paired:]  # approximate
        for line in unmatched_mallocs:
            issues.append({
                "type": "MEMORY_LEAK",
                "severity": "HIGH",
                "line": line,
                "message": f"malloc() at line {line} may not have a corresponding free() — potential memory leak.",
                "suggestion": "Ensure every malloc/calloc/realloc has a matching free() in all code paths.",
            })

    # Detect double free (more frees than mallocs)
    double_free_count = max(0, num_frees - num_mallocs)
    if double_free_count > 0:
        extra_frees = free_lines[num_mallocs:]
        for line in extra_frees:
            issues.append({
                "type": "DOUBLE_FREE",
                "severity": "CRITICAL",
                "line": line,
                "message": f"Potential double free() at line {line} — more free() calls than allocations.",
                "suggestion": "Set pointer to NULL immediately after free() to prevent double-free vulnerabilities.",
            })

    # Check for allocs inside loops without frees inside the same loop
    # (heuristic: if malloc_lines have adjacent lines that suggest loop context)
    source_line_context = parse_result.get("_source_lines", [])

    return {
        "memory_leak_count": leak_count,
        "double_free_count": double_free_count,
        "issues": issues,
    }
