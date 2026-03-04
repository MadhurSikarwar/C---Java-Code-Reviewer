import copy

def aggregate_issues(raw_issues):
    """
    Takes a raw list of issues (which may contain hundreds of duplicate paths
    for the same vulnerability, a classic SSA path-explosion side-effect) and
    merges them into a clean, deduplicated list.
    
    It categorizes duplicates by:
    - Line Number
    - Vulnerability Type
    - Variable Name (extracted from message or explicitly if available)
    
    Instead of 200 outputs, it returns 1 output and appends the path frequency.
    """
    if not raw_issues:
        return []

    # Map to hold uniquely identifiable issues
    # Key: (file_line, issue_type, primary_target)
    unique_issues_map = {}

    for issue in raw_issues:
        # 1. Extract Core Identifiers to establish a uniqueness signature
        line = issue.get("line", -1)
        i_type = issue.get("type", "UNKNOWN_ISSUE")
        message = issue.get("message", "")
        
        # We attempt to extract the target variable from standard messages
        # E.g., "Variable 'sum' is used before assignment" -> "sum"
        # E.g., "Memory leak detected for pointer 'temp'" -> "temp"
        target_var = extract_target_variable(message)
        
        signature = (line, i_type, target_var)
        
        if signature not in unique_issues_map:
            # First time seeing this exact vulnerability on this exact path line
            cloned_issue = copy.deepcopy(issue)
            cloned_issue["path_occurrences"] = 1
            cloned_issue["path_frequency"] = "Found on 1 unique execution path."
            unique_issues_map[signature] = cloned_issue
        else:
            # We hit a Path Explosion! This is the same issue down a different CFG branch.
            # We increment the occurrence counter instead of duplicating the entire JSON object.
            existing = unique_issues_map[signature]
            existing["path_occurrences"] += 1
            existing["path_frequency"] = f"Found across {existing['path_occurrences']} distinct execution paths."

            # If the current issue has a higher severity state (e.g., it was returned vs just read),
            # we escalate the deduplicated envelope's severity to match the worst-case path.
            existing["severity"] = resolve_highest_severity(existing.get("severity", "LOW"), issue.get("severity", "LOW"))

    # Convert the deduplicated map back to a standard flat list for the Frontend
    return list(unique_issues_map.values())

def extract_target_variable(message):
    """
    Helper to pull the variable/pointer name out of standard V3 Warning messages 
    so we can deduplicate across it.
    """
    import re
    # Matches: 'var_name', "var_name", or `var_name`
    match = re.search(r"['\"`]([^'\"`]+)['\"`]", message)
    if match:
        return match.group(1)
    
    # Fallback if no quotes are used and it's a known format
    if "Memory leak" in message:
        return "pointer_leak"
    
    return "unknown_target"

def resolve_highest_severity(sev_a, sev_b):
    """
    If the same issue is found on 2 paths, but Path B constitutes a CRITICAL 
    dereference while Path A constitutes a MEDIUM read, the merged issue MUST be CRITICAL.
    """
    weights = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    weight_a = weights.get(sev_a.upper(), 0)
    weight_b = weights.get(sev_b.upper(), 0)
    
    return sev_a if weight_a >= weight_b else sev_b
