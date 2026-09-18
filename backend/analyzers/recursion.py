"""
IntelliReview — Recursion Detector
Detects recursive functions and classifies them as Linear or Exponential using Call Graphs.
"""
import re
from typing import Dict, Any, List

from parsers.text_utils import sanitize

def detect_recursion(parse_result: Dict[str, Any], source: str = "", cfgs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    functions = parse_result.get("functions", [])
    language = parse_result.get("language", "C")
    issues = []

    if not source:
        return {"recursion_count": 0, "recursive_functions": [], "issues": []}

    # Scan code only: a comment or string literal that mentions the function's name is not a recursive call.
    lines = sanitize(source, java_text_blocks=(language != "C")).splitlines()
    
    # 1. Fallback Heuristic Execution (if CFGs unavailable)
    if cfgs is None:
        for func in functions:
            name = func["name"]
            if not name or name in ("main", "Main"): continue
            func_line = func.get("line", 0)
            if func_line == 0: continue
            
            in_body, brace_depth, found_self_call, call_line = False, 0, False, 0
            for i, line in enumerate(lines, 1):
                if i < func_line: continue
                if i == func_line: in_body = True
                if in_body:
                    brace_depth += line.count('{') - line.count('}')
                    if i > func_line and re.search(rf'\b{re.escape(name)}\s*\(', line):
                        found_self_call = True
                        call_line = i
                        break
                    if brace_depth <= 0 and i > func_line: break
                    
            if found_self_call:
                issues.append({
                    "type": "RECURSION",
                    "severity": "LOW",
                    "function": name,
                    "line": func_line,
                    "call_line": call_line,
                    "message": f"Function `{name}` calls itself at line {call_line}. Deep recursion risks stack overflow.",
                    "suggestion": "Convert to an iterative approach for large inputs."
                })
        
        return {"recursion_count": len(issues), "recursive_functions": [i["function"] for i in issues], "issues": issues}

    # 2. V3 Call Graph Analysis
    call_graph = {}
    for cfg in cfgs:
        call_graph[cfg.get("function")] = cfg.get("function_calls", [])
        
    for func in functions:
        name = func["name"]
        if not name or name in ("main", "Main"): continue
        
        func_calls = call_graph.get(name, [])
        self_calls = func_calls.count(name)
        
        if self_calls > 0:
            call_line = func.get("line", 0) + 1 # Fallback approximation
            
            if self_calls > 1:
                severity = "MEDIUM"
                msg = f"Exponential Recursion: Function `{name}` evaluates itself {self_calls} times per frame! This scales O(C^N) and guarantees Stack Overflows on minimal inputs."
            else:
                severity = "LOW"
                msg = f"Linear Recursion: Function `{name}` evaluates itself recursively. Track depth manually to prevent Stack Overflow bounds."
                
            issues.append({
                "type": "RECURSION",
                "severity": severity,
                "function": name,
                "line": func.get("line", 0),
                "call_line": call_line,
                "message": msg,
                "suggestion": "Consider dynamic programming (memoization) or converting entirely to a while-loop structure."
            })
            
    return {"recursion_count": len(issues), "recursive_functions": [i["function"] for i in issues], "issues": issues}
