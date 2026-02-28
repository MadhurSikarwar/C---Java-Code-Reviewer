"""
IntelliReview — Unsafe Function Detector
Flags usage of dangerous C/Java functions that can cause buffer overflows,
command injection, or other security vulnerabilities.
"""
from typing import Dict, Any, List

UNSAFE_C_DETAILS = {
    "gets": {
        "severity": "CRITICAL",
        "reason": "gets() reads unlimited input — always causes buffer overflow.",
        "fix": "Use fgets(buffer, size, stdin) instead.",
    },
    "strcpy": {
        "severity": "HIGH",
        "reason": "strcpy() does not check destination buffer size.",
        "fix": "Use strncpy(dest, src, dest_size - 1) or strlcpy().",
    },
    "strcat": {
        "severity": "HIGH",
        "reason": "strcat() does not check buffer bounds.",
        "fix": "Use strncat(dest, src, dest_size - strlen(dest) - 1).",
    },
    "sprintf": {
        "severity": "HIGH",
        "reason": "sprintf() can overflow the destination buffer.",
        "fix": "Use snprintf(buf, sizeof(buf), ...) with explicit size.",
    },
    "vsprintf": {
        "severity": "HIGH",
        "reason": "vsprintf() can overflow the destination buffer.",
        "fix": "Use vsnprintf().",
    },
    "scanf": {
        "severity": "MEDIUM",
        "reason": "scanf() without width specifier can overflow input buffer.",
        "fix": "Use scanf(\"%255s\", buf) with explicit max width.",
    },
    "memcpy": {
        "severity": "MEDIUM",
        "reason": "memcpy() with unvalidated sizes risks out-of-bounds write.",
        "fix": "Always validate that size <= destination buffer size.",
    },
    "strncpy": {
        "severity": "LOW",
        "reason": "strncpy() may not null-terminate if source is too long.",
        "fix": "Always manually null-terminate: buf[size-1] = '\\0';",
    },
}

UNSAFE_JAVA_DETAILS = {
    "exec": {
        "severity": "HIGH",
        "reason": "Runtime.exec() with user input enables OS command injection.",
        "fix": "Use ProcessBuilder with a whitelist of allowed commands.",
    },
    "eval": {
        "severity": "HIGH",
        "reason": "Dynamic code evaluation can lead to code injection.",
        "fix": "Avoid eval(); use a safe parser or scripting API.",
    },
    "readLine": {
        "severity": "LOW",
        "reason": "Unvalidated user input from readLine() can cause injection attacks.",
        "fix": "Always validate and sanitize input from readLine().",
    },
}


def detect_unsafe_functions(parse_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect unsafe function usage from the parse result and enrich with
    detailed severity, reason, and fix suggestions.
    """
    language = parse_result.get("language", "C")
    raw_unsafe = parse_result.get("unsafe_calls", [])
    reference = UNSAFE_C_DETAILS if language == "C" else UNSAFE_JAVA_DETAILS

    issues = []
    for call in raw_unsafe:
        func = call.get("function", "")
        # Match exact name or starts with name (Java method calls with class prefix)
        matched_key = None
        for key in reference:
            if func == key or func.endswith(f".{key}") or key in func:
                matched_key = key
                break

        if matched_key:
            detail = reference[matched_key]
            issues.append({
                "type": "UNSAFE_FUNCTION",
                "severity": detail["severity"],
                "line": call.get("line", 0),
                "function": func,
                "message": f"Unsafe use of `{func}` at line {call.get('line', '?')}: {detail['reason']}",
                "suggestion": detail["fix"],
            })
        else:
            # Pass through with generic message
            issues.append({
                "type": "UNSAFE_FUNCTION",
                "severity": call.get("severity", "MEDIUM"),
                "line": call.get("line", 0),
                "function": func,
                "message": f"Potentially unsafe function `{func}` at line {call.get('line', '?')}.",
                "suggestion": "Review this function for buffer overflow or injection risks.",
            })

    return {
        "unsafe_function_count": len(issues),
        "issues": issues,
    }
