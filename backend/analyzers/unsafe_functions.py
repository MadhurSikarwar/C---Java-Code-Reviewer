"""
IntelliReview — Unsafe Function Detector
Flags usage of dangerous C functions that can cause buffer overflows, format-string bugs,
or command injection. (Java findings come from analyzers/java_security.py, which needs
argument/taint analysis rather than a list of method names.)
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
    "memmove": {
        "severity": "MEDIUM",
        "reason": "memmove() with unvalidated sizes risks out-of-bounds write.",
        "fix": "Always validate that size <= destination buffer size.",
    },
    "strncpy": {
        "severity": "LOW",
        "reason": "strncpy() may not null-terminate if source is too long.",
        "fix": "Always manually null-terminate: buf[size-1] = '\\0';",
    },
    "buffer_overflow": {
        "severity": "HIGH",
        "reason": "a memory access was proven to be out of bounds.",
        "fix": "Bound every index and copy length by the destination buffer size.",
    },
    "uninit_read": {
        "severity": "MEDIUM",
        "reason": "memory is read before anything was written to it.",
        "fix": "Initialise the buffer (calloc, memset, or an initialiser) before reading it.",
    },
    "format_string": {
        "severity": "HIGH",
        "reason": "the format string is not a literal — attacker-controlled specifiers (%s, %n) allow memory "
                  "reads/writes (CWE-134).",
        "fix": "Use a constant format string: printf(\"%s\", value) instead of printf(value).",
    },
    "system": {
        "severity": "HIGH",
        "reason": "system() with a non-constant command allows OS command injection.",
        "fix": "Avoid system(); use execve() with a fixed argument vector and validate all inputs.",
    },
    "popen": {
        "severity": "HIGH",
        "reason": "popen() with a non-constant command allows OS command injection.",
        "fix": "Avoid popen() with user data; use fork/execve with a fixed argument vector.",
    },
    "execl": {"severity": "HIGH", "reason": "exec*() with a non-constant path/argument allows command injection.",
              "fix": "Validate the program path against a whitelist."},
    "execlp": {"severity": "HIGH", "reason": "exec*() with a non-constant path/argument allows command injection.",
               "fix": "Validate the program path against a whitelist."},
    "execv": {"severity": "HIGH", "reason": "exec*() with a non-constant path/argument allows command injection.",
              "fix": "Validate the program path against a whitelist."},
    "execvp": {"severity": "HIGH", "reason": "exec*() with a non-constant path/argument allows command injection.",
               "fix": "Validate the program path against a whitelist."},
    "tmpnam": {
        "severity": "MEDIUM",
        "reason": "tmpnam() returns a predictable name (race condition, CWE-377).",
        "fix": "Use mkstemp().",
    },
    "mktemp": {
        "severity": "MEDIUM",
        "reason": "mktemp() is vulnerable to race conditions (CWE-377).",
        "fix": "Use mkstemp().",
    },
    "alloca": {
        "severity": "MEDIUM",
        "reason": "alloca() has no failure indication and can overflow the stack.",
        "fix": "Use a fixed-size buffer or malloc().",
    },
    "getwd": {
        "severity": "HIGH",
        "reason": "getwd() cannot bound the size of the output buffer.",
        "fix": "Use getcwd(buf, size).",
    },
}


def detect_unsafe_functions(parse_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect unsafe function usage from the parse result and enrich with
    detailed severity, reason, and fix suggestions.
    """
    raw_unsafe = parse_result.get("unsafe_calls", [])

    issues: List[Dict[str, Any]] = []
    for call in raw_unsafe:
        func = call.get("function", "")
        detail = UNSAFE_C_DETAILS.get(func)
        if detail:
            shown = call.get("callee") or func
            if func == "uninit_read":
                issues.append({
                    "type": "UNINITIALIZED_VARIABLE",
                    "severity": call.get("severity", detail["severity"]),
                    "line": call.get("line", 0),
                    "function": func,
                    "message": f"Uninitialized memory at line {call.get('line', '?')}: {call.get('reason', detail['reason'])}",
                    "suggestion": detail["fix"],
                })
                continue
            if func == "buffer_overflow":
                # the bounds checker already produced a precise severity and explanation
                issues.append({
                    "type": "BUFFER_OVERFLOW",
                    "severity": call.get("severity", detail["severity"]),
                    "line": call.get("line", 0),
                    "function": func,
                    "message": f"Buffer overflow at line {call.get('line', '?')}: {call.get('reason', detail['reason'])}",
                    "suggestion": detail["fix"],
                })
                continue
            issues.append({
                "type": "FORMAT_STRING" if func == "format_string" else "UNSAFE_FUNCTION",
                # the parser may already have judged the call more precisely (e.g. a string copy that provably
                # overflows its destination is CRITICAL, a printf wrapper is LOW); the parser's judgement wins
                "severity": call.get("severity") if call.get("severity") in ("CRITICAL", "MEDIUM", "LOW") else detail["severity"],
                "line": call.get("line", 0),
                "function": func,
                "message": f"Unsafe use of `{shown}` at line {call.get('line', '?')}: {detail['reason']}",
                "suggestion": detail["fix"],
            })
        else:
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
