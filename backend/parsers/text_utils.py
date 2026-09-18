"""
IntelliReview — source text helpers.

`sanitize()` removes comments (and optionally the *contents* of string / char literals)
while preserving every newline, so line numbers computed on the result still match the
original source. Regex based rules should run on the sanitized text, otherwise
`// gets(buf)` or `"strcpy("` inside a string would be reported as real code.
"""
from typing import List, Tuple


def sanitize(source: str, keep_strings: bool = False, java_text_blocks: bool = False) -> str:
    """
    Return `source` with comments blanked out. If `keep_strings` is False the contents of
    string and character literals are blanked too (the quotes are kept).
    Newlines are always preserved.
    """
    out: List[str] = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        # line comment
        if c == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # block comment
        if c == "/" and nxt == "*":
            i += 2
            while i < n and not (source[i] == "*" and i + 1 < n and source[i + 1] == "/"):
                if source[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            out.append(" ")
            continue

        # Java text block  """ ... """
        if java_text_blocks and c == '"' and source.startswith('"""', i):
            end = source.find('"""', i + 3)
            end = n if end == -1 else end + 3
            body = source[i:end]
            if keep_strings:
                out.append(body)
            else:
                out.append('"""' + "".join("\n" if ch == "\n" else " " for ch in body[3:-3]) + '"""')
            i = end
            continue

        # string / char literal
        if c in ('"', "'"):
            quote = c
            j = i + 1
            while j < n and source[j] != quote and source[j] != "\n":
                if source[j] == "\\" and j + 1 < n:
                    j += 1
                j += 1
            end = min(j + 1, n)
            lit = source[i:end]
            if keep_strings:
                out.append(lit)
            else:
                inner = lit[1:-1] if lit.endswith(quote) and len(lit) >= 2 else lit[1:]
                out.append(quote + " " * len(inner) + (quote if lit.endswith(quote) and len(lit) >= 2 else ""))
            i = end
            continue

        out.append(c)
        i += 1
    return "".join(out)


def braces_balanced(clean_source: str) -> Tuple[bool, str]:
    """Check (), [] and {} balance on comment/string-free text. Returns (ok, reason)."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[Tuple[str, int]] = []
    line = 1
    for ch in clean_source:
        if ch == "\n":
            line += 1
        elif ch in "([{":
            stack.append((ch, line))
        elif ch in pairs:
            if not stack or stack[-1][0] != pairs[ch]:
                return False, f"Unexpected '{ch}' at line {line}"
            stack.pop()
    if stack:
        ch, ln = stack[-1]
        return False, f"Unclosed '{ch}' opened at line {ln}"
    return True, ""


def looks_like_code(clean_source: str) -> bool:
    """Cheap sanity check: real C/C++/Java has structure characters, prose does not."""
    s = clean_source.strip()
    if not s:
        return False
    structural = sum(s.count(ch) for ch in ";{}()")
    return structural >= 3 and (";" in s or "{" in s)
