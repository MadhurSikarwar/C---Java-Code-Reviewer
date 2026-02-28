"""
IntelliReview — Code Quality Analyzer
Computes the 6 new signal features from raw source code:
  - comment_ratio          (comments / total lines)
  - magic_number_count     (hardcoded numeric literals)
  - exception_handling     (try/catch/throw blocks)
  - halstead_volume        (approximated from unique operators/operands)
  - duplicate_block_score  (repeated line-group patterns)
  - dead_code_estimate     (unused vars, unreachable returns)
"""
import re
import math
from typing import Dict, Any


# ── Operator / operand sets for Halstead approximation ─────────────────
C_OPERATORS = {
    '+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=',
    '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '++', '--',
    '+=', '-=', '*=', '/=', '%=', '->', '.', '?', ':',
}
JAVA_OPERATORS = C_OPERATORS | {'instanceof', 'new', 'throw', 'throws'}

# Numeric literal regex (not part of identifiers or version numbers)
_MAGIC_PATTERN = re.compile(
    r'(?<![.\w])\b([0-9]+(?:\.[0-9]+)?)\b(?![\w.])',
)
_MAGIC_WHITELIST = {0, 1, 2, -1}   # very common, not considered magic

# Try/catch/throw keywords
_EXCEPT_C    = re.compile(r'\b(setjmp|longjmp|perror|errno)\b')
_EXCEPT_JAVA = re.compile(r'\b(try|catch|finally|throw|throws)\b')

# Dead code patterns
_DEAD_AFTER_RETURN = re.compile(
    r'\breturn\b[^;]*;[ \t]*\n([ \t]*[^\s\n{}][^\n]*\n){1,3}',
    re.MULTILINE,
)
_UNUSED_VAR_C = re.compile(
    r'\b(?:int|char|float|double|long)\s+(\w+)\s*;',
)

# Comment patterns
_C_BLOCK_COMMENT   = re.compile(r'/\*.*?\*/', re.DOTALL)
_LINE_COMMENT      = re.compile(r'//[^\n]*')
_JAVA_ANNOTATION   = re.compile(r'@\w+')


def analyze_code_quality(source: str, language: str) -> Dict[str, Any]:
    """
    Compute 6 new code quality features from raw source code.
    Returns a dict with feature values.
    """
    lines = source.splitlines()
    total_lines = max(len(lines), 1)
    lang = language.upper()

    return {
        "comment_ratio":        _comment_ratio(source, lines, total_lines),
        "magic_number_count":   _magic_numbers(source),
        "exception_handling":   _exception_handling(source, lang),
        "halstead_volume":      _halstead_volume(source, lang),
        "duplicate_block_score":_duplicate_blocks(lines),
        "dead_code_estimate":   _dead_code(source, lines, lang),
    }


# ── Feature implementations ────────────────────────────────────────────

def _comment_ratio(source: str, lines: list, total: int) -> float:
    """Fraction of lines that are comments."""
    # Remove block comments, count line comments
    no_strings = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', source)
    block_comment_text = _C_BLOCK_COMMENT.findall(no_strings)
    block_lines = sum(c.count('\n') + 1 for c in block_comment_text)
    line_comments = len(_LINE_COMMENT.findall(no_strings))
    comment_lines = block_lines + line_comments
    ratio = round(min(comment_lines / total, 1.0), 3)
    return ratio


def _magic_numbers(source: str) -> int:
    """Count hardcoded magic numeric literals (excluding 0, 1, 2, -1)."""
    # Strip comments and strings first
    clean = _C_BLOCK_COMMENT.sub(' ', source)
    clean = _LINE_COMMENT.sub(' ', clean)
    clean = re.sub(r'"[^"]*"', '""', clean)
    clean = re.sub(r"'[^']*'", "''", clean)

    matches = _MAGIC_PATTERN.findall(clean)
    count = 0
    for m in matches:
        try:
            val = float(m)
            if val not in _MAGIC_WHITELIST and val not in {-1.0, 0.0, 1.0, 2.0}:
                count += 1
        except ValueError:
            pass
    return min(count, 50)  # cap at 50


def _exception_handling(source: str, lang: str) -> int:
    """Count try/catch/finally/throw occurrences."""
    if lang == "JAVA":
        return len(_EXCEPT_JAVA.findall(source))
    else:
        # C: use setjmp/longjmp/errno as proxy
        return len(_EXCEPT_C.findall(source))


def _halstead_volume(source: str, lang: str) -> int:
    """
    Approximate Halstead Volume = N * log2(n)
      N = total operators + operands
      n = unique operators + unique operands
    We tokenize the source crudely.
    """
    # Strip strings, comments
    clean = _C_BLOCK_COMMENT.sub(' ', source)
    clean = _LINE_COMMENT.sub(' ', clean)
    clean = re.sub(r'"[^"]*"', ' STR ', clean)

    ops_set = JAVA_OPERATORS if lang == "JAVA" else C_OPERATORS

    # Count operators
    operator_tokens = []
    for op in sorted(ops_set, key=len, reverse=True):
        escaped = re.escape(op)
        found = re.findall(escaped, clean)
        operator_tokens.extend(found)

    # Count operands (identifiers + numbers)
    operand_tokens = re.findall(r'\b[a-zA-Z_]\w*\b|\b\d+(?:\.\d+)?\b', clean)

    N = len(operator_tokens) + len(operand_tokens)
    n = len(set(operator_tokens)) + len(set(operand_tokens))

    if n <= 1:
        return 0
    volume = int(N * math.log2(n))
    return min(volume, 10000)  # cap outliers


def _duplicate_blocks(lines: list) -> int:
    """
    Score (0–10) how many repeated 3-line blocks exist.
    Higher = more copy-paste patterns.
    """
    if len(lines) < 6:
        return 0

    # Create normalized 3-line windows
    windows = []
    for i in range(len(lines) - 2):
        block = tuple(
            re.sub(r'\s+', ' ', lines[i+j].strip())
            for j in range(3)
            if lines[i+j].strip() and not lines[i+j].strip().startswith('//')
        )
        if len(block) == 3 and all(len(b) > 5 for b in block):
            windows.append(block)

    # Count duplicates
    seen = {}
    dupes = 0
    for w in windows:
        seen[w] = seen.get(w, 0) + 1
    dupes = sum(v - 1 for v in seen.values() if v > 1)

    # Normalize to 0–10
    return min(dupes, 10)


def _dead_code(source: str, lines: list, lang: str) -> int:
    """
    Estimate dead code: unreachable statements after return
    + declared-but-never-used variable names (C only heuristic).
    """
    count = 0

    # Unreachable code after return
    dead_blocks = _DEAD_AFTER_RETURN.findall(source)
    count += len(dead_blocks)

    # C: declared vars that are never used again
    if lang == "C":
        declared = _UNUSED_VAR_C.findall(source)
        for var in declared:
            # Check if var is used more than once (declaration + usage)
            uses = len(re.findall(rf'\b{re.escape(var)}\b', source))
            if uses <= 1:
                count += 1

    return min(count, 15)
