"""
IntelliReview — inline suppressions

Real code always has findings you have looked at and accepted. Tell the analyzer so, in a comment:

    strcpy(dst, src);          // intellireview: ignore
    // intellireview: ignore[USE_AFTER_FREE, DOUBLE_FREE]
    free(p); free(p);
    /* intellireview: ignore-file */          <- anywhere: silence the whole file

`ignore` applies to findings on the same line or on the line directly below the comment. With `[TYPE, ...]` only those finding
types are silenced. Suppressed findings are not hidden silently: the API reports them, and the UI says how many there were.
"""
import re
from typing import Any, Dict, List, Tuple

_DIRECTIVE = re.compile(r"intellireview\s*:\s*(ignore-file|ignore)(?:\s*\[([^\]]*)\])?", re.I)


def parse_directives(source: str) -> Dict[str, Any]:
    """{'file': bool, 'file_types': set|None, 'lines': {line: set|None}} — None means "every type"."""
    per_line: Dict[int, Any] = {}
    file_all, file_types = False, set()
    for i, text in enumerate(source.splitlines(), 1):
        m = _DIRECTIVE.search(text)
        if not m:
            continue
        types = {t.strip().upper() for t in (m.group(2) or "").split(",") if t.strip()} or None
        if m.group(1).lower() == "ignore-file":
            if types is None:
                file_all = True
            else:
                file_types |= types
            continue
        code_before = re.split(r"//|/\*|#", text, maxsplit=1)[0].strip()
        targets = [i] if code_before else [i, i + 1]          # trailing comment -> this line; own-line comment -> next line too
        for t in targets:
            cur = per_line.get(t, set())
            per_line[t] = None if (types is None or cur is None) else (cur | types)
    return {"file": file_all, "file_types": file_types, "lines": per_line}


def apply_suppressions(source: str, issues: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (kept, suppressed)."""
    if "intellireview" not in source.lower():
        return issues, []
    d = parse_directives(source)
    kept, dropped = [], []
    for it in issues:
        t = str(it.get("type", "")).upper()
        line = int(it.get("line") or 0)
        hit = d["file"] or t in d["file_types"]
        if not hit and line in d["lines"]:
            allowed = d["lines"][line]
            hit = allowed is None or t in allowed
        (dropped if hit else kept).append(it)
    return kept, dropped
