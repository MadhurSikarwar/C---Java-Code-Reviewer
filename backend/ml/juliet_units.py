"""
IntelliReview - split a (preprocessed, comment-free) C or Java translation unit into top-level ITEMS and build a
self-contained sample for one labelled function.

Why: a Juliet flow variant is not one function.  `bad()` may call a sink half `badSink(data)` in the same file, read a
file-level `static char *badData`, or branch on `staticReturnsTrue()`.  Cutting only the text of `bad()` makes the flaw
invisible for *any* analyser.  So a sample is: the labelled function + everything in the file it (transitively) uses.
Every name that is not the labelled function is renamed to `fn_<i>_<k>` so that names like `badSink` cannot leak the label
and so that several samples can be glued into one big file (composites) without collisions.

For C the helpers are made `static` (a closed world: their callers are exactly the ones in the sample).
"""
import re
from typing import Dict, List, Optional, Set

_KW = {"if", "for", "while", "switch", "return", "sizeof", "else", "do", "catch", "new", "throws", "synchronized", "case"}


def _matching_paren_start(head: str) -> Optional[int]:
    """Index of the '(' that matches the last ')' in `head` (function parameter list), or None."""
    j = head.rfind(")")
    if j == -1:
        return None
    depth = 0
    for i in range(j, -1, -1):
        if head[i] == ")":
            depth += 1
        elif head[i] == "(":
            depth -= 1
            if depth == 0:
                return i
    return None


def split_items(text: str, language: str) -> List[dict]:
    """Top-level items of `text` (for Java pass the class BODY).  Each item:
       {text, kind: func|decl|proto|type|class, name, params}"""
    items: List[dict] = []
    n = len(text)
    i = 0
    start: Optional[int] = None
    depth = paren = 0
    first_brace = -1
    while i < n:
        c = text[i]
        if start is None:
            if c.isspace() or c == ";":
                i += 1
                continue
            start = i
            first_brace = -1
        if c == "(":
            paren += 1
        elif c == ")":
            paren -= 1
        elif c == "{":
            if depth == 0:
                first_brace = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and paren == 0:
                head = text[start:first_brace]
                head_np = re.sub(r"\([^()]*\)", "", head)
                if "=" not in head_np and re.search(r"\)\s*(?:throws[\w\s.,]+)?\s*$", head):
                    items.append(_func_item(text[start:i + 1], head))
                    start = None
                elif language != "C" and re.search(r"\b(?:class|interface|enum)\b", head):
                    items.append({"text": text[start:i + 1], "kind": "class", "name": None, "params": ""})
                    start = None
                # else: struct/enum body or initializer list -> ends at the following ';'
        elif c == ";" and depth == 0 and paren == 0:
            items.append(_decl_item(text[start:i + 1].strip(), language))
            start = None
        i += 1
    return items


def _func_item(full: str, head: str) -> dict:
    p = _matching_paren_start(head)
    name = None
    params = ""
    if p is not None:
        m = re.search(r"(\w+)\s*$", head[:p])
        name = m.group(1) if m else None
        params = head[p + 1:head.rfind(")")].strip()
    return {"text": full.strip(), "kind": "func", "name": name, "params": params}


def _decl_item(t: str, language: str) -> dict:
    if re.match(r"\s*(?:typedef|struct|union|enum)\b", t) and language == "C":
        return {"text": t, "kind": "type", "name": None, "params": ""}
    before_init = t.split("=", 1)[0]
    if "(" in re.sub(r"\([^()]*\)\s*\(", "", before_init) and "=" not in t.split("(", 1)[0]:
        m = re.search(r"(\w+)\s*\(", before_init)
        return {"text": t, "kind": "proto", "name": m.group(1) if m else None, "params": ""}
    m = re.search(r"(\w+)\s*(?:\[[^\]]*\]\s*)*$", before_init.rstrip("; \t\n"))
    return {"text": t, "kind": "decl", "name": m.group(1) if m else None, "params": ""}


def class_body(code: str) -> Optional[tuple]:
    """(prefix, body, suffix) of the first top-level Java class; body excludes the outer braces."""
    m = re.search(r"\bclass\s+\w+[^{]*\{", code)
    if not m:
        return None
    open_idx = m.end() - 1
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return code[:open_idx + 1], code[open_idx + 1:i], code[i:]
    return None


def _idents(text: str) -> Set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", text))


def build_sample(items: List[dict], target: dict, idx: int, language: str) -> Optional[dict]:
    """Labelled function `target` + everything of `items` it uses, renamed.  Returns {body, decls} or None."""
    defined: Dict[str, dict] = {}
    for it in items:
        if it["name"] and it["kind"] in ("func", "decl"):
            defined.setdefault(it["name"], it)          # first definition wins (a prototype never shadows it)
    keep: List[dict] = [target]
    seen: Set[int] = {id(target)}
    work = [target]
    while work:
        cur = work.pop()
        for w in _idents(cur["text"]):
            it = defined.get(w)
            if it is not None and id(it) not in seen and w != target["name"]:
                seen.add(id(it))
                keep.append(it)
                work.append(it)
    order = {id(it): k for k, it in enumerate(items)}
    keep.sort(key=lambda it: order[id(it)])
    names = [it["name"] for it in keep if it is not target and it["name"]]
    rename = {target["name"]: f"fn_{idx}"}
    for k, nm in enumerate(dict.fromkeys(names), 1):
        rename[nm] = f"fn_{idx}_{k}"
    rx = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in sorted(rename, key=len, reverse=True)) + r")\b")

    def sub(t: str) -> str:
        return rx.sub(lambda m: rename[m.group(0)], t)

    parts: List[str] = []
    for it in keep:
        t = sub(it["text"])
        if language == "C" and it["kind"] == "func" and it is not target and not re.match(r"\s*static\b", t):
            t = "static " + t
        parts.append(t)
    # protos of kept functions are not needed once the definition is present; types are shared decls
    types = []
    for it in items:
        if it["kind"] == "type":
            types.append(it["text"])
    return {"body": "\n\n".join(parts), "types": types}
