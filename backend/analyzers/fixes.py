"""
IntelliReview — concrete one-line fixes

For the findings that have a mechanical, *safe* repair, attach `issue["fix"] = {title, line, before, after, note}` so the UI can
show the change and apply it with one click. Nothing is guessed: if a fix would only be correct under an assumption we cannot
verify (e.g. `sizeof(buf)` when `buf` is a pointer), no edit is offered, only the written advice.

  C:    printf(x)            -> printf("%s", x)              fprintf(f, x) -> fprintf(f, "%s", x)
        gets(buf)            -> fgets(buf, sizeof(buf), stdin)     (buf declared as an array in this file)
        strcpy(d, s)         -> snprintf(d, sizeof(d), "%s", s)    (d declared as an array)
        strcat(d, s)         -> strncat(d, s, sizeof(d) - strlen(d) - 1)
        sprintf(d, ...)      -> snprintf(d, sizeof(d), ...)
  Java: getInstance("MD5"/"SHA-1"/...) -> getInstance("SHA-256")
        catch (E e) {}       -> catch (E e) { e.printStackTrace(); }
        NAME = "literal";    -> NAME = System.getenv("NAME");      (credential-looking names only)
"""
import re
from typing import Any, Dict, List, Optional

_ARR = lambda src, name: re.search(rf"\b(?:char|wchar_t|unsigned\s+char)\s+{re.escape(name)}\s*\[", src) is not None  # noqa: E731
_IDENT = r"[A-Za-z_]\w*(?:\s*(?:\[[^\]]*\]|\.\w+|->\w+))*"
_CRED = re.compile(r"(?i)pass(?:word|wd)?|pwd|secret|api_?key|token|credential|private_?key")


def _fix_c(src: str, text: str, itype: str) -> Optional[Dict[str, str]]:
    if itype == "FORMAT_STRING":
        m = re.search(rf"\b(printf|vprintf)\s*\(\s*({_IDENT})\s*\)", text)
        if m:
            return {"title": "Print the value through a constant format string",
                    "after": text[:m.start()] + f'{m.group(1)}("%s", {m.group(2)})' + text[m.end():]}
        m = re.search(rf"\b(fprintf|dprintf|syslog)\s*\(\s*([\w.]+|LOG_\w+)\s*,\s*({_IDENT})\s*\)", text)
        if m:
            return {"title": "Print the value through a constant format string",
                    "after": text[:m.start()] + f'{m.group(1)}({m.group(2)}, "%s", {m.group(3)})' + text[m.end():]}
    if itype == "UNSAFE_FUNCTION":
        m = re.search(r"\bgets\s*\(\s*(\w+)\s*\)", text)
        if m and _ARR(src, m.group(1)):
            b = m.group(1)
            return {"title": "Use fgets with the buffer's size",
                    "after": text[:m.start()] + f"fgets({b}, sizeof({b}), stdin)" + text[m.end():]}
        m = re.search(r"\bstrcpy\s*\(\s*(\w+)\s*,\s*([^;]+?)\s*\)\s*;", text)
        if m and _ARR(src, m.group(1)):
            d = m.group(1)
            return {"title": "Copy with a bound",
                    "after": text[:m.start()] + f'snprintf({d}, sizeof({d}), "%s", {m.group(2)});' + text[m.end():]}
        m = re.search(r"\bstrcat\s*\(\s*(\w+)\s*,\s*([^;]+?)\s*\)\s*;", text)
        if m and _ARR(src, m.group(1)):
            d = m.group(1)
            return {"title": "Append with a bound",
                    "after": text[:m.start()] + f"strncat({d}, {m.group(2)}, sizeof({d}) - strlen({d}) - 1);" + text[m.end():]}
        m = re.search(r"\bsprintf\s*\(\s*(\w+)\s*,", text)
        if m and _ARR(src, m.group(1)):
            d = m.group(1)
            return {"title": "Format with a bound",
                    "after": text[:m.start()] + f"snprintf({d}, sizeof({d})," + text[m.end():]}
    return None


def _fix_java(src: str, text: str, itype: str) -> Optional[Dict[str, str]]:
    if itype == "WEAK_CRYPTO":
        m = re.search(r'(getInstance\s*\(\s*)"(MD2|MD4|MD5|SHA-?1)"', text, re.I)
        if m:
            return {"title": "Use SHA-256", "after": text[:m.start()] + m.group(1) + '"SHA-256"' + text[m.end():],
                    "note": "SHA-256 is fine for integrity checks. For storing passwords use a slow KDF such as PBKDF2 or bcrypt."}
    if itype == "EMPTY_CATCH":
        m = re.search(r"catch\s*\(\s*([\w.| ]+?)\s+(\w+)\s*\)\s*\{\s*\}", text)
        if m:
            return {"title": "Don't swallow the exception",
                    "after": text[:m.start()] + f"catch ({m.group(1)} {m.group(2)}) {{ {m.group(2)}.printStackTrace(); }}" + text[m.end():]}
    if itype == "HARDCODED_CREDENTIAL":
        m = re.search(r'\b(\w+)\s*=\s*"[^"\n]+"\s*;', text)
        if m and _CRED.search(m.group(1)):
            env = re.sub(r"[^A-Za-z0-9]+", "_", m.group(1)).upper()
            return {"title": "Read it from the environment",
                    "after": text[:m.start()] + f'{m.group(1)} = System.getenv("{env}");' + text[m.end():],
                    "note": f"Set the {env} environment variable (or use your secrets manager)."}
    return None


def attach_fixes(source: str, language: str, issues: List[Dict[str, Any]]) -> None:
    """Mutates `issues`, adding a `fix` dict where a safe mechanical edit exists."""
    lines = source.splitlines()
    for it in issues:
        ln = int(it.get("line") or 0)
        if ln < 1 or ln > len(lines):
            continue
        text = lines[ln - 1]
        fix = (_fix_c if language.upper() == "C" else _fix_java)(source, text, str(it.get("type", "")))
        if fix and fix["after"] != text:
            it["fix"] = {"title": fix["title"], "line": ln, "before": text, "after": fix["after"], "note": fix.get("note", "")}
