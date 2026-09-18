"""
IntelliReview — C Source Code Parser
Uses pycparser to build an AST from C source code.
Falls back to regex-based analysis (comments / strings stripped first) when pycparser
cannot parse the input, e.g. C++ or code that depends on unavailable macros/headers.
"""
import re
import sys
from typing import Dict, Any, List

from parsers.text_utils import sanitize
from parsers.c_bounds import FuncBounds
from parsers.c_taint import FuncTaint

# Functions whose first/ith argument is a printf-style format string.
# name -> index of the format argument
FORMAT_FUNCS = {
    "printf": 0, "vprintf": 0, "wprintf": 0,
    "fprintf": 1, "vfprintf": 1, "sprintf": 1, "vsprintf": 1, "dprintf": 1, "syslog": 1,
    "snprintf": 2, "vsnprintf": 2,
}

# Calls that are dangerous regardless of arguments.
ALWAYS_UNSAFE_C = {
    "gets": "CRITICAL", "strcpy": "HIGH", "strcat": "HIGH", "sprintf": "HIGH", "vsprintf": "HIGH",
    "scanf": "MEDIUM", "memcpy": "MEDIUM", "memmove": "MEDIUM", "strncpy": "LOW",
    "tmpnam": "MEDIUM", "mktemp": "MEDIUM", "alloca": "MEDIUM", "getwd": "HIGH",
}
# Dangerous only when the (first) argument is not a string literal.
UNSAFE_IF_DYNAMIC = {"system": "HIGH", "popen": "HIGH", "execl": "HIGH", "execlp": "HIGH",
                     "execv": "HIGH", "execvp": "HIGH"}

SYSTEM_ALIASES = {"SYSTEM": "system", "POPEN": "popen", "EXECL": "execl", "EXECLP": "execlp", "EXECV": "execv",
                  "EXECVP": "execvp"}

C_PREAMBLE = (
    "typedef unsigned int size_t; typedef unsigned char uint8_t; typedef unsigned short uint16_t; "
    "typedef unsigned int uint32_t; typedef unsigned long long uint64_t; typedef int int32_t; "
    "typedef long long int64_t; typedef void FILE; typedef int wchar_t; typedef long ssize_t; "
    "typedef int bool; "
)

CPP_MARKERS = re.compile(
    r"#\s*include\s*<(?:iostream|vector|string|map|set|memory|algorithm|fstream|sstream|cstdio|cstdlib|cstring)>"
    r"|\bstd::|\btemplate\s*<|\bnamespace\s+\w+|\bclass\s+\w+|\bnew\s+[\w:<>]+\s*[\(\[;]|\bdelete\b|\bnullptr\b"
    r"|\b(?:public|private|protected)\s*:"
)


def is_cpp_like(clean_source: str) -> bool:
    return bool(CPP_MARKERS.search(clean_source))


def parse_c_code(source: str) -> Dict[str, Any]:
    """
    Parse C source code and extract structural features.
    Returns a dict with counts and metadata.
    """
    result = {
        "language": "C",
        "functions": [],
        "num_functions": 0,
        "num_loops": 0,
        "max_nesting_depth": 0,
        "num_conditionals": 0,
        "num_pointer_uses": 0,
        "num_mallocs": 0,
        "num_frees": 0,
        "function_calls": [],
        "function_call_count": 0,
        "lines_of_code": 0,
        "malloc_lines": [],
        "free_lines": [],
        "unsafe_calls": [],
        "parse_errors": [],
        "analysis_mode": "ast",
        "cpp_like": False,
    }

    lines = source.splitlines()
    code_only = sanitize(source)
    result["lines_of_code"] = sum(1 for l in code_only.splitlines() if l.strip())
    result["cpp_like"] = is_cpp_like(code_only)

    # C++ can never be parsed by pycparser; skip straight to the heuristic parser.
    if not result["cpp_like"]:
        try:
            result.update(_parse_with_pycparser(source, lines))
            return result
        except RecursionError:
            result["parse_errors"].append("pycparser: nesting too deep for the AST parser; used heuristic analysis")
        except Exception as e:  # noqa: BLE001 - pycparser raises its own ParseError types
            result["parse_errors"].append(f"pycparser: {str(e)[:200]}")

    result["analysis_mode"] = "heuristic"
    result.update(_parse_with_regex(source, lines))
    return result


def _parse_with_pycparser(source: str, lines) -> Dict[str, Any]:
    """Use pycparser for accurate AST-based analysis."""
    import pycparser

    # pycparser has no preprocessor here: blank comments (keeping newlines) and drop directives
    # while keeping line numbers aligned with the original source.
    clean_source = sanitize(source, keep_strings=True)
    clean_source = re.sub(r"^[ \t]*#.*$", "", clean_source, flags=re.MULTILINE)
    body_source = clean_source

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 4000))
    try:
        # pycparser needs every type name to be a known typedef. Real code refers to types declared in headers we
        # cannot see (pthread_t, HANDLE, custom typedefs...), so on "before: <name>" we learn the unknown type that
        # precedes <name> and retry. The typedef preamble stays on the SAME line as the first source line so that
        # reported line numbers match the original source.
        learned: List[str] = []
        for _attempt in range(16):
            text = C_PREAMBLE + "".join(f"typedef int {t}; " for t in learned) + body_source
            try:
                ast = pycparser.CParser().parse(text, filename="<code>")
                break
            except Exception as e:  # pycparser.plyparser.ParseError
                t = _unknown_type_from_error(str(e), text, learned)
                if t is not None:
                    learned.append(t)
                    continue
                # no usable hint (e.g. "Invalid declaration" for an unknown type in a parameter list): learn every
                # identifier that sits where a type name would be, once
                guesses = [g for g in _type_name_candidates(body_source) if g not in learned]
                if not guesses:
                    raise
                learned.extend(guesses[:30])
        else:
            raise ValueError("too many unknown type names")
        visitor = _CASTVisitor(lines)
        visitor.visit(ast)
    finally:
        sys.setrecursionlimit(old_limit)

    return {
        "ast": ast,
        "learned_types": list(learned),
        "functions": visitor.functions,
        "num_functions": len(visitor.functions),
        "num_loops": visitor.num_loops,
        "max_nesting_depth": visitor.max_nesting_depth,
        "num_conditionals": visitor.num_conditionals,
        "num_pointer_uses": visitor.num_pointer_uses,
        "num_mallocs": visitor.num_mallocs,
        "num_frees": visitor.num_frees,
        "function_calls": list(visitor.function_calls),
        "function_call_count": len(visitor.function_calls),
        "malloc_lines": visitor.malloc_lines,
        "free_lines": visitor.free_lines,
        "unsafe_calls": visitor.unsafe_calls,
    }


_C_KEYWORDS = {
    "int", "char", "short", "long", "float", "double", "void", "signed", "unsigned", "struct", "union", "enum",
    "const", "volatile", "static", "extern", "register", "auto", "return", "if", "else", "while", "for", "do",
    "switch", "case", "default", "break", "continue", "goto", "sizeof", "typedef", "inline", "restrict",
}


def _unknown_type_from_error(msg: str, text: str, learned: List[str]):
    """'<code>:3:9: before: foo' -> the identifier standing in front of `foo` on that line (an unknown type name)."""
    m = re.match(r"<code>:(\d+):\d+: before: (\S+)", msg)
    if not m:
        return None
    lineno, tok = int(m.group(1)), m.group(2)
    lines = text.split("\n")
    if lineno < 1 or lineno > len(lines):
        return None
    for cand in re.finditer(r"\b([A-Za-z_]\w*)\b[\s\*\(]*\b" + re.escape(tok) + r"\b", lines[lineno - 1]):
        name = cand.group(1)
        if name not in _C_KEYWORDS and name not in learned:
            return name
    return None


def _format_wrapper_params(funcdef) -> set:
    """Names of the pointer parameters of a variadic / va_list-taking function (candidates for a format string)."""
    from pycparser import c_ast
    try:
        params = funcdef.decl.type.args.params if funcdef.decl.type.args else []
    except AttributeError:
        return set()
    variadic = any(isinstance(p, c_ast.EllipsisParam) for p in params)
    has_valist = any(isinstance(p, c_ast.Decl) and isinstance(p.type, c_ast.TypeDecl)
                     and isinstance(p.type.type, c_ast.IdentifierType) and "va_list" in p.type.type.names
                     for p in params)
    if not (variadic or has_valist):
        return set()
    return {p.name for p in params if isinstance(p, c_ast.Decl) and p.name and isinstance(p.type, c_ast.PtrDecl)}


def _type_name_candidates(code: str) -> List[str]:
    """Identifiers used as `<type> <name>` in declarations, e.g. `HANDLE h`, `pthread_t *t`, `my_t x = 1`."""
    out: List[str] = []
    for m in re.finditer(r"(?<![\w.>])(?<!struct )(?<!union )(?<!enum )([A-Za-z_]\w*)[ \t]+\**[ \t]*[A-Za-z_]\w*[ \t]*[,;)=\[]", code):
        name = m.group(1)
        if name not in _C_KEYWORDS and name not in out:
            out.append(name)
    return out


def _is_string_literal(node) -> bool:
    from pycparser import c_ast
    return isinstance(node, c_ast.Constant) and node.type == "string"


def _literal_len(node) -> int:
    """Length of a C string literal constant node in bytes (best effort, escapes count as 1)."""
    raw = node.value
    body = raw[raw.index('"') + 1: raw.rindex('"')]
    body = re.sub(r"\\(x[0-9a-fA-F]{1,2}|[0-7]{1,3}|.)", "X", body)
    return len(body)


class _CASTVisitor(object):
    """pycparser AST visitor to extract features."""

    def __init__(self, lines):
        self.lines = lines
        self.functions = []
        self.num_loops = 0
        self.max_nesting_depth = 0
        self.num_conditionals = 0
        self.num_pointer_uses = 0
        self.num_mallocs = 0
        self.num_frees = 0
        self.function_calls = set()
        self.malloc_lines = []
        self.free_lines = []
        self.unsafe_calls = []
        self._loop_depth = 0
        self._fb = None                          # FuncBounds of the function currently being visited
        self._ft = None                          # FuncTaint of the function currently being visited
        self._fmt_wrapper_params = set()         # params of a printf-style wrapper (variadic / takes a va_list)

    def visit(self, node):
        from pycparser import c_ast
        if node is None:
            return

        if isinstance(node, c_ast.FuncDef):
            func_name = node.decl.name if node.decl else "unknown"
            coord = node.coord.line if node.coord else 0
            self.functions.append({"name": func_name, "line": coord})
            saved, saved_t, saved_w = self._fb, self._ft, self._fmt_wrapper_params
            self._fmt_wrapper_params = _format_wrapper_params(node)
            self._fb = FuncBounds(node)
            self._fb.run()                       # provable buffer overflows + proofs that a copy is safe
            self.unsafe_calls.extend(self._fb.issues)
            self._ft = FuncTaint(node)
            self._ft.run()                       # which expressions can carry attacker-influenced data
            self._visit_children(node)
            self._fb, self._ft, self._fmt_wrapper_params = saved, saved_t, saved_w

        elif isinstance(node, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            self.num_loops += 1
            self._loop_depth += 1
            self.max_nesting_depth = max(self.max_nesting_depth, self._loop_depth)
            self._visit_children(node)
            self._loop_depth -= 1

        elif isinstance(node, (c_ast.If, c_ast.Switch)):
            self.num_conditionals += 1
            self._visit_children(node)

        elif isinstance(node, c_ast.FuncCall):
            self._visit_call(node)
            self._visit_children(node)

        elif isinstance(node, c_ast.PtrDecl):
            self.num_pointer_uses += 1
            self._visit_children(node)

        else:
            self._visit_children(node)

    def _add_unsafe(self, name, line, severity, reason=None, callee=None):
        self.unsafe_calls.append({
            "function": name, "line": line, "severity": severity,
            **({"reason": reason} if reason else {}),
            **({"callee": callee} if callee else {}),
        })

    def _visit_call(self, node):
        from pycparser import c_ast
        if not isinstance(node.name, c_ast.ID):
            return
        call_name = SYSTEM_ALIASES.get(node.name.name, node.name.name)   # Juliet-style SYSTEM(...) wrappers
        self.function_calls.add(call_name)
        line = node.coord.line if node.coord else 0
        args = node.args.exprs if node.args else []

        if call_name in ("malloc", "calloc", "realloc"):
            self.num_mallocs += 1
            self.malloc_lines.append(line)
        elif call_name == "free":
            self.num_frees += 1
            self.free_lines.append(line)

        # printf(user_string) — format string vulnerability
        if call_name in FORMAT_FUNCS:
            idx = FORMAT_FUNCS[call_name]
            if len(args) > idx and not _is_string_literal(args[idx]):
                fmt = args[idx]
                is_wrapper = isinstance(fmt, __import__("pycparser").c_ast.ID) and fmt.name in self._fmt_wrapper_params
                # `void logf(const char *fmt, ...) { vprintf(fmt, ap); }` is the standard wrapper idiom, not a bug
                self._add_unsafe("format_string", line, "LOW" if is_wrapper else self._sev_by_taint([fmt]),
                                 callee=call_name)

        if call_name in ALWAYS_UNSAFE_C:
            sev = ALWAYS_UNSAFE_C[call_name]
            if sev == "HIGH" and call_name in ("strcpy", "strcat", "sprintf", "vsprintf"):
                sev = self._sev_by_taint(args[1:])       # HIGH only if attacker-influenced data is copied
            verdict = self._fb.verdicts.get(id(node)) if self._fb else None
            if verdict == "SKIP":
                sev = None                       # proven to fit (or already reported as a definite overflow)
            elif verdict == "MEDIUM":
                sev = "MEDIUM"                   # sizes fit individually, but existing content may not
            elif call_name == "sprintf" and len(args) >= 2 and _is_string_literal(args[1]) \
                    and "%" not in args[1].value and self._fb:
                dc = self._fb._elems_of(args[0])
                if dc is not None and _literal_len(args[1]) + 1 <= dc:
                    sev = None                   # sprintf of a %-free literal that fits
            if sev:
                self._add_unsafe(call_name, line, sev)

        elif call_name in UNSAFE_IF_DYNAMIC and args and not _is_string_literal(args[0]):
            self._add_unsafe(call_name, line, self._sev_by_taint(args[:1]))

    def _sev_by_taint(self, exprs) -> str:
        """HIGH if any of `exprs` can carry attacker-influenced data, MEDIUM when its provenance is unknown/local."""
        if self._ft is not None and any(self._ft.expr_tainted(e) for e in exprs):
            return "HIGH"
        return "MEDIUM"

    def _visit_children(self, node):
        for _, child in node.children():
            self.visit(child)


def _parse_with_regex(source: str, lines) -> Dict[str, Any]:
    """Heuristic parser used for C++ and for C that pycparser cannot handle."""
    code = sanitize(source)                 # comments + literals blanked, newlines kept
    clines = code.splitlines()

    keywords = {"if", "for", "while", "switch", "return", "sizeof", "else", "do", "catch"}
    functions: List[Dict[str, Any]] = []
    func_re = re.compile(r"^[ \t]*(?:[\w:<>~\*&]+[ \t\*&]+)+(?:[\w:]+::)?(\w+)[ \t]*\([^;{}]*\)[ \t\w]*\{?[ \t]*$")
    for i, line in enumerate(clines, 1):
        m = func_re.match(line)
        if m and m.group(1) not in keywords and not line.lstrip().startswith(("return", "else", "typedef")):
            # require an opening brace on this or the next non-empty line
            nxt = next((l for l in clines[i:i + 3] if l.strip()), "")
            if "{" in line or nxt.strip().startswith("{"):
                functions.append({"name": m.group(1), "line": i})
    if not functions:  # single-line functions: int f(int a){...}
        for m in re.finditer(r"(?:^|[;}\n])\s*(?:[\w\*&]+\s+)+(\w+)\s*\([^;{}]*\)\s*\{", code):
            if m.group(1) not in keywords:
                functions.append({"name": m.group(1), "line": code.count("\n", 0, m.start(1)) + 1})

    num_loops = len(re.findall(r"\b(?:for|while|do)\b\s*[\(\{]", code))
    depth = max_depth = 0
    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    num_conditionals = len(re.findall(r"\b(?:if|switch)\b", code))
    num_pointer_uses = len(re.findall(r"[\w\)\]]\s*\*+\s*\w|->|(?<![\w\)\]\s])\s*\*\s*\w", code))

    malloc_lines, free_lines = [], []
    for i, l in enumerate(clines, 1):
        if re.search(r"\b(?:malloc|calloc|realloc)\s*\(|\bnew\b\s*[\w:<>]", l):
            malloc_lines.append(i)
        if re.search(r"\bfree\s*\(|\bdelete\b", l):
            free_lines.append(i)

    unsafe_calls = []
    for i, l in enumerate(clines, 1):
        for f, sev in ALWAYS_UNSAFE_C.items():
            if sev in ("HIGH", "CRITICAL") and re.search(rf"\b{f}\s*\(", l):
                # without an AST we cannot tell where the data comes from: only gets() is unconditionally critical
                unsafe_calls.append({"function": f, "line": i, "severity": "CRITICAL" if f == "gets" else "MEDIUM"})
        for f, idx in FORMAT_FUNCS.items():
            m = re.search(rf"\b{f}\s*\(\s*((?:[^,()]|\([^()]*\))*?)\s*[,)]", l) if idx == 0 else None
            if m and m.group(1) and not m.group(1).startswith('"'):
                unsafe_calls.append({"function": "format_string", "line": i, "severity": "MEDIUM", "callee": f})

    all_calls = set(re.findall(r"\b(\w+)\s*\(", code)) - keywords

    return {
        "functions": functions,
        "num_functions": len(functions),
        "num_loops": num_loops,
        "max_nesting_depth": max_depth,
        "num_conditionals": num_conditionals,
        "num_pointer_uses": num_pointer_uses,
        "num_mallocs": len(malloc_lines),
        "num_frees": len(free_lines),
        "function_calls": list(all_calls),
        "function_call_count": len(all_calls),
        "malloc_lines": malloc_lines,
        "free_lines": free_lines,
        "unsafe_calls": unsafe_calls,
    }
