"""
IntelliReview — buffer-bounds reasoning for C (pycparser AST), flow-sensitive
============================================================================
A small abstract interpreter over one function.  It walks the statements in order and keeps, per program point,

    * for every pointer / array: the alternatives (size in bytes, element size, offset in elements) of the buffer it may point at
      (`p = buf`, `p = buf - 8`, `p = malloc(100 * sizeof(int))`, `p++` ...),
    * for every integer: an interval [lo, hi] and whether the value came from OUTSIDE the program (rand, atoi of input,
      scanf ...) and has not been bounded yet,
    * for every char buffer: what is known about the length of the string in it (a literal, `memset(b, 'A', 49); b[49] = 0;`,
      "filled to the last byte, never terminated").

Conditions are folded when constant (`if (5 == 5)`, `if (staticFive == 5)`, an always-true helper) so a dead branch never
pollutes the state; the others refine the intervals on each side (`if (i >= 0 && i < 10)`), and the branches are joined afterwards.
It then reports only what is provable on some path (`buf[i]` outside the buffer, `memcpy` / `strcpy` / `fgets` / ... writing or
reading more than the buffer holds, `strlen` of an unterminated buffer, an index that came from external data and was never
bounded above), and can also prove a copy SAFE, in which case the generic "unsafe function" warning is suppressed.
Anything it cannot decide is left to the generic rules.
"""
import re
from typing import Dict, List, Optional, Tuple, Set

from pycparser import c_ast

BASE_SIZES = {
    "char": 1, "signed char": 1, "unsigned char": 1, "short": 2, "unsigned short": 2, "short int": 2, "int": 4,
    "unsigned": 4, "unsigned int": 4, "signed": 4, "signed int": 4, "float": 4, "long": 8, "unsigned long": 8,
    "long int": 8, "unsigned long int": 8, "long long": 8, "unsigned long long": 8, "double": 8, "size_t": 8,
    "wchar_t": 4, "int8_t": 1, "uint8_t": 1, "int16_t": 2, "uint16_t": 2, "int32_t": 4, "uint32_t": 4, "int64_t": 8,
    "uint64_t": 8, "twoIntsStruct": 8, "charVoid": 32, "BYTE": 1, "uint": 4,
}
ALLOC_SIZE_FUNCS = {"malloc", "xmalloc", "alloca", "_alloca"}
NORETURN = {"exit", "abort", "_exit", "_Exit", "quick_exit", "pthread_exit"}
READONLY_PREFIXES = ("print", "puts", "fputs", "fprintf", "printf", "log", "strlen", "strcmp", "strncmp", "memcmp", "wcslen",
                     "wcscmp", "assert", "show", "dump", "strchr", "strstr", "wcschr")
EXTERNAL_INT_FUNCS = {"rand", "random", "getchar", "getc", "fgetc", "getwchar"}
PARSE_INT_FUNCS = {"atoi", "atol", "atoll", "strtol", "strtoul", "strtoll", "strtoull", "wtoi"}
INF = 10 ** 9
MAX_ALTS = 4
Interval = Tuple[int, int]
# an integer fact: (lo, hi, exact, external)
IntFact = Tuple[int, int, bool, bool]
# a buffer alternative: (bytes, element size, offset in elements)
Buf = Tuple[int, int, int]
UNTERM = "unterminated"


def _unwrap(e):
    while isinstance(e, c_ast.Cast):
        e = e.expr
    return e


def _type_size(t) -> Optional[int]:
    """Size in bytes of a pycparser type node."""
    if isinstance(t, c_ast.Typename):
        return _type_size(t.type)
    if isinstance(t, c_ast.TypeDecl):
        return _type_size(t.type)
    if isinstance(t, c_ast.PtrDecl):
        return 8
    if isinstance(t, c_ast.IdentifierType):
        return BASE_SIZES.get(" ".join(t.names))
    return None


def _literal_elems(node) -> Optional[int]:
    """Elements (incl. NUL) of a string literal constant."""
    if isinstance(node, c_ast.Constant) and node.type in ("string", "wstring"):
        raw = node.value
        try:
            body = raw[raw.index('"') + 1: raw.rindex('"')]
        except ValueError:
            return None
        body = re.sub(r"\\(x[0-9a-fA-F]{1,2}|[0-7]{1,3}|.)", "X", body)
        return len(body) + 1
    return None


def _int_value(c: c_ast.Constant) -> Optional[int]:
    if c.type == "char":
        v = c.value.lstrip("L")
        m = re.fullmatch(r"'(\\.|[^\\])'", v)
        if not m:
            return None
        ch = m.group(1)
        if ch == "\\0":
            return 0
        return ord(ch[-1])
    if c.type not in ("int", "unsigned int", "long int", "unsigned long int", "long long int", "unsigned long long int"):
        return None
    try:
        return int(c.value.rstrip("uUlL"), 0)
    except ValueError:
        return None


def _ids_in(node) -> Set[str]:
    out: Set[str] = set()

    class V(c_ast.NodeVisitor):
        def visit_ID(self, n):
            out.add(n.name)

        def visit_FuncCall(self, n):        # callee names are not variables
            if n.args:
                self.visit(n.args)

    if node is not None:
        V().visit(node)
    return out


class _State:
    """Everything known at one program point.  Values are immutable, so a shallow dict copy is a snapshot."""
    __slots__ = ("bufs", "ints", "slen", "fill", "info", "uninit")

    def __init__(self):
        self.bufs: Dict[str, Tuple[Buf, ...]] = {}
        self.ints: Dict[str, IntFact] = {}
        self.slen: Dict[str, object] = {}         # name -> (lo, hi) | UNTERM
        self.fill: Dict[str, Tuple[int, int]] = {}  # name -> (elements set to a non-zero char, char)
        self.info: Dict[str, Tuple[int, bool, bool]] = {}   # name -> (pointee/element size or scalar size, is_array, is_ptr)
        self.uninit: Set[str] = set()             # buffers (declared arrays, alloca/malloc results) nothing was written to yet

    def copy(self) -> "_State":
        s = _State()
        s.bufs, s.ints, s.slen, s.fill, s.info = (dict(self.bufs), dict(self.ints), dict(self.slen), dict(self.fill),
                                                  dict(self.info))
        s.uninit = set(self.uninit)
        return s

    def forget(self, name: str):
        self.bufs.pop(name, None)
        self.ints.pop(name, None)
        self.slen.pop(name, None)
        self.fill.pop(name, None)
        self.uninit.discard(name)


def _join(a: Optional[_State], b: Optional[_State]) -> Optional[_State]:
    if a is None:
        return b
    if b is None:
        return a
    out = _State()
    out.info = {k: v for k, v in a.info.items() if b.info.get(k) == v}
    for k in a.bufs.keys() & b.bufs.keys():
        alts = tuple(dict.fromkeys(a.bufs[k] + b.bufs[k]))
        if len(alts) <= MAX_ALTS:
            out.bufs[k] = alts
    for k in a.ints.keys() & b.ints.keys():
        x, y = a.ints[k], b.ints[k]
        out.ints[k] = (min(x[0], y[0]), max(x[1], y[1]), x[2] and y[2] and x[:2] == y[:2], x[3] or y[3])
    for k in a.slen.keys() & b.slen.keys():
        x, y = a.slen[k], b.slen[k]
        if x == UNTERM and y == UNTERM:
            out.slen[k] = UNTERM
        elif x != UNTERM and y != UNTERM:
            out.slen[k] = (min(x[0], y[0]), max(x[1], y[1]))
    for k in a.fill.keys() & b.fill.keys():
        if a.fill[k] == b.fill[k]:
            out.fill[k] = a.fill[k]
    out.uninit = a.uninit | b.uninit                # possibly still uninitialised on one of the paths
    return out


class FuncBounds:
    def __init__(self, funcdef: c_ast.FuncDef, consts: Optional[Dict[str, int]] = None, taint=None):
        self.f = funcdef
        self.fconsts: Dict[str, int] = consts or {}
        self.taint = taint                        # parsers.c_taint.FuncTaint (already run) or None
        self.issues: List[dict] = []
        self.verdicts: Dict[int, str] = {}       # id(call) -> "SKIP" | "MEDIUM"
        self._seen: Set[Tuple] = set()
        self._write = False
        self._loop_vars: List[Set[str]] = []

    # ------------------------------------------------------------------ public
    def run(self):
        st = _State()
        try:
            args = self.f.decl.type.args
            for p in (args.params if args else []):
                if isinstance(p, c_ast.Decl) and p.name:
                    self._declare(st, p, param=True)
            self._exec(self.f.body, st)
        except RecursionError:
            pass

    # ------------------------------------------------------------------ helpers
    def _report(self, line: int, key: Tuple, severity: str, reason: str, kind: str = "buffer_overflow"):
        if key in self._seen:
            return
        self._seen.add(key)
        self.issues.append({"function": kind, "line": line, "severity": severity, "reason": reason})

    def _elems_of(self, e, st: Optional[_State] = None) -> Optional[int]:
        """Element count of a buffer expression (string literals count their NUL).  Used by c_parser for sprintf literals."""
        lit = _literal_elems(_unwrap(e))
        if lit is not None:
            return lit
        e = _unwrap(e)
        if isinstance(e, c_ast.ID) and st is not None and e.name in st.bufs:
            alts = st.bufs[e.name]
            if len(alts) == 1:
                return alts[0][0] // alts[0][1] - alts[0][2] if alts[0][1] else None
        return None

    # ------------------------------------------------------------------ declarations
    def _declare(self, st: _State, n: c_ast.Decl, param: bool = False):
        if n.name is None or isinstance(n.type, c_ast.FuncDecl):
            return
        name, t = n.name, n.type
        st.forget(name)
        if isinstance(t, c_ast.ArrayDecl):
            esz = _type_size(t.type)
            st.info[name] = (esz or 0, True, False)
            count = None
            if t.dim is not None:
                iv = self._iv(t.dim, st)
                count = iv[0] if iv and iv[0] == iv[1] else None
            elif isinstance(n.init, c_ast.Constant) and n.init.type in ("string", "wstring"):
                count = _literal_elems(n.init)
            elif isinstance(n.init, c_ast.InitList):
                count = len(n.init.exprs)
            if param:
                return                                   # `char a[]` parameter: really a pointer of unknown extent
            if count is not None and esz:
                st.bufs[name] = ((count * esz, esz, 0),)
            if n.init is None and "static" not in (n.storage or []) and "extern" not in (n.storage or []):
                st.uninit.add(name)
            init = n.init
            if isinstance(init, c_ast.Constant) and init.type in ("string", "wstring"):
                st.slen[name] = (_literal_elems(init) - 1,) * 2
            elif isinstance(init, c_ast.InitList):
                first = init.exprs[0] if init.exprs else None
                v = self._iv(first, st) if first is not None else None
                if len(init.exprs) == 1 and v == (0, 0):
                    st.slen[name] = (0, 0)
            return
        if isinstance(t, c_ast.PtrDecl):
            st.info[name] = (_type_size(t.type) or 0, False, True)
            if n.init is not None and not param:
                self._assign(st, name, n.init)
            return
        sz = _type_size(t)
        st.info[name] = (sz or 0, False, False)
        if n.init is not None and not param and not isinstance(n.init, c_ast.InitList):
            self._assign(st, name, n.init)

    # ------------------------------------------------------------------ evaluation
    def _fact(self, e, st: _State) -> Optional[IntFact]:
        e = _unwrap(e)
        if e is None:
            return None
        if isinstance(e, c_ast.Constant):
            v = _int_value(e)
            return (v, v, True, False) if v is not None else None
        if isinstance(e, c_ast.ID):
            if e.name in st.ints:
                return st.ints[e.name]
            if e.name in self.fconsts and e.name not in st.info:
                v = self.fconsts[e.name]
                return (v, v, True, False)
            return None
        if isinstance(e, c_ast.UnaryOp):
            if e.op == "sizeof":
                s = self._sizeof(e.expr, st)
                return (s, s, True, False) if s is not None else None
            v = self._fact(e.expr, st)
            if v is None:
                return None
            if e.op == "-":
                return (-v[1], -v[0], v[2], v[3])
            if e.op == "+":
                return v
            if e.op == "!":
                return (0, 1, False, False)
            return None
        if isinstance(e, c_ast.FuncCall) and isinstance(e.name, c_ast.ID):
            name = e.name.name
            args = e.args.exprs if e.args else []
            if name in ("strlen", "wcslen") and args:
                a = _unwrap(args[0])
                lit = _literal_elems(a)
                if lit is not None:
                    return (lit - 1, lit - 1, True, False)
                if isinstance(a, c_ast.ID) and isinstance(st.slen.get(a.name), tuple):
                    lo, hi = st.slen[a.name]
                    return (lo, hi, lo == hi, False)
                return None
            if name in EXTERNAL_INT_FUNCS:
                return (0 if name != "getchar" else -1, INF, False, True)
            if name in PARSE_INT_FUNCS and args and self._tainted_expr(args[0]):
                return (-INF, INF, False, True)
            return None
        if isinstance(e, c_ast.BinaryOp):
            a, b = self._fact(e.left, st), self._fact(e.right, st)
            if e.op in ("&&", "||", "==", "!=", "<", ">", "<=", ">="):
                return (0, 1, False, False)
            if a is None or b is None:
                if e.op == "&" and (a is not None or b is not None):
                    k = a or b
                    if k[0] == k[1] and k[0] >= 0:
                        return (0, k[0], False, False)          # x & mask
                if e.op == "%":
                    m = b
                    if m is not None and m[0] == m[1] and m[0] > 0:
                        return (-(m[0] - 1), m[0] - 1, False, False)
                return None
            ext = a[3] or b[3]
            exact = a[2] and b[2]
            if e.op == "+":
                return (_cl(a[0] + b[0]), _cl(a[1] + b[1]), exact, ext)
            if e.op == "-":
                return (_cl(a[0] - b[1]), _cl(a[1] - b[0]), exact, ext)
            if e.op == "*":
                c = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
                return (_cl(min(c)), _cl(max(c)), exact, ext)
            if e.op == "/" and b[0] == b[1] and b[0] > 0 and a[0] >= 0:
                return (a[0] // b[0], _cl(a[1]) // b[0] if a[1] < INF else INF, exact, ext)
            if e.op == "%" and b[0] == b[1] and b[0] > 0 and a[0] >= 0:
                return (0, min(b[0] - 1, a[1]), False, ext)
        if isinstance(e, c_ast.TernaryOp):
            a, b = self._fact(e.iftrue, st), self._fact(e.iffalse, st)
            if a and b:
                return (min(a[0], b[0]), max(a[1], b[1]), False, a[3] or b[3])
        return None

    def _iv(self, e, st: _State) -> Optional[Interval]:
        f = self._fact(e, st)
        return (f[0], f[1]) if f else None

    def _tainted_expr(self, e) -> bool:
        return bool(self.taint is not None and e is not None and self.taint.expr_tainted(e))

    def _sizeof(self, e, st: _State) -> Optional[int]:
        if isinstance(e, c_ast.Typename):
            return _type_size(e)
        e = _unwrap(e)
        if isinstance(e, c_ast.ID):
            info = st.info.get(e.name)
            if info is None:
                return None
            esz, is_arr, is_ptr = info
            if is_arr:
                alts = st.bufs.get(e.name)
                return alts[0][0] if alts and len(alts) == 1 else None
            if is_ptr:
                return 8
            return esz or None
        if isinstance(e, c_ast.UnaryOp) and e.op == "*":
            b = _unwrap(e.expr)
            if isinstance(b, c_ast.ID) and b.name in st.info:
                return st.info[b.name][0] or None
        if isinstance(e, c_ast.ArrayRef):
            b = _unwrap(e.name)
            if isinstance(b, c_ast.ID) and b.name in st.info:
                return st.info[b.name][0] or None
        return None

    def _buf_expr(self, e, st: _State) -> Optional[Tuple[Buf, ...]]:
        """Alternatives of the buffer an expression points into (with offset applied)."""
        e = _unwrap(e)
        if isinstance(e, c_ast.ID):
            return st.bufs.get(e.name)
        if isinstance(e, c_ast.BinaryOp) and e.op in ("+", "-"):
            base = self._buf_expr(e.left, st)
            k = self._iv(e.right, st)
            if base and k and k[0] == k[1]:
                d = k[0] if e.op == "+" else -k[0]
                return tuple((b, s, o + d) for b, s, o in base)
            return None
        if isinstance(e, c_ast.UnaryOp) and e.op == "&":
            ar = _unwrap(e.expr)
            if isinstance(ar, c_ast.ArrayRef):
                base = self._buf_expr(ar.name, st)
                k = self._iv(ar.subscript, st)
                if base and k and k[0] == k[1]:
                    return tuple((b, s, o + k[0]) for b, s, o in base)
        return None

    # ------------------------------------------------------------------ assignment
    def _assign(self, st: _State, name: str, rv):
        """`name = rv` (also a declaration initialiser)."""
        info = st.info.get(name)
        if info is None:
            info = (0, False, False)
        esz, is_arr, is_ptr = info
        r = _unwrap(rv)
        if isinstance(r, c_ast.ID):
            st.uninit.discard(r.name)                # an alias may be used to fill it: stop reporting
        st.uninit.discard(name)
        st.ints.pop(name, None)
        st.slen.pop(name, None)
        st.fill.pop(name, None)
        if is_ptr:
            alts = None
            if isinstance(r, c_ast.Constant) or (isinstance(r, c_ast.ID) and r.name in ("NULL", "nullptr")):
                st.bufs.pop(name, None)
                return
            if isinstance(r, c_ast.FuncCall) and isinstance(r.name, c_ast.ID):
                args = r.args.exprs if r.args else []
                fn = r.name.name
                if fn in ALLOC_SIZE_FUNCS and args:
                    st.uninit.add(name)
                    iv = self._iv(args[0], st)
                    if iv and iv[0] == iv[1] and esz:
                        alts = ((iv[0], esz, 0),)
                elif fn == "calloc" and len(args) == 2:
                    a, b = self._iv(args[0], st), self._iv(args[1], st)
                    if a and b and a[0] == a[1] and b[0] == b[1] and esz:
                        alts = ((a[0] * b[0], esz, 0),)
                elif fn == "realloc" and len(args) == 2:
                    iv = self._iv(args[1], st)
                    if iv and iv[0] == iv[1] and esz:
                        alts = ((iv[0], esz, 0),)
            else:
                alts = self._buf_expr(r, st)
                if alts and isinstance(r, c_ast.ID):
                    if r.name in st.slen:
                        st.slen[name] = st.slen[r.name]
                    if r.name in st.fill:
                        st.fill[name] = st.fill[r.name]
                    if esz and alts and all(s != esz for _, s, _ in alts):
                        # `wchar_t *w = (wchar_t*)bytes;` - the element size of the pointer wins
                        alts = tuple((b, esz, o * s // esz if esz else o) for b, s, o in alts)
            if alts:
                st.bufs[name] = alts
            else:
                st.bufs.pop(name, None)
            return
        # integer / other scalar
        f = self._fact(r, st)
        if f is None and self._tainted_expr(r) and not isinstance(r, c_ast.Constant):
            f = (-INF, INF, False, True)
        if f is None:
            st.ints.pop(name, None)
        else:
            st.ints[name] = f

    # ------------------------------------------------------------------ conditions
    def _cond_value(self, cond, st: _State) -> Optional[bool]:
        f = self._fact(cond, st)
        if f is not None and f[0] == f[1] and f[2]:
            return bool(f[0])
        if isinstance(_unwrap(cond), c_ast.BinaryOp):
            c = _unwrap(cond)
            a, b = self._fact(c.left, st), self._fact(c.right, st)
            if a and b and a[2] and b[2] and a[0] == a[1] and b[0] == b[1]:
                x, y = a[0], b[0]
                return {"==": x == y, "!=": x != y, "<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y}.get(c.op)
        return None

    def _refine(self, cond, truth: bool, st: _State) -> Optional[_State]:
        """State on the `truth` edge of `cond`; None if that edge is impossible."""
        c = _unwrap(cond)
        v = self._cond_value(c, st)
        if v is not None:
            return st if v == truth else None
        if isinstance(c, c_ast.UnaryOp) and c.op == "!":
            return self._refine(c.expr, not truth, st)
        if isinstance(c, c_ast.BinaryOp) and c.op == "&&":
            if truth:
                s1 = self._refine(c.left, True, st)
                return self._refine(c.right, True, s1) if s1 is not None else None
            return st
        if isinstance(c, c_ast.BinaryOp) and c.op == "||":
            if not truth:
                s1 = self._refine(c.left, False, st)
                return self._refine(c.right, False, s1) if s1 is not None else None
            return st
        if isinstance(c, c_ast.ID):
            n = st.copy()
            f = n.ints.get(c.name)
            if f is not None and not truth:
                n.ints[c.name] = (0, 0, f[2], f[3])
            return n
        if isinstance(c, c_ast.BinaryOp) and c.op in ("<", "<=", ">", ">=", "==", "!="):
            left, right = _unwrap(c.left), _unwrap(c.right)
            op = c.op
            if isinstance(left, c_ast.ID) and isinstance(right, c_ast.ID) and left.name not in st.ints \
                    and right.name in st.ints:
                left, right = right, left
                op = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[op]
            elif not isinstance(left, c_ast.ID) and isinstance(right, c_ast.ID):
                left, right = right, left
                op = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[op]
            if isinstance(left, c_ast.ID):
                k = self._iv(right, st)
                if k is None:
                    # compared with something we cannot evaluate (`i < len`): we only know it was *checked*
                    n = st.copy()
                    f = n.ints.get(left.name)
                    if f is not None and f[3]:
                        n.ints[left.name] = (f[0], f[1], f[2], False)      # validated against a limit we cannot see
                    return n
                n = st.copy()
                f = n.ints.get(left.name) or (-INF, INF, False, False)
                if left.name in n.info and n.info[left.name][2]:
                    return n
                if not truth:
                    op = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}[op]
                lo, hi = f[0], f[1]
                if op == "<":
                    hi = min(hi, k[1] - 1)
                elif op == "<=":
                    hi = min(hi, k[1])
                elif op == ">":
                    lo = max(lo, k[0] + 1)
                elif op == ">=":
                    lo = max(lo, k[0])
                elif op == "==":
                    if k[0] == k[1]:
                        lo, hi = max(lo, k[0]), min(hi, k[0])
                if lo > hi:
                    return None
                n.ints[left.name] = (lo, hi, f[2] or (lo == hi), f[3] and (hi >= INF or lo <= -INF))
                return n
        return st

    # ------------------------------------------------------------------ statements
    def _exec(self, n, st: Optional[_State]) -> Optional[_State]:
        """Execute statement `n` in `st`; returns the state afterwards (None = control does not continue)."""
        if n is None or st is None:
            return st
        t = type(n)
        if t is c_ast.Compound:
            saved = {}
            names = [d.name for d in (n.block_items or []) if isinstance(d, c_ast.Decl) and d.name]
            outer = st
            for nm in names:
                saved[nm] = (outer.bufs.get(nm), outer.ints.get(nm), outer.slen.get(nm), outer.fill.get(nm), outer.info.get(nm))
            items = n.block_items or []
            i = 0
            cur: Optional[_State] = st
            while i < len(items) and cur is not None:
                it = items[i]
                if isinstance(it, c_ast.Goto):
                    j = next((k for k in range(i + 1, len(items)) if isinstance(items[k], c_ast.Label)
                              and items[k].name == it.name), None)
                    if j is None:
                        return None
                    i = j
                    continue
                cur = self._exec(it, cur)
                i += 1
            if cur is not None:
                for nm, (b, iv, sl, fi, inf) in saved.items():            # leave the scope: restore what it shadowed
                    for d, v in ((cur.bufs, b), (cur.ints, iv), (cur.slen, sl), (cur.fill, fi), (cur.info, inf)):
                        if v is None:
                            d.pop(nm, None)
                        else:
                            d[nm] = v
            return cur
        if t is c_ast.Decl:
            self._scan_expr(n.init, st)
            self._declare(st, n)
            return st
        if t is c_ast.If:
            self._scan_expr(n.cond, st)
            a = self._refine(n.cond, True, st)
            b = self._refine(n.cond, False, st)
            ra = self._exec(n.iftrue, a.copy() if a is not None else None) if a is not None else None
            rb = self._exec(n.iffalse, b.copy() if b is not None else None) if b is not None else b
            if n.iffalse is None:
                rb = b
            return _join(ra, rb)
        if t is c_ast.For:
            return self._for(n, st)
        if t in (c_ast.While, c_ast.DoWhile):
            return self._while(n, st)
        if t is c_ast.Switch:
            return self._switch(n, st)
        if t is c_ast.Label:
            return self._exec(n.stmt, st)
        if t in (c_ast.Return, c_ast.Break, c_ast.Continue, c_ast.Goto):
            if t is c_ast.Return:
                self._scan_expr(n.expr, st)
            return None
        if t in (c_ast.Case, c_ast.Default):
            cur = st
            for s in n.stmts or []:
                cur = self._exec(s, cur)
            return cur
        if t is c_ast.EmptyStatement:
            return st
        # expression statement
        self._scan_expr(n, st)
        if isinstance(n, c_ast.FuncCall) and isinstance(n.name, c_ast.ID) and n.name.name in NORETURN:
            return None
        return self._after_expr(n, st)

    def _assigned_names(self, n) -> Set[str]:
        out: Set[str] = set()

        class V(c_ast.NodeVisitor):
            def visit_Assignment(s, x):
                b = _base(x.lvalue)
                if b:
                    out.add(b)
                s.generic_visit(x)

            def visit_UnaryOp(s, x):
                if x.op in ("p++", "p--", "++", "--", "&"):
                    b = _base(x.expr)
                    if b:
                        out.add(b)
                s.generic_visit(x)

            def visit_Decl(s, x):
                if x.name:
                    out.add(x.name)
                s.generic_visit(x)

            def visit_FuncCall(s, x):
                if isinstance(x.name, c_ast.ID) and x.name.name in ("fgets", "memset", "strcpy", "strncpy", "memcpy", "memmove",
                                                                     "strcat", "read", "recv", "snprintf", "sprintf", "fread"):
                    a = x.args.exprs if x.args else []
                    idx = 1 if x.name.name in ("read", "recv") else 0
                    if len(a) > idx:
                        b = _base(a[idx])
                        if b:
                            out.add(b)
                s.generic_visit(x)
        if n is not None:
            V().visit(n)
        return out

    def _havoc(self, st: _State, names: Set[str]):
        for nm in names:
            if nm in st.info and st.info[nm][1]:
                st.slen.pop(nm, None)                # array contents unknown; the array itself keeps its size
                st.fill.pop(nm, None)
                continue
            st.forget(nm)

    def _for(self, n: c_ast.For, st: _State) -> Optional[_State]:
        cur = st
        if n.init is not None:
            if isinstance(n.init, c_ast.DeclList):
                for d in n.init.decls:
                    self._scan_expr(d.init, cur)
                    self._declare(cur, d)
            else:
                self._scan_expr(n.init, cur)
                cur = self._after_expr(n.init, cur)
        binding = self._loop_binding(n, cur)
        body_assigned = self._assigned_names(n.stmt) | self._assigned_names(n.next)
        entry = cur.copy()
        self._havoc(entry, body_assigned - ({binding[0]} if binding else set()))
        if binding:
            entry.ints[binding[0]] = (binding[1], binding[2], True, False)
        elif isinstance(n.cond, c_ast.BinaryOp):
            entry = self._refine(n.cond, True, entry) or entry
        self._scan_expr(n.cond, entry)
        self._exec(n.stmt, entry)
        out = cur.copy()
        self._havoc(out, body_assigned)
        if binding:
            out.ints[binding[0]] = (binding[3], binding[3], False, False)
        return out

    def _loop_binding(self, n: c_ast.For, st: _State):
        """(var, lo, hi, exit value) for `for (i = a; i < b; i++)` style loops, else None."""
        var = start = None
        init = n.init
        if isinstance(init, c_ast.DeclList) and len(init.decls) == 1 and init.decls[0].init is not None:
            var, start = init.decls[0].name, self._iv(init.decls[0].init, st)
        elif isinstance(init, c_ast.Assignment) and init.op == "=" and isinstance(_unwrap(init.lvalue), c_ast.ID):
            var, start = _unwrap(init.lvalue).name, self._iv(init.rvalue, st)
        elif isinstance(init, c_ast.ExprList) and len(init.exprs) == 1 and isinstance(init.exprs[0], c_ast.Assignment):
            a = init.exprs[0]
            if a.op == "=" and isinstance(_unwrap(a.lvalue), c_ast.ID):
                var, start = _unwrap(a.lvalue).name, self._iv(a.rvalue, st)
        cond = n.cond
        if var is None or start is None or not isinstance(cond, c_ast.BinaryOp):
            return None
        left, right = _unwrap(cond.left), _unwrap(cond.right)
        if not (isinstance(left, c_ast.ID) and left.name == var):
            return None
        limit = self._iv(right, st)
        if limit is None:
            return None
        step = n.next
        if isinstance(step, c_ast.ExprList) and len(step.exprs) == 1:
            step = step.exprs[0]
        up = down = False
        if isinstance(step, c_ast.UnaryOp) and isinstance(_unwrap(step.expr), c_ast.ID) and _unwrap(step.expr).name == var:
            up, down = step.op in ("p++", "++"), step.op in ("p--", "--")
        elif isinstance(step, c_ast.Assignment) and isinstance(_unwrap(step.lvalue), c_ast.ID) \
                and _unwrap(step.lvalue).name == var and self._iv(step.rvalue, st) == (1, 1):
            up, down = step.op == "+=", step.op == "-="
        if any(isinstance(x, (c_ast.Break, c_ast.Return, c_ast.Goto)) for x in _iter_nodes(n.stmt)):
            return None
        if up and cond.op in ("<", "<=", "!="):
            hi = limit[1] - (1 if cond.op in ("<", "!=") else 0)
            return (var, start[0], hi, hi + 1) if start[0] <= hi else None
        if down and cond.op in (">", ">=", "!="):
            lo = limit[0] + (1 if cond.op in (">", "!=") else 0)
            return (var, lo, start[1], lo - 1) if lo <= start[1] else None
        return None

    def _while(self, n, st: _State) -> Optional[_State]:
        body = n.stmt
        items = list(body.block_items or []) if isinstance(body, c_ast.Compound) else [body]
        inner = [x for it in items for x in _iter_nodes(it)]
        exits = [x for x in inner if isinstance(x, (c_ast.Break, c_ast.Continue, c_ast.Goto, c_ast.Return))]
        v = self._cond_value(n.cond, st)
        once_true = (isinstance(n, c_ast.While) and v is True and items and isinstance(items[-1], c_ast.Break)
                     and len(exits) == 1)                                   # `while (1) { ...; break; }`
        once_false = isinstance(n, c_ast.DoWhile) and v is False and not exits   # `do { ... } while (0)`
        if once_true or once_false:
            cur: Optional[_State] = st
            for it in (items[:-1] if once_true else items):
                cur = self._exec(it, cur)
                if cur is None:
                    break
            if cur is not None:
                return cur
        body_assigned = self._assigned_names(n.stmt) | self._assigned_names(n.cond)
        entry = st.copy()
        self._havoc(entry, body_assigned)
        self._scan_expr(n.cond, entry)
        r = entry if isinstance(n, c_ast.DoWhile) else self._refine(n.cond, True, entry)
        if r is not None:
            self._exec(n.stmt, r.copy())
        out = st.copy()
        self._havoc(out, body_assigned)
        return out

    def _switch(self, n: c_ast.Switch, st: _State) -> Optional[_State]:
        self._scan_expr(n.cond, st)
        body = n.stmt
        items = body.block_items if isinstance(body, c_ast.Compound) and body.block_items else [body]
        cases = [it for it in items if isinstance(it, (c_ast.Case, c_ast.Default))]
        if not cases:
            return st
        sel = self._iv(n.cond, st)
        exact = sel is not None and sel[0] == sel[1]
        chosen = None
        if exact:
            for c in cases:
                if isinstance(c, c_ast.Case):
                    cv = self._iv(c.expr, st)
                    if cv is not None and cv[0] == sel[0]:
                        chosen = c
                        break
            if chosen is None:
                chosen = next((c for c in cases if isinstance(c, c_ast.Default)), None)
        if chosen is not None:
            cur: Optional[_State] = st.copy()
            for s_ in chosen.stmts or []:
                if isinstance(s_, c_ast.Break):
                    break
                cur = self._exec(s_, cur)
                if cur is None:
                    break
            if cur is not None:
                return cur
        out = st.copy()
        self._havoc(out, self._assigned_names(body))
        return out

    # ------------------------------------------------------------------ expressions (checks + side effects)
    def _after_expr(self, n, st: _State) -> _State:
        """Apply the side effects of an expression statement."""
        if isinstance(n, c_ast.ExprList):
            for e in n.exprs:
                st = self._after_expr(e, st)
            return st
        if isinstance(n, c_ast.Assignment):
            lv = _unwrap(n.lvalue)
            if isinstance(lv, c_ast.ID):
                if n.op == "=":
                    self._assign(st, lv.name, n.rvalue)
                elif n.op in ("+=", "-="):
                    k = self._iv(n.rvalue, st)
                    info = st.info.get(lv.name)
                    if info and info[2] and lv.name in st.bufs and k and k[0] == k[1]:
                        d = k[0] if n.op == "+=" else -k[0]
                        st.bufs[lv.name] = tuple((b, s, o + d) for b, s, o in st.bufs[lv.name])
                    else:
                        f = st.ints.get(lv.name)
                        if f and k:
                            if n.op == "+=":
                                st.ints[lv.name] = (_cl(f[0] + k[0]), _cl(f[1] + k[1]), f[2] and k[0] == k[1], f[3])
                            else:
                                st.ints[lv.name] = (_cl(f[0] - k[1]), _cl(f[1] - k[0]), f[2] and k[0] == k[1], f[3])
                        else:
                            st.forget(lv.name) if not (info and info[1]) else None
                else:
                    st.ints.pop(lv.name, None)
            else:
                base = _base(lv)
                if base:
                    st.uninit.discard(base)
                if isinstance(lv, c_ast.ArrayRef):
                    b = _unwrap(lv.name)
                    if isinstance(b, c_ast.ID):
                        self._array_store(st, b.name, lv.subscript, n.rvalue)
            return st
        if isinstance(n, c_ast.UnaryOp) and n.op in ("p++", "++", "p--", "--"):
            e = _unwrap(n.expr)
            if isinstance(e, c_ast.ID):
                d = 1 if n.op in ("p++", "++") else -1
                info = st.info.get(e.name)
                if info and info[2] and e.name in st.bufs:
                    st.bufs[e.name] = tuple((b, s, o + d) for b, s, o in st.bufs[e.name])
                elif e.name in st.ints:
                    f = st.ints[e.name]
                    st.ints[e.name] = (_cl(f[0] + d), _cl(f[1] + d), f[2], f[3])
                else:
                    st.forget(e.name)
            return st
        return st

    def _array_store(self, st: _State, name: str, sub, val):
        """`buf[k] = v`: a NUL at a constant index terminates a string that was filled up to there."""
        k, v = self._iv(sub, st), self._iv(val, st)
        cur = st.slen.get(name)
        if k and k[0] == k[1] and v == (0, 0):
            fl = st.fill.get(name)
            if fl and k[0] <= fl[0]:
                st.slen[name] = (k[0], k[0])
            elif isinstance(cur, tuple):
                st.slen[name] = (min(cur[0], k[0]), min(cur[1], k[0]))
            else:
                st.slen.pop(name, None)
        elif cur is not None and not (cur == UNTERM and v and v[0] > 0):
            st.slen.pop(name, None)
            st.fill.pop(name, None)

    def _scan_expr(self, n, st: _State):
        """Checks: every array access and call inside expression `n` against the current state."""
        if n is None:
            return
        t = type(n)
        if t is c_ast.Assignment:
            self._scan_expr(n.rvalue, st)
            saved, self._write = self._write, True
            self._scan_expr(n.lvalue, st)
            self._write = saved
        elif t is c_ast.ArrayRef:
            self._check_index(n, st)
            saved, self._write = self._write, False
            self._scan_expr(n.subscript, st)
            self._scan_expr(n.name, st)
            self._write = saved
        elif t is c_ast.FuncCall:
            saved, self._write = self._write, False
            if n.args:
                self._scan_expr(n.args, st)
            self._write = saved
            self._check_call(n, st)
            self._call_effects(n, st)
        elif t in (c_ast.Constant, c_ast.ID):
            return
        else:
            for _, c in n.children():
                self._scan_expr(c, st)

    def _check_index(self, n: c_ast.ArrayRef, st: _State):
        base = _unwrap(n.name)
        if isinstance(base, c_ast.ID) and base.name in st.uninit and not self._write:
            line = n.coord.line if n.coord else 0
            self._report(line, ("uninit", line, base.name), "MEDIUM",
                         f"'{base.name}' is read before anything was written to it (uninitialised memory, CWE-457).",
                         "uninit_read")
        if not isinstance(base, c_ast.ID) or base.name not in st.bufs:
            return
        alts = st.bufs[base.name]
        f = self._fact(n.subscript, st)
        if f is None:
            return
        lo, hi, exact, ext = f
        line = n.coord.line if n.coord else 0
        what = "write" if self._write else "read"
        bad_hi = bad_lo = 0
        det_hi = det_lo = None
        for b, s, o in alts:
            count = b // s if s else 0
            if not count:
                return
            if hi < INF and hi + o >= count:
                bad_hi += 1
                det_hi = (hi + o, count, o)
            elif hi >= INF and ext:
                bad_hi += 1
                det_hi = (None, count, o)
            if lo > -INF and lo + o < 0:
                bad_lo += 1
                det_lo = (lo + o, count, o)
            elif lo <= -INF and ext:
                bad_lo += 1
                det_lo = (None, count, o)
        n_alts = len(alts)
        sure = exact or ext
        if bad_hi:
            top, count, off = det_hi
            where = f" (the pointer starts {abs(off)} element(s) {'before' if off < 0 else 'after'} the buffer)" if off else ""
            if top is None:
                msg = (f"out-of-bounds {what}: index into '{base.name}' (holds {count} element(s)) comes from external data "
                       f"and is never checked against an upper bound (CWE-129/{'787' if self._write else '125'}).")
            else:
                msg = (f"out-of-bounds {what}: index can reach {top} but '{base.name}' holds only {count} "
                       f"element(s){where} (CWE-{'787' if self._write else '125'}).")
            sev = ("CRITICAL" if self._write else "HIGH") if (bad_hi == n_alts and sure) else "MEDIUM"
            self._report(line, ("idx", line, base.name, "hi"), sev, msg)
        elif bad_lo:
            low, count, off = det_lo
            where = f" (the pointer starts {abs(off)} element(s) {'before' if off < 0 else 'after'} the buffer)" if off else ""
            if low is None:
                msg = (f"out-of-bounds {what}: index into '{base.name}' comes from external data and is never checked "
                       f"against a lower bound (CWE-129/{'124' if self._write else '127'}).")
            else:
                msg = (f"out-of-bounds {what}: index can be {low} (before the start of the buffer){where} "
                       f"(CWE-{'124' if self._write else '127'}).")
            sev = ("CRITICAL" if self._write else "HIGH") if (bad_lo == n_alts and sure) else "MEDIUM"
            self._report(line, ("idx", line, base.name, "lo"), sev, msg)

    # (dest index, size-arg index, src index, unit)  unit: 'b' bytes | 'e' elements of dest
    _SIZED = {
        "memcpy": (0, 2, 1, "b"), "memmove": (0, 2, 1, "b"), "memset": (0, 2, None, "b"),
        "strncpy": (0, 2, 1, "e"), "strncat": (0, 2, 1, "e"), "wcsncpy": (0, 2, 1, "e"), "wcsncat": (0, 2, 1, "e"),
        "wmemcpy": (0, 2, 1, "e"), "wmemmove": (0, 2, 1, "e"), "wmemset": (0, 2, None, "e"),
        "fgets": (0, 1, None, "b"), "fgetws": (0, 1, None, "e"), "snprintf": (0, 1, None, "b"),
        "swprintf": (0, 1, None, "e"), "read": (1, 2, None, "b"), "recv": (1, 2, None, "b"),
    }

    _FMT = re.compile(r"%([-+ 0#]*)(\d+)?(?:\.(\d+))?(hh|h|ll|l|z|j|t|L)?([diouxXeEfFgGaAcspn%])")
    _INT_DIGITS = {"": 11, "h": 6, "hh": 4, "l": 20, "ll": 20, "z": 20, "j": 20, "t": 20}

    def _format_len(self, fmt, rest, st: _State) -> Optional[int]:
        """Upper bound of the characters (incl. NUL) that a printf format writes, or None if some conversion is unbounded."""
        raw = fmt.value
        body = raw[raw.index('"') + 1: raw.rindex('"')]
        body = re.sub(r"\\(x[0-9a-fA-F]{1,2}|[0-7]{1,3}|.)", "X", body)
        total, pos, k = 0, 0, 0
        for m in self._FMT.finditer(body):
            total += m.start() - pos
            pos = m.end()
            width = int(m.group(2)) if m.group(2) else 0
            prec = int(m.group(3)) if m.group(3) is not None else None
            conv, length = m.group(5), m.group(4) or ""
            if conv == "%":
                total += 1
                continue
            k += 1
            if conv in "di":
                n = self._INT_DIGITS.get(length, 11)
            elif conv in "uo":
                n = self._INT_DIGITS.get(length, 10) - (1 if length in ("", "h", "hh") and conv == "u" else 0)
            elif conv in "xX":
                n = 16 if length in ("l", "ll", "z", "j", "t") else 8
            elif conv == "c":
                n = 1
            elif conv == "s":
                if prec is not None:
                    n = prec
                else:
                    a = _unwrap(rest[k - 1]) if k - 1 < len(rest) else None
                    lit = _literal_elems(a)
                    if lit is not None:
                        n = lit - 1
                    elif isinstance(a, c_ast.ID) and isinstance(st.slen.get(a.name), tuple):
                        n = st.slen[a.name][1]
                    else:
                        return None
            else:
                return None                      # floating point, pointers, %n: not worth bounding
            total += max(n, width)
        total += len(body) - pos
        return total + 1

    def _dest_space(self, e, st: _State):
        """(bytes available from the pointer to the end of its buffer, bytes before it, element size) per alternative."""
        alts = self._buf_expr(e, st)
        if not alts:
            return None
        return [(b - o * s, o * s, s) for b, s, o in alts]

    def _check_call(self, n: c_ast.FuncCall, st: _State):
        if not isinstance(n.name, c_ast.ID):
            return
        name = n.name.name
        args = n.args.exprs if n.args else []
        line = n.coord.line if n.coord else 0

        if name in ("strlen", "wcslen") and args:
            a = _unwrap(args[0])
            if isinstance(a, c_ast.ID) and st.slen.get(a.name) == UNTERM and a.name in st.bufs:
                self._report(line, ("unterm", line, a.name), "HIGH",
                             f"{name}() reads '{a.name}', which is filled to its last byte and never NUL-terminated: "
                             f"it reads past the end of the buffer (CWE-126).")
            return

        if name == "sprintf" and len(args) >= 2 and isinstance(_unwrap(args[1]), c_ast.Constant) \
                and _unwrap(args[1]).type == "string":
            space = self._dest_space(args[0], st)
            need = self._format_len(_unwrap(args[1]), args[2:], st)          # worst-case characters incl. NUL, or None
            if space is not None and need is not None:
                if all(need <= sp[0] // sp[2] for sp in space):
                    self.verdicts[id(n)] = "SKIP"                            # provably fits, whatever the values are
                elif "%" not in _unwrap(args[1]).value:
                    self._report(line, ("call", line, name), "CRITICAL",
                                 f"sprintf() writes {need} byte(s) into a buffer of {space[0][0]} (CWE-120).")
                    self.verdicts[id(n)] = "SKIP"
            return

        if name in ("strcpy", "wcscpy", "strcat", "wcscat") and len(args) >= 2:
            space = self._dest_space(args[0], st)
            src = _unwrap(args[1])
            lit = _literal_elems(src)
            need: Optional[Interval] = None
            if lit is not None:
                need = (lit - 1, lit - 1)
            elif isinstance(src, c_ast.ID) and isinstance(st.slen.get(src.name), tuple):
                need = st.slen[src.name]
            elif isinstance(src, c_ast.ID) and st.slen.get(src.name) == UNTERM and src.name in st.bufs:
                self._report(line, ("unterm", line, src.name), "HIGH",
                             f"{name}() reads '{src.name}', which is filled to its last byte and never NUL-terminated (CWE-126).")
                return
            if space is None:
                return
            cat = name.endswith("cat")
            dst = _unwrap(args[0])
            cur = st.slen.get(dst.name) if isinstance(dst, c_ast.ID) else None
            if cat and not isinstance(cur, tuple):
                if need is not None and all(need[1] + 1 <= sp[0] // sp[2] for sp in space):
                    self.verdicts[id(n)] = "MEDIUM"                      # fits alone; existing content unknown
                return
            if need is None:
                # source is an array whose content we do not know: a bigger array than the destination MAY overflow it
                sb = self._buf_expr(src, st) if isinstance(src, c_ast.ID) else None
                if sb and len(sb) == 1 and len(space) == 1 and sb[0][0] // sb[0][1] > space[0][0] // space[0][2] and not cat:
                    self._report(line, ("call", line, name), "MEDIUM",
                                 f"{name}() copies from a {sb[0][0] // sb[0][1]}-element buffer into one of "
                                 f"{space[0][0] // space[0][2]}; overflow if its content is that long (CWE-120).")
                return
            worst = need[1] + (cur[1] if cat else 0) + 1
            bad = [sp for sp in space if worst > sp[0] // sp[2]]
            if bad:
                sev = "CRITICAL" if len(bad) == len(space) else "MEDIUM"
                self._report(line, ("call", line, name), sev,
                             f"{name}() copies {worst} element(s) into a buffer of {space[0][0] // space[0][2]} (CWE-120).")
                self.verdicts[id(n)] = "SKIP"
            else:
                self.verdicts[id(n)] = "SKIP"
            return

        if name in self._SIZED:
            di, ni, si, unit = self._SIZED[name]
            if len(args) <= max(di, ni):
                return
            nv = self._iv(args[ni], st)
            space = self._dest_space(args[di], st)
            if nv is None or space is None:
                return
            need_of = lambda s: nv[1] * (1 if unit == "b" else s[2])           # noqa: E731
            over = [s for s in space if need_of(s) > s[0]]
            # a pointer that starts BEFORE its buffer: offset < 0 (space counts from the start of the pointer target)
            neg = [b for b in (self._buf_expr(args[di], st) or ()) if b[2] < 0 and nv[1] > 0]
            if neg and name not in ("fgets", "fgetws"):
                self._report(line, ("call", line, name, "lo"), "CRITICAL" if len(neg) == len(space) else "MEDIUM",
                             f"{name}() writes starting {abs(neg[0][2])} element(s) before the start of the buffer (CWE-124).")
                self.verdicts[id(n)] = "SKIP"
                return
            if over:
                need = need_of(over[0])
                self._report(line, ("call", line, name), "CRITICAL" if len(over) == len(space) else "MEDIUM",
                             f"{name}() can write up to {need} byte(s) into a {over[0][0]}-byte buffer (CWE-120).")
                self.verdicts[id(n)] = "SKIP"
                return
            if si is not None and len(args) > si:
                srcs = self._buf_expr(args[si], st)
                if srcs:
                    for b, s, o in srcs:
                        avail = b - o * s
                        need = nv[1] * (1 if unit == "b" else s)
                        if need > avail or o < 0:
                            self._report(line, ("call", line, name, "read"), "HIGH",
                                         f"{name}() reads up to {need} byte(s) from a {avail}-byte buffer (CWE-126).")
                            break
            self.verdicts[id(n)] = "SKIP"                # proven to fit the destination
            return

        if name == "fread" and len(args) == 4:
            a, b, sp = self._iv(args[1], st), self._iv(args[2], st), self._dest_space(args[0], st)
            if a and b and sp and a[1] * b[1] > sp[0][0]:
                self._report(line, ("call", line, name), "CRITICAL",
                             f"fread() can write {a[1] * b[1]} byte(s) into a {sp[0][0]}-byte buffer (CWE-120).")

    def _call_effects(self, n: c_ast.FuncCall, st: _State):
        """State changes made by a call statement."""
        if not isinstance(n.name, c_ast.ID):
            return
        name = n.name.name
        args = n.args.exprs if n.args else []
        if not name.startswith(READONLY_PREFIXES):
            for a in args:
                e = _unwrap(a)
                if isinstance(e, c_ast.UnaryOp) and e.op == "&":
                    e = _unwrap(e.expr)
                b = _base(e)
                if b:
                    st.uninit.discard(b)
        if name in ("memset", "wmemset") and len(args) == 3 and isinstance(_unwrap(args[0]), c_ast.ID):
            d = _unwrap(args[0]).name
            c, cnt = self._iv(args[1], st), self._iv(args[2], st)
            alts = st.bufs.get(d)
            st.slen.pop(d, None)
            st.fill.pop(d, None)
            if c and cnt and cnt[0] == cnt[1] and alts and len(alts) == 1 and c[0] == c[1] and alts[0][2] == 0:
                b, sz, _o = alts[0]
                elems = cnt[0] if name == "wmemset" else cnt[0] // sz
                if c[0] != 0:
                    st.fill[d] = (elems, c[0])
                    if elems * sz >= b:
                        st.slen[d] = UNTERM                 # every element set to a non-zero char: no terminator
                elif elems * sz >= b:
                    st.slen[d] = (0, 0)
            return
        if name in ("strcpy", "wcscpy", "strncpy", "strcat", "wcscat", "fgets", "fgetws", "read", "recv", "sprintf",
                    "snprintf", "memcpy", "memmove", "wmemcpy", "fread", "scanf", "sscanf", "fscanf", "strncat", "wcsncpy"):
            idx = 1 if name in ("read", "recv") else 0
            if name in ("scanf", "sscanf", "fscanf"):
                for a in args[1 if name == "scanf" else 2:]:
                    e = _unwrap(a)
                    if isinstance(e, c_ast.UnaryOp) and e.op == "&" and isinstance(_unwrap(e.expr), c_ast.ID):
                        st.ints[_unwrap(e.expr).name] = (-INF, INF, False, True)      # parsed from external input
                    elif isinstance(e, c_ast.ID):
                        st.slen.pop(e.name, None)
                return
            if len(args) > idx and isinstance(_unwrap(args[idx]), c_ast.ID):
                d = _unwrap(args[idx]).name
                st.fill.pop(d, None)
                if name in ("strcpy", "wcscpy") and len(args) > 1:
                    lit = _literal_elems(_unwrap(args[1]))
                    src = _unwrap(args[1])
                    if lit is not None:
                        st.slen[d] = (lit - 1, lit - 1)
                    elif isinstance(src, c_ast.ID) and src.name in st.slen and st.slen[src.name] != UNTERM:
                        st.slen[d] = st.slen[src.name]
                    else:
                        st.slen.pop(d, None)
                elif name in ("fgets", "fgetws", "read", "recv", "fread") and d in st.bufs:
                    st.slen.pop(d, None)
                else:
                    st.slen.pop(d, None)
            return
        if name == "free" and args and isinstance(_unwrap(args[0]), c_ast.ID):
            st.forget(_unwrap(args[0]).name)
            return
        if name.startswith(READONLY_PREFIXES):
            return
        for a in args:                              # an unknown callee may change whatever it is handed
            e = _unwrap(a)
            if isinstance(e, c_ast.UnaryOp) and e.op == "&" and isinstance(_unwrap(e.expr), c_ast.ID):
                st.forget(_unwrap(e.expr).name)
            elif isinstance(e, c_ast.ID) and e.name in st.info and (st.info[e.name][1] or st.info[e.name][2]):
                st.slen.pop(e.name, None)
                st.fill.pop(e.name, None)


def _cl(v: int) -> int:
    return max(-INF, min(INF, v))


def _base(e) -> Optional[str]:
    e = _unwrap(e)
    if isinstance(e, c_ast.ID):
        return e.name
    if isinstance(e, c_ast.ArrayRef):
        return _base(e.name)
    if isinstance(e, c_ast.StructRef):
        return _base(e.name)
    if isinstance(e, c_ast.UnaryOp):
        return _base(e.expr)
    return None


def _iter_nodes(n):
    if n is None:
        return
    yield n
    for _, c in n.children():
        yield from _iter_nodes(c)
