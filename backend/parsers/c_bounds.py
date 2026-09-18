"""
IntelliReview — buffer-bounds reasoning for C (pycparser AST)
=============================================================
A deliberately small, conservative interval analysis. Per function it learns

    * the size of every buffer it can see:  `char b[50]`, `p = malloc(100*sizeof(int))`, `p = b` (alias),
    * integer constants (variables assigned exactly once from a constant expression),
    * value ranges of simple `for` loop counters   (`for (i = 0; i < 100; i++)  ->  i in [0, 99]`),

and then reports only PROVABLE violations:

    * `buf[i]` where the whole range of `i` exceeds the buffer (or is negative)          -> CWE-121/122/124/126/127
    * memcpy/memmove/memset/strncpy/fgets/snprintf/read/fread with a constant size larger than the destination
    * strcpy/strcat whose source (array or literal) is larger than the destination

It can also prove an operation SAFE (e.g. `strcpy(buf16, "ok")`, `memcpy(dst, src, sizeof(dst))`), in which case the generic
"unsafe function" warning for that call is suppressed. Anything it cannot decide is left to the generic rules.
Accesses guarded by an enclosing condition that mentions the index variable are skipped (the guard may make them safe).
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
ALLOC_SIZE_FUNCS = {"malloc", "xmalloc", "alloca"}
Interval = Tuple[int, int]


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
    if isinstance(node, c_ast.Constant) and node.type == "string":
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
        v = c.value
        m = re.fullmatch(r"'(\\.|[^\\])'", v)
        return ord(m.group(1)[-1]) if m else None
    if c.type not in ("int", "unsigned int", "long int", "unsigned long int", "long long int"):
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


class FuncBounds:
    def __init__(self, funcdef: c_ast.FuncDef):
        self.f = funcdef
        self.bytes: Dict[str, int] = {}
        self.esz: Dict[str, int] = {}
        self.is_array: Set[str] = set()
        self.ambiguous: Set[str] = set()
        self.off: Dict[str, int] = {}          # pointer -> element offset from the start of the buffer it points into
        self.consts: Dict[str, int] = {}
        self._const_expr: Dict[str, c_ast.Node] = {}
        self._assigns: Dict[str, int] = {}
        self.loop: List[Dict[str, Interval]] = []
        self.guards: List[Set[str]] = []
        self._write = False
        self.issues: List[dict] = []
        self.verdicts: Dict[int, str] = {}       # id(call) -> "SKIP" | "MEDIUM"
        self._seen: Set[Tuple] = set()

    # ------------------------------------------------------------------ public
    def run(self):
        try:
            self._prepass()
            self._walk(self.f.body)
        except RecursionError:
            pass

    # ------------------------------------------------------------------ pre-pass
    def _set_bytes(self, name: str, nbytes: Optional[int]):
        if nbytes is None or name in self.ambiguous:
            return
        if name in self.bytes and self.bytes[name] != nbytes:
            self.ambiguous.add(name)
            self.bytes.pop(name, None)
            return
        self.bytes[name] = nbytes

    def _learn_assignment(self, name: str, rv):
        self._assigns[name] = self._assigns.get(name, 0) + 1
        self._const_expr[name] = rv
        r = _unwrap(rv)
        if isinstance(r, c_ast.ID) and r.name in self.bytes:
            self._set_bytes(name, self.bytes[r.name])                  # p = buf
            if name not in self.esz and r.name in self.esz:
                self.esz.setdefault(name, self.esz[r.name])
        elif isinstance(r, c_ast.BinaryOp) and r.op in ("+", "-") and isinstance(_unwrap(r.left), c_ast.ID) \
                and _unwrap(r.left).name in self.bytes:
            iv = self._iv(r.right)                                        # p = buf + k   /   p = buf - k
            if iv and iv[0] == iv[1]:
                k = iv[0] if r.op == "+" else -iv[0]
                src = _unwrap(r.left).name
                if name in self.off and self.off[name] != k + self.off.get(src, 0):
                    self.ambiguous.add(name)
                    self.bytes.pop(name, None)
                else:
                    self._set_bytes(name, self.bytes[src])
                    self.off[name] = k + self.off.get(src, 0)
                    if name not in self.esz and src in self.esz:
                        self.esz[name] = self.esz[src]
        elif isinstance(r, c_ast.UnaryOp) and r.op == "&" and isinstance(_unwrap(r.expr), c_ast.ArrayRef):
            ar = _unwrap(r.expr)
            b = _unwrap(ar.name)
            iv = self._iv(ar.subscript)
            if isinstance(b, c_ast.ID) and b.name in self.bytes and iv == (0, 0):
                self._set_bytes(name, self.bytes[b.name])              # p = &buf[0]
        elif isinstance(r, c_ast.FuncCall) and isinstance(r.name, c_ast.ID):
            args = r.args.exprs if r.args else []
            if r.name.name in ALLOC_SIZE_FUNCS and args:
                iv = self._iv(args[0])
                if iv and iv[0] == iv[1]:
                    self._set_bytes(name, iv[0])
            elif r.name.name == "calloc" and len(args) == 2:
                a, b = self._iv(args[0]), self._iv(args[1])
                if a and b and a[0] == a[1] and b[0] == b[1]:
                    self._set_bytes(name, a[0] * b[0])

    def _prepass(self):
        fb = self

        class Pre(c_ast.NodeVisitor):
            def visit_Decl(self, n):
                if n.name is None or isinstance(n.type, c_ast.FuncDecl):
                    return
                t = n.type
                if isinstance(t, c_ast.ArrayDecl):
                    fb.is_array.add(n.name)
                    esz = _type_size(t.type)
                    if esz:
                        fb.esz[n.name] = esz
                    count = None
                    if t.dim is not None:
                        iv = fb._iv(t.dim)
                        count = iv[0] if iv and iv[0] == iv[1] else None
                    elif isinstance(n.init, c_ast.Constant) and n.init.type == "string":
                        count = _literal_elems(n.init)
                    elif isinstance(n.init, c_ast.InitList):
                        count = len(n.init.exprs)
                    if count is not None and esz:
                        fb._set_bytes(n.name, count * esz)
                elif isinstance(t, c_ast.PtrDecl):
                    inner = _type_size(t.type)
                    if inner:
                        fb.esz[n.name] = inner
                if n.init is not None and not isinstance(n.init, c_ast.InitList) and not isinstance(t, c_ast.ArrayDecl):
                    fb._learn_assignment(n.name, n.init)
                if n.init is not None:
                    self.visit(n.init)

            def visit_Assignment(self, n):
                lv = _unwrap(n.lvalue)
                if isinstance(lv, c_ast.ID):
                    if n.op == "=":
                        fb._learn_assignment(lv.name, n.rvalue)
                    else:
                        fb._assigns[lv.name] = fb._assigns.get(lv.name, 0) + 2
                self.visit(n.rvalue)
                self.visit(n.lvalue)

            def visit_UnaryOp(self, n):
                e = _unwrap(n.expr)
                if n.op in ("p++", "p--", "++", "--", "&") and isinstance(e, c_ast.ID):
                    fb._assigns[e.name] = fb._assigns.get(e.name, 0) + 2
                self.generic_visit(n)

        Pre().visit(self.f.body)
        for _ in range(3):                       # resolve constants that depend on other constants
            for name, cnt in self._assigns.items():
                if cnt == 1 and name not in self.consts and name not in self.is_array:
                    iv = self._iv(self._const_expr.get(name))
                    if iv and iv[0] == iv[1]:
                        self.consts[name] = iv[0]

    # ------------------------------------------------------------------ evaluation
    def _sizeof(self, e) -> Optional[int]:
        if isinstance(e, c_ast.Typename):
            return _type_size(e)
        e = _unwrap(e)
        if isinstance(e, c_ast.ID):
            if e.name in self.is_array:
                return self.bytes.get(e.name)
            return 8 if e.name in self.esz else None
        if isinstance(e, c_ast.UnaryOp) and e.op == "*":
            b = _unwrap(e.expr)
            if isinstance(b, c_ast.ID):
                return self.esz.get(b.name)
        if isinstance(e, c_ast.ArrayRef):
            b = _unwrap(e.name)
            if isinstance(b, c_ast.ID):
                return self.esz.get(b.name)
        return None

    def _iv(self, e) -> Optional[Interval]:
        if e is None:
            return None
        e = _unwrap(e)
        if isinstance(e, c_ast.Constant):
            v = _int_value(e)
            return (v, v) if v is not None else None
        if isinstance(e, c_ast.ID):
            for frame in reversed(self.loop):
                if e.name in frame:
                    return frame[e.name]
            if e.name in self.consts:
                return (self.consts[e.name], self.consts[e.name])
            return None
        if isinstance(e, c_ast.UnaryOp):
            if e.op == "sizeof":
                s = self._sizeof(e.expr)
                return (s, s) if s is not None else None
            v = self._iv(e.expr)
            if v is None:
                return None
            if e.op == "-":
                return (-v[1], -v[0])
            if e.op == "+":
                return v
            return None
        if isinstance(e, c_ast.BinaryOp):
            a, b = self._iv(e.left), self._iv(e.right)
            if a is None or b is None:
                return None
            if e.op == "+":
                return (a[0] + b[0], a[1] + b[1])
            if e.op == "-":
                return (a[0] - b[1], a[1] - b[0])
            if e.op == "*":
                c = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
                return (min(c), max(c))
            if e.op == "/" and b[0] == b[1] and b[0] > 0 and a[0] >= 0:
                return (a[0] // b[0], a[1] // b[0])
        return None

    def _bytes_of(self, e) -> Optional[int]:
        e = _unwrap(e)
        if isinstance(e, c_ast.ID):
            return self.bytes.get(e.name)
        if isinstance(e, c_ast.UnaryOp) and e.op == "&":
            ar = _unwrap(e.expr)
            if isinstance(ar, c_ast.ArrayRef) and self._iv(ar.subscript) == (0, 0):
                b = _unwrap(ar.name)
                if isinstance(b, c_ast.ID):
                    return self.bytes.get(b.name)
        return None

    def _elems_of(self, e) -> Optional[int]:
        """Element count of a buffer expression (string literals count their NUL)."""
        lit = _literal_elems(_unwrap(e))
        if lit is not None:
            return lit
        e = _unwrap(e)
        if isinstance(e, c_ast.ID):
            b, s = self.bytes.get(e.name), self.esz.get(e.name)
            if b is not None and s:
                return b // s
        return None

    # ------------------------------------------------------------------ reporting
    def _report(self, line: int, key: Tuple, severity: str, reason: str):
        if key in self._seen:
            return
        self._seen.add(key)
        self.issues.append({"function": "buffer_overflow", "line": line, "severity": severity, "reason": reason})

    # ------------------------------------------------------------------ walking
    def _walk(self, n):
        if n is None:
            return
        t = type(n)
        if t is c_ast.For:
            self._for(n)
        elif t is c_ast.If:
            self._walk(n.cond)
            self.guards.append(_ids_in(n.cond))
            self._walk(n.iftrue)
            self._walk(n.iffalse)
            self.guards.pop()
        elif t in (c_ast.While, c_ast.DoWhile):
            self._walk(n.cond)
            self.guards.append(_ids_in(n.cond))
            self._walk(n.stmt)
            self.guards.pop()
        elif t is c_ast.Assignment:
            self._walk(n.rvalue)
            self._write = True
            self._walk(n.lvalue)
            self._write = False
        elif t is c_ast.ArrayRef:
            self._check_index(n)
            saved, self._write = self._write, False
            self._walk(n.subscript)
            self._write = saved
        elif t is c_ast.FuncCall:
            self._check_call(n)
            if n.args:
                saved, self._write = self._write, False
                self._walk(n.args)
                self._write = saved
        else:
            for _, c in n.children():
                self._walk(c)

    def _for(self, n: c_ast.For):
        binding = self._loop_binding(n)
        if binding:
            self.loop.append(binding)
        self._walk(n.cond)
        self._walk(n.stmt)
        if binding:
            self.loop.pop()

    def _loop_binding(self, n: c_ast.For) -> Optional[Dict[str, Interval]]:
        var = start = None
        init = n.init
        if isinstance(init, c_ast.DeclList) and len(init.decls) == 1 and init.decls[0].init is not None:
            var, start = init.decls[0].name, self._iv(init.decls[0].init)
        elif isinstance(init, c_ast.Assignment) and init.op == "=" and isinstance(_unwrap(init.lvalue), c_ast.ID):
            var, start = _unwrap(init.lvalue).name, self._iv(init.rvalue)
        elif isinstance(init, c_ast.ExprList) and len(init.exprs) == 1 and isinstance(init.exprs[0], c_ast.Assignment):
            a = init.exprs[0]
            if a.op == "=" and isinstance(_unwrap(a.lvalue), c_ast.ID):
                var, start = _unwrap(a.lvalue).name, self._iv(a.rvalue)
        cond = n.cond
        if var is None or start is None or not isinstance(cond, c_ast.BinaryOp):
            return None
        left, right = _unwrap(cond.left), _unwrap(cond.right)
        if not (isinstance(left, c_ast.ID) and left.name == var):
            return None
        limit = self._iv(right)
        if limit is None:
            return None
        step = n.next
        if isinstance(step, c_ast.ExprList) and len(step.exprs) == 1:
            step = step.exprs[0]
        up = down = False
        if isinstance(step, c_ast.UnaryOp) and isinstance(_unwrap(step.expr), c_ast.ID) and _unwrap(step.expr).name == var:
            up, down = step.op in ("p++", "++"), step.op in ("p--", "--")
        elif isinstance(step, c_ast.Assignment) and isinstance(_unwrap(step.lvalue), c_ast.ID) \
                and _unwrap(step.lvalue).name == var and self._iv(step.rvalue) == (1, 1):
            up, down = step.op == "+=", step.op == "-="
        if up and cond.op in ("<", "<=", "!="):
            hi = limit[1] - (1 if cond.op in ("<", "!=") else 0)
            return {var: (start[0], hi)} if start[0] <= hi else None
        if down and cond.op in (">", ">=", "!="):
            lo = limit[0] + (1 if cond.op in (">", "!=") else 0)
            return {var: (lo, start[1])} if lo <= start[1] else None
        return None

    def _guarded(self, names: Set[str]) -> bool:
        return any(names & g for g in self.guards)

    def _check_index(self, n: c_ast.ArrayRef):
        base = _unwrap(n.name)
        if not isinstance(base, c_ast.ID):
            return
        count = self._elems_of(base)
        if count is None:
            return
        iv = self._iv(n.subscript)
        if iv is None or self._guarded(_ids_in(n.subscript)):
            return
        line = n.coord.line if n.coord else 0
        what = "write" if self._write else "read"
        k = self.off.get(base.name, 0)                       # `p = buf - 8` starts 8 elements before the buffer
        lo, hi = iv[0] + k, iv[1] + k
        where = f" (the pointer starts {abs(k)} element(s) {'before' if k < 0 else 'after'} the buffer)" if k else ""
        if hi >= count:
            self._report(line, ("idx", line, base.name, "hi"), "CRITICAL" if self._write else "HIGH",
                         f"out-of-bounds {what}: index can reach {hi} but '{base.name}' holds only {count} "
                         f"element(s){where} (CWE-{'787' if self._write else '125'}).")
        elif lo < 0:
            self._report(line, ("idx", line, base.name, "lo"), "CRITICAL" if self._write else "HIGH",
                         f"out-of-bounds {what}: index can be {lo} (before the start of the buffer){where} "
                         f"(CWE-{'124' if self._write else '127'}).")

    # (dest index, size-arg index, src index, unit)  unit: 'b' bytes | 'e' elements of dest
    _SIZED = {
        "memcpy": (0, 2, 1, "b"), "memmove": (0, 2, 1, "b"), "memset": (0, 2, None, "b"),
        "strncpy": (0, 2, 1, "e"), "strncat": (0, 2, 1, "e"), "wcsncpy": (0, 2, 1, "e"), "wcsncat": (0, 2, 1, "e"),
        "wmemcpy": (0, 2, 1, "e"), "wmemmove": (0, 2, 1, "e"), "wmemset": (0, 2, None, "e"),
        "fgets": (0, 1, None, "b"), "fgetws": (0, 1, None, "e"), "snprintf": (0, 1, None, "b"),
        "swprintf": (0, 1, None, "e"), "read": (1, 2, None, "b"), "recv": (1, 2, None, "b"),
    }

    def _check_call(self, n: c_ast.FuncCall):
        if not isinstance(n.name, c_ast.ID):
            return
        name = n.name.name
        args = n.args.exprs if n.args else []
        line = n.coord.line if n.coord else 0

        if name in ("strcpy", "wcscpy", "strcat", "wcscat") and len(args) >= 2:
            dc, sc = self._elems_of(args[0]), self._elems_of(args[1])
            if dc is not None and sc is not None:
                cat = name.endswith("cat")
                if sc > dc:
                    self._report(line, ("call", line, name), "CRITICAL",
                                 f"{name}() copies {sc} element(s) into a buffer of {dc} (CWE-120).")
                    self.verdicts[id(n)] = "SKIP"
                elif cat:
                    self.verdicts[id(n)] = "MEDIUM"      # dest may already hold data: cannot prove it fits
                else:
                    self.verdicts[id(n)] = "SKIP"        # provably fits
            return

        if name in self._SIZED:
            di, ni, si, unit = self._SIZED[name]
            if len(args) <= max(di, ni):
                return
            nv = self._iv(args[ni])
            db = self._bytes_of(args[di])
            dest = _unwrap(args[di])
            esz = self.esz.get(dest.name, 1) if isinstance(dest, c_ast.ID) else 1
            if nv is None or db is None:
                return
            mult = 1 if unit == "b" else esz
            need = nv[1] * mult
            if need > db:
                self._report(line, ("call", line, name), "CRITICAL",
                             f"{name}() can write up to {need} byte(s) into a {db}-byte buffer (CWE-120).")
                self.verdicts[id(n)] = "SKIP"
                return
            if si is not None and len(args) > si:
                sb = self._bytes_of(args[si])
                if sb is not None and need > sb:
                    self._report(line, ("call", line, name, "read"), "HIGH",
                                 f"{name}() reads up to {need} byte(s) from a {sb}-byte buffer (CWE-126).")
            self.verdicts[id(n)] = "SKIP"                # proven to fit the destination
            return

        if name == "fread" and len(args) == 4:
            a, b, db = self._iv(args[1]), self._iv(args[2]), self._bytes_of(args[0])
            if a and b and db is not None and a[1] * b[1] > db:
                self._report(line, ("call", line, name), "CRITICAL",
                             f"fread() can write {a[1] * b[1]} byte(s) into a {db}-byte buffer (CWE-120).")
