"""
IntelliReview — lightweight taint tracking for C (pycparser AST)
================================================================
Answers one question per function: *could attacker-influenced data reach this expression?*

Sources of taint
  * pointer / array parameters of the function (incl. argv)   — callers are outside our view, so they are untrusted
  * return values of getenv / gets / fgets / getline / getpass ...
  * buffers filled by fgets / gets / recv / recvfrom / read / fread / scanf-family
Propagation (flow-insensitive, iterated to a fix-point)
  * `a = expr`, `T a = expr`, `a[i] = expr`, `a->f = expr` when expr is tainted
  * strcpy / strcat / strncpy / strncat / memcpy / memmove / sprintf / snprintf: destination becomes tainted if a
    source argument is

Used to decide severity: an unbounded `strcpy`, `system(x)` or `printf(x)` is HIGH only when `x` is tainted, and MEDIUM
when its provenance is unknown/local. (A logging wrapper `vprintf(fmt, ap)` in a helper is not "High Risk".)
"""
from typing import Set

from pycparser import c_ast

RETURNS_TAINTED = {"getenv", "gets", "fgets", "getline", "getpass", "readline", "gets_s", "fgetws", "secure_getenv"}
FILLS_DEST = {"fgets": [0], "gets": [0], "recv": [1], "recvfrom": [1], "read": [1], "fread": [0], "fgetws": [0],
              "getline": [0], "pread": [1]}
SCAN_FAMILY = {"scanf": 1, "sscanf": 2, "fscanf": 2, "wscanf": 1}          # first destination argument index
COPIES = {"strcpy": 0, "strcat": 0, "strncpy": 0, "strncat": 0, "memcpy": 0, "memmove": 0, "wcscpy": 0, "wcscat": 0,
          "sprintf": 0, "snprintf": 0, "wcsncpy": 0, "wcsncat": 0, "strdup": None, "strndup": None}
PROPAGATING_RETURNS = {"strchr", "strrchr", "strstr", "strtok", "strdup", "strndup", "strpbrk", "memchr"}


def _unwrap(e):
    while isinstance(e, c_ast.Cast):
        e = e.expr
    return e


def _base_id(e):
    """Name of the variable that an lvalue / address expression ultimately refers to."""
    e = _unwrap(e)
    if isinstance(e, c_ast.ID):
        return e.name
    if isinstance(e, c_ast.UnaryOp) and e.op in ("&", "*"):
        return _base_id(e.expr)
    if isinstance(e, c_ast.ArrayRef):
        return _base_id(e.name)
    if isinstance(e, c_ast.StructRef):
        return _base_id(e.name)
    if isinstance(e, c_ast.BinaryOp) and e.op in ("+", "-"):
        return _base_id(e.left) or _base_id(e.right)
    return None


class FuncTaint:
    def __init__(self, funcdef: c_ast.FuncDef):
        self.f = funcdef
        self.tainted: Set[str] = set()

    def run(self):
        try:
            args = self.f.decl.type.args
            for p in (args.params if args else []):
                if isinstance(p, c_ast.Decl) and p.name and isinstance(p.type, (c_ast.PtrDecl, c_ast.ArrayDecl)):
                    self.tainted.add(p.name)
        except AttributeError:
            pass
        for _ in range(4):
            before = len(self.tainted)
            self._scan(self.f.body)
            if len(self.tainted) == before:
                break

    # ------------------------------------------------------------------
    def expr_tainted(self, e) -> bool:
        e = _unwrap(e)
        if e is None:
            return False
        if isinstance(e, c_ast.ID):
            return e.name in self.tainted
        if isinstance(e, c_ast.Constant):
            return False
        if isinstance(e, c_ast.FuncCall):
            name = e.name.name if isinstance(e.name, c_ast.ID) else None
            if name in RETURNS_TAINTED:
                return True
            if name in PROPAGATING_RETURNS or name is None:
                return any(self.expr_tainted(a) for a in (e.args.exprs if e.args else []))
            return False
        return any(self.expr_tainted(c) for _, c in e.children())

    def _taint(self, e):
        name = _base_id(e)
        if name:
            self.tainted.add(name)

    def _scan(self, node):
        if node is None:
            return
        if isinstance(node, c_ast.Assignment):
            if self.expr_tainted(node.rvalue):
                self._taint(node.lvalue)
        elif isinstance(node, c_ast.Decl):
            if node.init is not None and node.name and not isinstance(node.init, c_ast.InitList) \
                    and self.expr_tainted(node.init):
                self.tainted.add(node.name)
        elif isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID):
            name = node.name.name
            args = node.args.exprs if node.args else []
            for i in FILLS_DEST.get(name, []):
                if i < len(args):
                    self._taint(args[i])
            if name in SCAN_FAMILY:
                for a in args[SCAN_FAMILY[name]:]:
                    self._taint(a)
            if name in COPIES and COPIES[name] is not None and len(args) > COPIES[name] + 1:
                if any(self.expr_tainted(a) for a in args[COPIES[name] + 1:]):
                    self._taint(args[COPIES[name]])
        for _, c in node.children():
            self._scan(c)
