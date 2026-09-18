"""
IntelliReview V3 — Core Control Flow Graph (CFG) Builder
=========================================================
Converts a pycparser AST into a real Control Flow Graph.
Nodes are basic blocks; edges are True/False/Unconditional/case/break/goto jumps.

Every node also carries two *ordered* operation streams that the analyzers consume:

  ordered_ops : (op, var, line)  op in ASSIGN | USE | KILL      -> uninitialized-variable analysis
  ptr_ops     : (op, var, line)  op in ALLOC | REALLOC | NULL | FREE | DEREF | USEARG | PASS | ESCAPE
                                 | ASSIGN_OTHER                  -> pointer state analysis

and, for branch nodes, `null_check = {"var": p, "true_is_null": bool}` so that the pointer analysis can
follow `if (!p) return;` without reporting a leak on the (correct) NULL path.
"""
from typing import List, Dict, Any, Optional, Tuple
from pycparser import c_ast

ALLOC_FUNCS = {"malloc", "calloc", "aligned_alloc", "strdup", "strndup", "wcsdup", "xmalloc", "xcalloc"}
NORETURN_FUNCS = {"exit", "abort", "_exit", "_Exit", "quick_exit", "pthread_exit"}
# OS resources that must be released: opened by RES_OPEN_FUNCS, closed by RES_CLOSE_FUNCS
RES_OPEN_FUNCS = {"fopen", "fdopen", "tmpfile", "opendir", "popen", "open", "creat", "socket", "accept",
                  "fopen64", "OPEN", "SOCKET"}
RES_CLOSE_FUNCS = {"fclose", "pclose", "closedir", "close", "CLOSE_SOCKET", "closesocket"}

# Callees that only *read/write through* a pointer and never take ownership of it.
NON_OWNING = {
    "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf", "puts", "fputs", "putchar", "fputc",
    "strlen", "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp", "strchr", "strrchr", "strstr",
    "strtok", "memcpy", "memmove", "memset", "memcmp", "memchr", "fgets", "gets", "scanf", "sscanf", "fscanf",
    "fread", "fwrite", "read", "write", "recv", "send", "atoi", "atol", "atof", "strtol", "strtoul", "strtod",
    "perror", "isalpha", "isdigit", "toupper", "tolower", "wcslen", "wcscpy", "wcsncpy", "wcscat", "wcsncat",
    "wprintf", "swprintf", "assert", "qsort", "bsearch", "strcasecmp", "strncasecmp",
    # stdio / POSIX calls that use a FILE* / descriptor without taking ownership of it
    "fgetc", "getc", "fputc", "putc", "ungetc", "fseek", "ftell", "rewind", "feof", "ferror", "fflush", "setvbuf",
    "fileno", "fgetwc", "fputwc", "fgetws", "fputws", "fwprintf", "vfwprintf", "lseek", "fstat", "ioctl", "fcntl",
    "dup", "dup2", "select", "poll", "listen", "bind", "connect", "getsockopt", "setsockopt", "recvfrom", "sendto",
    "readdir", "rewinddir", "flock", "fsync", "ftruncate", "mmap", "getline", "getdelim", "fscanf", "vfscanf",
}
# library calls that read or write THROUGH these pointer arguments: passing NULL / freed memory there is a real dereference
DEREF_ARGS = {
    "strcpy": (0, 1), "strncpy": (0, 1), "strcat": (0, 1), "strncat": (0, 1), "memcpy": (0, 1), "memmove": (0, 1),
    "memset": (0,), "strlen": (0,), "strcmp": (0, 1), "strncmp": (0, 1), "memcmp": (0, 1), "sprintf": (0,),
    "snprintf": (0,), "fgets": (0,), "gets": (0,), "read": (1,), "recv": (1,), "wcscpy": (0, 1), "wcslen": (0,),
    "wcscat": (0, 1), "wcsncpy": (0, 1), "strchr": (0,), "strstr": (0, 1), "atoi": (0,), "puts": (0,), "fputs": (0,),
}
NON_OWNING_PREFIXES = ("print", "puts", "put", "log", "show", "display", "dump", "write", "str", "mem", "is", "has",
                       "get", "check", "print_", "trace", "debug")


def _unwrap(expr):
    """Strip casts and parentheses-like wrappers."""
    while isinstance(expr, c_ast.Cast):
        expr = expr.expr
    return expr


def _is_null_const(expr) -> bool:
    expr = _unwrap(expr)
    if isinstance(expr, c_ast.ID) and expr.name in ("NULL", "nullptr"):
        return True
    return isinstance(expr, c_ast.Constant) and expr.type == "int" and expr.value.rstrip("uUlL") in ("0", "0x0")


def _is_minus_one(expr) -> bool:
    e = _unwrap(expr)
    if isinstance(e, c_ast.ID) and e.name in ("INVALID_SOCKET", "SOCKET_ERROR", "INVALID_HANDLE_VALUE"):
        return True
    return isinstance(e, c_ast.UnaryOp) and e.op == "-" and isinstance(e.expr, c_ast.Constant) \
        and e.expr.value.rstrip("uUlL") == "1"


def _is_zero(expr) -> bool:
    e = _unwrap(expr)
    return isinstance(e, c_ast.Constant) and e.type == "int" and e.value.rstrip("uUlL") in ("0", "0x0")


def _alloc_call_kind(expr) -> Optional[Tuple[str, Optional[str]]]:
    """('ALLOC'|'REALLOC', realloc_source_var) if expr is an allocation call, else None."""
    expr = _unwrap(expr)
    if isinstance(expr, c_ast.FuncCall) and isinstance(expr.name, c_ast.ID):
        name = expr.name.name
        if name in ALLOC_FUNCS:
            return ("ALLOC", None)
        if name in RES_OPEN_FUNCS:
            return ("RES", None)
        if name == "realloc":
            first = _unwrap(expr.args.exprs[0]) if expr.args and expr.args.exprs else None
            return ("REALLOC", first.name if isinstance(first, c_ast.ID) else None)
    return None


def _const_value(expr, consts: Dict[str, int]) -> Optional[int]:
    """Value of a compile-time constant integer expression (literals + file-level constants), else None."""
    e = _unwrap(expr)
    if e is None:
        return None
    if isinstance(e, c_ast.Constant):
        if e.type in ("int", "unsigned int", "long int", "unsigned long int"):
            try:
                return int(e.value.rstrip("uUlL"), 0)
            except ValueError:
                return None
        return None
    if isinstance(e, c_ast.ID):
        return consts.get(e.name)
    if isinstance(e, c_ast.UnaryOp):
        v = _const_value(e.expr, consts)
        if v is None:
            return None
        if e.op == "-":
            return -v
        if e.op == "!":
            return int(not v)
        if e.op == "+":
            return v
        return None
    if isinstance(e, c_ast.BinaryOp):
        a, b = _const_value(e.left, consts), _const_value(e.right, consts)
        if a is None or b is None:
            return None
        ops = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
               "==": lambda: int(a == b), "!=": lambda: int(a != b), "<": lambda: int(a < b), ">": lambda: int(a > b),
               "<=": lambda: int(a <= b), ">=": lambda: int(a >= b), "&&": lambda: int(bool(a) and bool(b)),
               "||": lambda: int(bool(a) or bool(b))}
        if e.op in ops:
            return ops[e.op]()
        if e.op == "/" and b != 0:
            return int(a / b)
        if e.op == "%" and b != 0:
            return a % b
    return None


def _file_constants(ast) -> Dict[str, int]:
    """File-scope variables with a constant initialiser that are `const`, or `static` and never modified anywhere."""
    if ast is None:
        return {}
    cands: Dict[str, int] = {}
    for ext in getattr(ast, "ext", []) or []:
        if isinstance(ext, c_ast.Decl) and ext.name and ext.init is not None and isinstance(ext.type, c_ast.TypeDecl):
            v = _const_value(ext.init, cands)
            if v is not None and ("const" in (ext.quals or []) or "static" in (ext.storage or [])):
                cands[ext.name] = v
    if not cands:
        return {}
    modified = set()

    class M(c_ast.NodeVisitor):
        def visit_Assignment(self, n):
            lv = _unwrap(n.lvalue)
            if isinstance(lv, c_ast.ID):
                modified.add(lv.name)
            self.generic_visit(n)

        def visit_UnaryOp(self, n):
            e = _unwrap(n.expr)
            if n.op in ("p++", "p--", "++", "--", "&") and isinstance(e, c_ast.ID):
                modified.add(e.name)
            self.generic_visit(n)

    M().visit(ast)
    return {k: v for k, v in cands.items() if k not in modified}


def _null_check(cond) -> Optional[Dict[str, Any]]:
    """Recognise `p`, `!p`, `p == NULL`, `p != NULL`, `!(p = malloc(..))`, `(p = malloc(..)) == NULL`."""
    def var_of(e):
        e = _unwrap(e)
        if isinstance(e, c_ast.Assignment) and isinstance(e.lvalue, c_ast.ID) and e.op == "=":
            return e.lvalue.name
        if isinstance(e, c_ast.ID):
            return e.name
        return None

    cond = _unwrap(cond)
    if isinstance(cond, c_ast.UnaryOp) and cond.op == "!":
        v = var_of(cond.expr)
        return {"var": v, "true_is_null": True} if v else None
    if isinstance(cond, c_ast.BinaryOp) and cond.op in ("==", "!=", "<", ">=", ">", "<="):
        a, b = cond.left, cond.right
        v = var_of(a)
        if v is not None:
            if cond.op in ("==", "!=") and (_is_null_const(b) or _is_minus_one(b)):
                return {"var": v, "true_is_null": cond.op == "=="}
            if cond.op == "<" and _is_zero(b):
                return {"var": v, "true_is_null": True}          # fd < 0   -> failed
            if cond.op == ">=" and _is_zero(b):
                return {"var": v, "true_is_null": False}         # fd >= 0  -> ok
            if cond.op == ">" and _is_minus_one(b):
                return {"var": v, "true_is_null": False}         # fd > -1  -> ok
        v = var_of(b)
        if v is not None and cond.op in ("==", "!=") and (_is_null_const(a) or _is_minus_one(a)):
            return {"var": v, "true_is_null": cond.op == "=="}
        return None
    v = var_of(cond)
    return {"var": v, "true_is_null": False} if v else None


class CFGNode:
    def __init__(self, node_id: int, name: str = "Block"):
        self.id = node_id
        self.name = name
        self.statements = []      # raw pycparser AST statements in this block
        self.out_edges = []       # (target_node_id, edge_condition)
        self.is_return = False
        self.noreturn = False     # exit()/abort(): program ends, no leak check
        self.cond_expr = None     # branch condition evaluated at the end of this block

    def add_stmt(self, stmt):
        self.statements.append(stmt)

    def add_edge(self, target_id: int, condition: str = "unconditional"):
        self.out_edges.append((target_id, condition))

    def to_dict(self, locals_info: Dict[str, Any]):
        ordered_ops: List[Tuple[str, str, int]] = []
        ptr_ops: List[Tuple[str, str, int]] = []
        calls: List[str] = []
        decls: List[str] = []
        tracked = locals_info["tracked_scalars"]
        array_vars = locals_info.get("array_vars", set())

        def line_of(n) -> int:
            try:
                return int(n.coord.line) if n is not None and n.coord else -1
            except (AttributeError, TypeError, ValueError):
                return -1

        def bare_id(expr) -> Optional[str]:
            e = _unwrap(expr)
            return e.name if isinstance(e, c_ast.ID) else None

        class Inspector(c_ast.NodeVisitor):
            # --- expressions -------------------------------------------------
            def visit_FuncCall(self, n):
                name = n.name.name if isinstance(n.name, c_ast.ID) else None
                args = n.args.exprs if n.args else []
                line = line_of(n)
                if name:
                    calls.append(name)
                if name == "free":
                    if args:
                        a0 = _unwrap(args[0])
                        v = bare_id(a0)
                        if v:
                            ptr_ops.append(("FREE", v, line))
                        elif isinstance(a0, c_ast.UnaryOp) and a0.op == "&":
                            ptr_ops.append(("BADFREE", "stack", line))       # free(&x): not heap memory
                        elif isinstance(a0, c_ast.BinaryOp) and a0.op in ("+", "-") and \
                                (bare_id(a0.left) or bare_id(a0.right)):
                            ptr_ops.append(("BADFREE", "offset", line))      # free(p + n): not the start of the block
                        else:
                            self.visit(args[0])
                    return
                if name in RES_CLOSE_FUNCS:
                    v = bare_id(args[0]) if args else None
                    if v:
                        ptr_ops.append(("RES_FREE", v, line))
                        ordered_ops.append(("USE", v, line))
                        return
                if name in ("va_start", "va_copy") and args and bare_id(args[0]):
                    ordered_ops.append(("ASSIGN", bare_id(args[0]), line))     # the macro initialises its va_list
                    return
                if name in NORETURN_FUNCS:
                    for a in args:
                        self.visit(a)
                    return
                non_owning = bool(name) and (name in NON_OWNING or name.startswith(NON_OWNING_PREFIXES))
                for i, a in enumerate(args):
                    v = bare_id(a)
                    if v:
                        ordered_ops.append(("USE", v, line))
                        # PASS@callee@argindex lets the pointer analysis look the callee up: a function defined in
                        # this file that never frees or stores that parameter does not take ownership of it
                        if name in DEREF_ARGS and i in DEREF_ARGS[name]:
                            ptr_ops.append(("DEREF", v, line))
                        else:
                            ptr_ops.append(("USEARG" if non_owning else (f"PASS@{name}@{i}" if name else "PASS"), v, line))
                    else:
                        self.visit(a)
                if name is None:            # call through an expression: visit it
                    self.visit(n.name)

            def visit_Assignment(self, n):
                line = line_of(n)
                lv = _unwrap(n.lvalue)
                rv = n.rvalue
                if isinstance(lv, c_ast.ID):
                    var = lv.name
                    if n.op != "=":
                        ordered_ops.append(("USE", var, line))
                    if n.op in ("+=", "-="):
                        self.visit(rv)
                        ptr_ops.append(("OFFSET", var, line))
                    else:
                        self._classify_rvalue(var, rv, line)
                    ordered_ops.append(("ASSIGN", var, line))
                else:
                    # s->f = p, *out = p, a[i] = p ...: the value escapes into memory we do not track
                    src = bare_id(rv)
                    if src:
                        ordered_ops.append(("USE", src, line))
                        ptr_ops.append(("ESCAPE", src, line))
                    else:
                        self.visit(rv)
                    self._visit_lvalue(lv)

            def _visit_lvalue(self, lv):
                """Writing through lvalue: base pointer is dereferenced, indices are read."""
                if isinstance(lv, c_ast.StructRef):
                    base = _unwrap(lv.name)
                    if isinstance(base, c_ast.ID):
                        if lv.type == "->":
                            ptr_ops.append(("DEREF", base.name, line_of(lv)))
                            ordered_ops.append(("USE", base.name, line_of(lv)))
                        else:
                            ordered_ops.append(("ASSIGN", base.name, line_of(lv)))  # partial init of a struct
                    else:
                        self.visit(base)
                elif isinstance(lv, c_ast.ArrayRef):
                    base = _unwrap(lv.name)
                    if isinstance(base, c_ast.ID):
                        ptr_ops.append(("DEREF", base.name, line_of(lv)))
                        ordered_ops.append(("ASSIGN", base.name, line_of(lv)))  # element write initialises the array
                    else:
                        self.visit(base)
                    self.visit(lv.subscript)
                elif isinstance(lv, c_ast.UnaryOp) and lv.op == "*":
                    self._deref_expr(lv.expr, line_of(lv))
                else:
                    self.visit(lv)

            def _deref_expr(self, expr, line):
                e = _unwrap(expr)
                if isinstance(e, c_ast.ID):
                    ptr_ops.append(("DEREF", e.name, line))
                    ordered_ops.append(("USE", e.name, line))
                elif isinstance(e, c_ast.BinaryOp) and e.op in ("+", "-"):
                    self._deref_expr(e.left, line)
                    self._deref_expr(e.right, line)
                else:
                    self.visit(e)

            def _classify_rvalue(self, var, rv, line):
                """Emit the pointer ops for `var = rv` (does not emit ASSIGN for the data-flow stream)."""
                kind = _alloc_call_kind(rv)
                if kind and kind[0] == "RES":
                    call = _unwrap(rv)
                    for a in (call.args.exprs if call.args else []):
                        self.visit(a)
                    ptr_ops.append(("RES_ALLOC", var, line))
                    return
                if kind:
                    call = _unwrap(rv)
                    for a in (call.args.exprs if call.args else []):
                        if bare_id(a) and kind[0] == "REALLOC" and bare_id(a) == kind[1]:
                            continue
                        self.visit(a)
                    if kind[0] == "REALLOC":
                        if kind[1] == var:
                            ptr_ops.append(("REALLOC", var, line))
                        else:
                            if kind[1]:
                                ptr_ops.append(("ESCAPE", kind[1], line))
                            ptr_ops.append(("ALLOC", var, line))
                    else:
                        ptr_ops.append(("ALLOC", var, line))
                    return
                if _is_null_const(rv):
                    ptr_ops.append(("NULL", var, line))
                    return
                rc = _unwrap(rv)
                if isinstance(rc, c_ast.FuncCall) and isinstance(rc.name, c_ast.ID) and rc.name.name in ("alloca", "_alloca"):
                    for a in (rc.args.exprs if rc.args else []):
                        self.visit(a)
                    ptr_ops.append(("STACKPTR", var, line))                   # alloca() memory lives on the stack
                    return
                r0 = _unwrap(rv)
                if isinstance(r0, c_ast.UnaryOp) and r0.op == "&" and isinstance(_unwrap(r0.expr), (c_ast.ID, c_ast.ArrayRef)):
                    self.visit(r0.expr)
                    ptr_ops.append(("STACKPTR", var, line))                   # p = &x  /  p = &a[i]
                    return
                if isinstance(r0, c_ast.BinaryOp) and r0.op in ("+", "-") and bare_id(r0.left) == var:
                    self.visit(r0.right)
                    ptr_ops.append(("OFFSET", var, line))                     # p = p + n
                    return
                src = bare_id(rv)
                if src and src in array_vars:
                    ordered_ops.append(("USE", src, line))
                    ptr_ops.append(("STACKPTR", var, line))                   # p = buf  (buf is a local array)
                    return
                if src:                       # alias: q = p   (state is copied if it is NULL / freed / stack, else both untracked)
                    ordered_ops.append(("USE", src, line))
                    ptr_ops.append((f"ALIAS@{src}", var, line))
                    return
                self.visit(rv)
                ptr_ops.append(("ASSIGN_OTHER", var, line))

            def visit_Decl(self, n):
                if n.name is None:
                    return
                line = line_of(n)
                decls.append(n.name)
                if n.init is not None:
                    if isinstance(n.init, c_ast.InitList):
                        self.visit(n.init)
                    else:
                        self._classify_rvalue(n.name, n.init, line)
                    ordered_ops.append(("ASSIGN", n.name, line))
                else:
                    ordered_ops.append(("KILL", n.name, line))
                    if isinstance(n.type, c_ast.ArrayDecl):
                        ordered_ops.append(("ASSIGN", n.name, line))   # arrays are filled element-wise

            def visit_ID(self, n):
                ordered_ops.append(("USE", n.name, line_of(n)))

            def visit_UnaryOp(self, n):
                line = line_of(n)
                if n.op == "sizeof":
                    return                                   # unevaluated
                if n.op == "&":
                    e = _unwrap(n.expr)
                    if isinstance(e, c_ast.ID):
                        ordered_ops.append(("ASSIGN", e.name, line))   # address taken: may be initialised by callee
                        return
                    self.visit(n.expr)
                    return
                if n.op == "*":
                    self._deref_expr(n.expr, line)
                    return
                if n.op in ("p++", "p--", "++", "--"):
                    e = _unwrap(n.expr)
                    if isinstance(e, c_ast.ID):
                        ordered_ops.append(("USE", e.name, line))
                        ordered_ops.append(("ASSIGN", e.name, line))
                        ptr_ops.append(("OFFSET", e.name, line))
                        return
                self.visit(n.expr)

            def visit_StructRef(self, n):
                base = _unwrap(n.name)
                if isinstance(base, c_ast.ID):
                    ordered_ops.append(("USE", base.name, line_of(n)))
                    if n.type == "->":
                        ptr_ops.append(("DEREF", base.name, line_of(n)))
                else:
                    self.visit(base)

            def visit_ArrayRef(self, n):
                base = _unwrap(n.name)
                if isinstance(base, c_ast.ID):
                    ordered_ops.append(("USE", base.name, line_of(n)))
                    ptr_ops.append(("DEREF", base.name, line_of(n)))
                else:
                    self.visit(base)
                self.visit(n.subscript)

            def visit_Return(self, n):
                if n.expr is not None:
                    v = bare_id(n.expr)
                    if v:
                        ordered_ops.append(("USE", v, line_of(n)))
                        ptr_ops.append(("ESCAPE", v, line_of(n)))
                    else:
                        self.visit(n.expr)

            def visit_Typedef(self, n):
                return

            def visit_Struct(self, n):
                return

        inspector = Inspector()
        for stmt in self.statements:
            inspector.visit(stmt)
        if self.cond_expr is not None:
            inspector.visit(self.cond_expr)

        min_line = -1
        for st in self.statements + ([self.cond_expr] if self.cond_expr is not None else []):
            coord = getattr(st, "coord", None)
            if coord is not None:
                ln = int(coord.line)
                if min_line == -1 or ln < min_line:
                    min_line = ln

        ret_line = -1
        for st in self.statements:
            if isinstance(st, c_ast.Return) and getattr(st, "coord", None) is not None:
                ret_line = int(st.coord.line)

        return {
            "id": self.id,
            "line": min_line,
            "return_line": ret_line if ret_line != -1 else min_line,
            "name": self.name,
            "num_stmts": len(self.statements),
            "out_edges": self.out_edges,
            "is_return": self.is_return,
            "noreturn": self.noreturn,
            "ordered_ops": ordered_ops,
            "ptr_ops": ptr_ops,
            "calls": calls,
            "decls": list(dict.fromkeys(decls)),
            "cond_var_has_uses": any(o[0] == "USE" for o in ordered_ops) if self.cond_expr is not None else False,
            "null_check": _null_check(self.cond_expr) if self.cond_expr is not None else None,
            "cond_is_const_true": _is_const_true(self.cond_expr) if self.cond_expr is not None else False,
            "cond_uses": sorted({o[1] for o in _cond_uses(self.cond_expr)}) if self.cond_expr is not None else [],
        }


def _is_const_true(expr) -> bool:
    e = _unwrap(expr)
    return isinstance(e, c_ast.Constant) and e.type == "int" and e.value.rstrip("uUlL") not in ("0", "0x0")


def _cond_uses(expr):
    found = []

    class V(c_ast.NodeVisitor):
        def visit_ID(self, n):
            found.append(("USE", n.name))

    V().visit(expr)
    return found


class CFGBuilder(c_ast.NodeVisitor):
    def __init__(self, consts: Optional[Dict[str, int]] = None):
        self.consts: Dict[str, int] = consts or {}
        self.nodes: Dict[int, CFGNode] = {}
        self.node_counter = 0
        self.current_node_id = None
        self.entry_node_id = None
        self.loop_break_targets = []
        self.loop_continue_targets = []
        self.switch_stack = []            # [{"cond": node_id, "has_default": bool, "exit": node_id}]
        self.labels: Dict[str, int] = {}
        self.pending_gotos: List[Tuple[int, str]] = []
        self.function_calls = []

    # ---- helpers ------------------------------------------------------
    def _create_node(self, name: str = "Block") -> int:
        nid = self.node_counter
        self.node_counter += 1
        self.nodes[nid] = CFGNode(nid, name)
        return nid

    def _add_edge(self, src: Optional[int], dst: Optional[int], condition: str = "unconditional"):
        if src is not None and dst is not None:
            self.nodes[src].add_edge(dst, condition)

    def _live(self) -> bool:
        return self.current_node_id is not None and not self.nodes[self.current_node_id].is_return

    def _extract_func_calls(self, node):
        if node is None:
            return
        if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID):
            self.function_calls.append(node.name.name)
        for _, child in node.children():
            self._extract_func_calls(child)

    # ---- function ------------------------------------------------------
    def build_from_func(self, func_ast: c_ast.FuncDef) -> Dict[str, Any]:
        self.nodes = {}
        self.node_counter = 0
        self.function_calls = []
        self.labels = {}
        self.pending_gotos = []
        self.switch_stack = []

        params, ptr_vars = self._collect_params(func_ast)

        entry = self._create_node(name=f"Entry: {func_ast.decl.name}")
        self.entry_node_id = entry
        self.current_node_id = entry
        if func_ast.body:
            self.visit(func_ast.body)

        exit_node = self._create_node(name="Exit")
        if self._live():
            self._add_edge(self.current_node_id, exit_node)
        for nid, node in list(self.nodes.items()):
            if node.is_return and nid != exit_node:
                self._add_edge(nid, exit_node, "return")
        for src, label in self.pending_gotos:
            if label in self.labels:
                self._add_edge(src, self.labels[label], "goto")

        # Locals & tracked variables come from the function body declarations
        decl_finder = _DeclFinder()
        decl_finder.visit(func_ast.body)
        local_names = set(decl_finder.locals) | set(params)
        ptr_vars |= decl_finder.ptr_vars

        locals_info = {"tracked_scalars": decl_finder.tracked_scalars, "array_vars": decl_finder.array_vars}
        nodes = {nid: n.to_dict(locals_info) for nid, n in self.nodes.items()}

        # Reachability (for dead-code smell and analysers)
        return {
            "function": func_ast.decl.name,
            "line": int(func_ast.coord.line) if func_ast.coord else -1,
            "entry_id": self.entry_node_id,
            "exit_id": exit_node,
            "nodes": nodes,
            "function_calls": self.function_calls,
            "params": sorted(params),
            "param_order": self._param_order(func_ast),
            "locals": sorted(local_names),
            "ptr_vars": sorted(ptr_vars),
            "tracked_scalars": sorted(decl_finder.tracked_scalars),
        }

    @staticmethod
    def _param_order(func_ast):
        try:
            args = func_ast.decl.type.args
            return [p.name for p in (args.params if args else []) if isinstance(p, c_ast.Decl)]
        except AttributeError:
            return []

    def _collect_params(self, func_ast):
        params, ptr_vars = set(), set()
        try:
            args = func_ast.decl.type.args
            for p in (args.params if args else []):
                if isinstance(p, c_ast.Decl) and p.name:
                    params.add(p.name)
                    if isinstance(p.type, (c_ast.PtrDecl, c_ast.ArrayDecl)):
                        ptr_vars.add(p.name)
        except AttributeError:
            pass
        return params, ptr_vars

    # ---- statements -----------------------------------------------------
    def visit_Compound(self, node):
        for item in (node.block_items or []):
            self.visit(item)

    def visit_If(self, node):
        cond_node = self.current_node_id
        if cond_node is None:
            return
        if node.cond is not None:
            self.nodes[cond_node].cond_expr = node.cond
            self._extract_func_calls(node.cond)

        true_block = self._create_node("If True")
        false_block = self._create_node("If False")
        merge_block = self._create_node("If Merge")

        # A condition that is a compile-time constant makes one branch infeasible: do not follow it.
        truth = _const_value(node.cond, self.consts) if node.cond is not None else None
        take_true = truth is None or truth != 0
        take_false = truth is None or truth == 0

        if take_true:
            self._add_edge(cond_node, true_block, "True")
            self.current_node_id = true_block
            if node.iftrue:
                self.visit(node.iftrue)
            if self._live():
                self._add_edge(self.current_node_id, merge_block)

        if take_false:
            self._add_edge(cond_node, false_block, "False")
            self.current_node_id = false_block
            if node.iffalse:
                self.visit(node.iffalse)
            if self._live():
                self._add_edge(self.current_node_id, merge_block)

        self.current_node_id = merge_block

    def visit_While(self, node):
        cond_blk = self._create_node("While Cond")
        body_blk = self._create_node("While Body")
        exit_blk = self._create_node("While Exit")
        self.loop_break_targets.append(exit_blk)
        self.loop_continue_targets.append(cond_blk)

        self._add_edge(self.current_node_id, cond_blk)
        self.current_node_id = cond_blk
        if node.cond is not None:
            self.nodes[cond_blk].cond_expr = node.cond
            self._extract_func_calls(node.cond)
        truth = _const_value(node.cond, self.consts) if node.cond is not None else None
        if truth is None or truth != 0:
            self._add_edge(cond_blk, body_blk, "True")
        if truth is None or truth == 0:
            self._add_edge(cond_blk, exit_blk, "False")     # `while (1)` only exits through break / return

        self.current_node_id = body_blk
        if node.stmt:
            self.visit(node.stmt)
        if self._live():
            self._add_edge(self.current_node_id, cond_blk, "loop_back")

        self.loop_break_targets.pop()
        self.loop_continue_targets.pop()
        self.current_node_id = exit_blk

    def visit_DoWhile(self, node):
        body_blk = self._create_node("DoWhile Body")
        cond_blk = self._create_node("DoWhile Cond")
        exit_blk = self._create_node("DoWhile Exit")
        self.loop_break_targets.append(exit_blk)
        self.loop_continue_targets.append(cond_blk)

        self._add_edge(self.current_node_id, body_blk)
        self.current_node_id = body_blk
        if node.stmt:
            self.visit(node.stmt)
        if self._live():
            self._add_edge(self.current_node_id, cond_blk)
        if node.cond is not None:
            self.nodes[cond_blk].cond_expr = node.cond
            self._extract_func_calls(node.cond)
        truth = _const_value(node.cond, self.consts) if node.cond is not None else None
        if truth is None or truth != 0:
            self._add_edge(cond_blk, body_blk, "loop_back")
        if truth is None or truth == 0:
            self._add_edge(cond_blk, exit_blk, "False")     # `do { .. } while (0)` runs once and never loops

        self.loop_break_targets.pop()
        self.loop_continue_targets.pop()
        self.current_node_id = exit_blk

    def visit_For(self, node):
        init_blk = self._create_node("For Init")
        cond_blk = self._create_node("For Cond")
        body_blk = self._create_node("For Body")
        next_blk = self._create_node("For Next")
        exit_blk = self._create_node("For Exit")
        self.loop_break_targets.append(exit_blk)
        self.loop_continue_targets.append(next_blk)

        self._add_edge(self.current_node_id, init_blk)
        self.current_node_id = init_blk
        if node.init is not None:
            self.nodes[init_blk].add_stmt(node.init)
            self._extract_func_calls(node.init)

        self._add_edge(init_blk, cond_blk)
        self.current_node_id = cond_blk
        if node.cond is not None:
            self.nodes[cond_blk].cond_expr = node.cond
            self._extract_func_calls(node.cond)
        truth = 1 if node.cond is None else _const_value(node.cond, self.consts)
        if truth is None or truth != 0:
            self._add_edge(cond_blk, body_blk, "True")
        if truth is None or truth == 0:
            self._add_edge(cond_blk, exit_blk, "False")     # `for (;;)` only exits through break / return

        self.current_node_id = body_blk
        if node.stmt:
            self.visit(node.stmt)
        if self._live():
            self._add_edge(self.current_node_id, next_blk)

        self.current_node_id = next_blk
        if node.next is not None:
            self.nodes[next_blk].add_stmt(node.next)
            self._extract_func_calls(node.next)
        self._add_edge(next_blk, cond_blk, "loop_back")

        self.loop_break_targets.pop()
        self.loop_continue_targets.pop()
        self.current_node_id = exit_blk

    def visit_Switch(self, node):
        cond_node = self.current_node_id
        if cond_node is None:
            return
        if node.cond is not None:
            self.nodes[cond_node].cond_expr = node.cond
            self._extract_func_calls(node.cond)
        exit_blk = self._create_node("Switch Exit")
        ctx = {"cond": cond_node, "has_default": False, "exit": exit_blk}
        self.switch_stack.append(ctx)
        self.loop_break_targets.append(exit_blk)

        self.current_node_id = None            # nothing is live until the first case label
        if node.stmt:
            self.visit(node.stmt)
        if self._live():
            self._add_edge(self.current_node_id, exit_blk)

        self.loop_break_targets.pop()
        self.switch_stack.pop()
        if not ctx["has_default"]:
            self._add_edge(cond_node, exit_blk, "no_case")
        self.current_node_id = exit_blk

    def _enter_case(self, node, is_default):
        if not self.switch_stack:
            return
        ctx = self.switch_stack[-1]
        blk = self._create_node("Default" if is_default else "Case")
        self._add_edge(ctx["cond"], blk, "case")
        if self._live():
            self._add_edge(self.current_node_id, blk)     # fall-through
        if is_default:
            ctx["has_default"] = True
        self.current_node_id = blk
        for s in (node.stmts or []):
            self.visit(s)

    def visit_Case(self, node):
        self._enter_case(node, False)

    def visit_Default(self, node):
        self._enter_case(node, True)

    def visit_Label(self, node):
        blk = self._create_node(f"Label {node.name}")
        if self._live():
            self._add_edge(self.current_node_id, blk)
        self.labels[node.name] = blk
        self.current_node_id = blk
        if node.stmt:
            self.visit(node.stmt)

    def visit_Goto(self, node):
        if self.current_node_id is not None:
            self.pending_gotos.append((self.current_node_id, node.name))
            self.current_node_id = None

    def visit_Break(self, node):
        if self.loop_break_targets and self.current_node_id is not None:
            self._add_edge(self.current_node_id, self.loop_break_targets[-1], "break")
            self.current_node_id = None

    def visit_Continue(self, node):
        if self.loop_continue_targets and self.current_node_id is not None:
            self._add_edge(self.current_node_id, self.loop_continue_targets[-1], "continue")
            self.current_node_id = None

    def visit_Return(self, node):
        if self.current_node_id is not None:
            self.nodes[self.current_node_id].add_stmt(node)
            self._extract_func_calls(node)
            self.nodes[self.current_node_id].is_return = True
            self.current_node_id = None

    def visit_Decl(self, node):
        self.generic_visit(node)

    def generic_visit(self, node):
        if self.current_node_id is not None:
            cur = self.nodes[self.current_node_id]
            cur.add_stmt(node)
            self._extract_func_calls(node)
            # exit()/abort() end the program: treat like a return that is not a leak site
            if isinstance(node, c_ast.FuncCall) and isinstance(node.name, c_ast.ID) \
                    and node.name.name in NORETURN_FUNCS:
                cur.is_return = True
                cur.noreturn = True
                self.current_node_id = None


class _DeclFinder(c_ast.NodeVisitor):
    """Collects local declarations of a function body."""

    def __init__(self):
        self.locals = set()
        self.ptr_vars = set()
        self.tracked_scalars = set()   # declared without initializer, plain scalar / pointer, non-static
        self.array_vars = set()        # locals declared as arrays (their name decays to a stack pointer)

    def visit_Decl(self, n):
        if n.name is None:
            return
        if isinstance(n.type, c_ast.FuncDecl):
            return
        self.locals.add(n.name)
        if isinstance(n.type, c_ast.PtrDecl):
            self.ptr_vars.add(n.name)
        if isinstance(n.type, c_ast.ArrayDecl):
            self.array_vars.add(n.name)
        storage = set(n.storage or [])
        plain = isinstance(n.type, (c_ast.PtrDecl,)) or (
            isinstance(n.type, c_ast.TypeDecl) and isinstance(n.type.type, c_ast.IdentifierType))
        if n.init is None and plain and not (storage & {"static", "extern"}):
            self.tracked_scalars.add(n.name)
        if n.init is not None:
            self.visit(n.init)


def build_cfg_for_file(ast, max_functions: int = 2000) -> List[Dict[str, Any]]:
    """Builds a list of CFGs, one for each function in the C file."""
    if ast is None:
        return []
    cfgs: List[Dict[str, Any]] = []
    consts = _file_constants(ast)

    class FuncFinder(c_ast.NodeVisitor):
        def visit_FuncDef(self, node):
            if len(cfgs) >= max_functions:
                return
            cfgs.append(CFGBuilder(consts).build_from_func(node))

    FuncFinder().visit(ast)
    return cfgs
