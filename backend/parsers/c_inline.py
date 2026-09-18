"""
IntelliReview - intra-file interprocedural analysis for C by inlining
=====================================================================
Most real defects are split across functions: a "source" helper returns tainted or undersized memory, a "sink" helper copies,
frees or dereferences its parameter.  Looking at each function alone hides the flaw (and, worse, makes the analysers assume the
worst about every parameter).  This module builds an ANALYSIS COPY of the AST in which calls to functions defined in the same
file are expanded in place:

    x = src(a);         ->   { T p = a; ...callee body...; x = <return expr>; }
    sink(buf);          ->   { T p = buf; ...callee body... }
    fp(x)               ->   the same, when `fp` is a local function pointer bound once to a function defined here
    if (returnsTrue())  ->   if (1)          (callee is `return <compile-time constant>;`)

Callee locals and parameters are renamed (`name_i3`) so they cannot clash with the caller's; line numbers are preserved, so a
finding inside an expanded body is reported at the callee's own line.  An `&x` argument for a pointer parameter also rewrites
`*p` to `x` in the copy, so out-parameters (`void get(char **out)`) are seen through.

Limits (anything else is simply left as a call): recursion, more than one `return` or a `return` that is not the last
statement, labels / goto / static locals / varargs in the callee, callee bodies over MAX_NODES, nesting deeper than MAX_DEPTH.

A `static` function whose every reference was expanded is reported in `ctx_only`: its parameters are then fully described by its
callers, so the "an unknown caller may pass anything" assumption must NOT be applied to it on its own.
"""
import copy
from typing import Dict, List, Optional, Set, Tuple

from pycparser import c_ast

from analyzers.v3_cfg_builder import _const_value, _file_constants

MAX_NODES = 700          # AST nodes of a callee (after its own expansion) that we are willing to copy into a caller
MAX_DEPTH = 4
MAX_CALLER_NODES = 12000

_SKIP = ("coord", "__weakref__")


def _fields(node):
    for slot in type(node).__slots__:
        if slot in _SKIP:
            continue
        v = getattr(node, slot, None)
        if isinstance(v, c_ast.Node):
            yield slot, v, None
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, c_ast.Node):
                    yield slot, x, i


def _walk(node):
    yield node
    for _, ch, _i in _fields(node):
        yield from _walk(ch)


def _walk_values(node):
    """Like _walk, but does not descend into the member name of `a.b` / `a->b` (a field is not a variable)."""
    yield node
    for slot, ch, _i in _fields(node):
        if isinstance(node, c_ast.StructRef) and slot == "field":
            continue
        yield from _walk_values(ch)


def _size(node) -> int:
    return sum(1 for _ in _walk(node))


def _replace(node, fn):
    """Post-order rewrite: every child c is replaced by fn(c) (which may return c itself)."""
    for slot, ch, i in list(_fields(node)):
        _replace(ch, fn)
        new = fn(ch)
        if new is not ch:
            if i is None:
                setattr(node, slot, new)
            else:
                getattr(node, slot)[i] = new
    return node


def _unwrap(e):
    while isinstance(e, c_ast.Cast):
        e = e.expr
    return e


def _set_declname(t, name):
    while t is not None and not isinstance(t, c_ast.TypeDecl):
        t = getattr(t, "type", None)
    if t is not None:
        t.declname = name


def _params(fdef) -> Optional[List[c_ast.Decl]]:
    try:
        args = fdef.decl.type.args
    except AttributeError:
        return None
    if args is None:
        return []
    ps = list(args.params)
    if len(ps) == 1 and isinstance(ps[0], c_ast.Typename) and isinstance(ps[0].type, c_ast.TypeDecl) \
            and isinstance(ps[0].type.type, c_ast.IdentifierType) and ps[0].type.type.names == ["void"]:
        return []
    if any(not isinstance(p, c_ast.Decl) or p.name is None or any(isinstance(n, c_ast.FuncDecl) for n in _walk(p.type))
           for p in ps):
        return None                  # varargs (Ellipsis), unnamed or function-pointer parameters
    return ps


class _Info:
    def __init__(self, params, items, ret):
        self.params, self.items, self.ret = params, items, ret


class Inliner:
    def __init__(self, ast):
        self.orig = ast
        self.ast = copy.deepcopy(ast)
        self.consts = _file_constants(ast)
        self.defs: Dict[str, c_ast.FuncDef] = {}
        for ext in self.ast.ext:
            if isinstance(ext, c_ast.FuncDef) and ext.decl and ext.decl.name:
                self.defs.setdefault(ext.decl.name, ext)
        self.static = {n for n, f in self.defs.items() if "static" in (f.decl.storage or [])}
        self.rec = self._recursive()
        self.refs: Dict[str, int] = {}
        for node in _walk(ast):
            if isinstance(node, c_ast.ID) and node.name in self.defs:
                self.refs[node.name] = self.refs.get(node.name, 0) + 1
        self.consumed: Dict[str, int] = {}
        self.info: Dict[str, Optional[_Info]] = {}
        self.const_funcs = self._const_functions()
        self._uid = 0
        self._budget = 0

    # ---------------------------------------------------------------------------------- analysis of the call graph
    def _callees(self, f) -> Set[str]:
        return {n.name for n in _walk(f.body) if isinstance(n, c_ast.ID) and n.name in self.defs}

    def _recursive(self) -> Set[str]:
        graph = {n: self._callees(f) for n, f in self.defs.items()}
        rec: Set[str] = set()

        def reach(start):
            seen, stack = set(), list(graph[start])
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(graph.get(x, ()))
            return seen
        for n in graph:
            if n in reach(n):
                rec.add(n)
        return rec

    def _const_functions(self) -> Dict[str, int]:
        out = {}
        for n, f in self.defs.items():
            if _params(f) == [] and f.body and f.body.block_items and len(f.body.block_items) == 1 \
                    and isinstance(f.body.block_items[0], c_ast.Return) and f.body.block_items[0].expr is not None:
                v = _const_value(f.body.block_items[0].expr, self.consts)
                if v is not None:
                    out[n] = v
        return out

    # ---------------------------------------------------------------------------------- per-function driver
    def run(self):
        order = self._topological()
        for name in order:
            f = self.defs[name]
            self._budget = MAX_CALLER_NODES
            self._transform_body(f)
        ctx_only = {n for n in self.static if self.refs.get(n, 0) > 0 and self.consumed.get(n, 0) >= self.refs.get(n, 0)
                    and n not in self.rec and n != "main"}
        return self.ast, ctx_only

    def _topological(self) -> List[str]:
        graph = {n: self._callees(f) - {n} for n, f in self.defs.items()}
        seen, out = set(), []

        def dfs(n, stack):
            if n in seen or n in stack:
                return
            stack.add(n)
            for m in graph.get(n, ()):
                dfs(m, stack)
            stack.discard(n)
            seen.add(n)
            out.append(n)
        for n in self.defs:
            dfs(n, set())
        return out

    def _transform_body(self, f):
        # function pointers bound exactly once to a function of this file
        bindings: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        for n in _walk(f.body):
            if isinstance(n, c_ast.Decl) and n.name and n.init is not None and isinstance(_unwrap(n.init), c_ast.ID) \
                    and _unwrap(n.init).name in self.defs and _is_fnptr(n.type):
                bindings[n.name] = _unwrap(n.init).name
                counts[n.name] = counts.get(n.name, 0) + 1
            elif isinstance(n, c_ast.Assignment) and n.op == "=" and isinstance(_unwrap(n.lvalue), c_ast.ID):
                counts[_unwrap(n.lvalue).name] = counts.get(_unwrap(n.lvalue).name, 0) + 1
                if isinstance(_unwrap(n.rvalue), c_ast.ID) and _unwrap(n.rvalue).name in self.defs:
                    bindings[_unwrap(n.lvalue).name] = _unwrap(n.rvalue).name
        self._fnptr = {k: v for k, v in bindings.items() if counts.get(k) == 1}
        self._self = f.decl.name
        f.body = self._xstmt_one(f.body)
        self._fold(f)
        self.info.pop(f.decl.name, None)          # its own body changed: recompute lazily for later callers

    # ---------------------------------------------------------------------------------- statement rewriting
    @staticmethod
    def _wrap(stmts):
        return stmts[0] if len(stmts) == 1 else c_ast.Compound(block_items=stmts)

    def _xstmt_one(self, s):
        return self._wrap(self._xstmt(s))

    def _xstmt(self, s) -> list:
        if s is None:
            return []
        if isinstance(s, c_ast.Compound):
            new = []
            for it in s.block_items or []:
                new.extend(self._xstmt(it))
            s.block_items = new
            return [s]
        if isinstance(s, c_ast.If):
            s.iftrue = self._xstmt_one(s.iftrue) if s.iftrue is not None else s.iftrue
            if s.iffalse is not None:
                s.iffalse = self._xstmt_one(s.iffalse)
            return [s]
        if isinstance(s, (c_ast.While, c_ast.DoWhile, c_ast.For, c_ast.Switch)):
            s.stmt = self._xstmt_one(s.stmt)
            return [s]
        if isinstance(s, (c_ast.Case, c_ast.Default)):
            new = []
            for it in s.stmts or []:
                new.extend(self._xstmt(it))
            s.stmts = new
            return [s]
        if isinstance(s, c_ast.Label):
            s.stmt = self._xstmt_one(s.stmt)
            return [s]
        call, result = None, None
        if isinstance(s, c_ast.FuncCall):
            call = s
        elif isinstance(s, c_ast.Assignment) and s.op == "=" and isinstance(_unwrap(s.rvalue), c_ast.FuncCall):
            call, result = _unwrap(s.rvalue), ("assign", s)
        elif isinstance(s, c_ast.Decl) and s.init is not None and isinstance(_unwrap(s.init), c_ast.FuncCall) \
                and not isinstance(s.type, (c_ast.ArrayDecl, c_ast.FuncDecl)):
            call, result = _unwrap(s.init), ("decl", s)
        if call is None:
            return [s]
        expanded = self._expand(call, result)
        return expanded if expanded is not None else [s]

    # ---------------------------------------------------------------------------------- expansion of one call
    def _callee_name(self, call) -> Optional[str]:
        if not isinstance(call.name, c_ast.ID):
            return None
        n = call.name.name
        if n in self.defs:
            return n
        return self._fnptr.get(n)

    def _info(self, name) -> Optional[_Info]:
        if name in self.info:
            return self.info[name]
        self.info[name] = None                      # blocks accidental re-entry
        f = self.defs[name]
        ps = _params(f)
        body = f.body.block_items if f.body and f.body.block_items else []
        ok = ps is not None and name not in self.rec and not f.decl.storage.count("extern")
        ret = None
        items = list(body)
        if ok and items and isinstance(items[-1], c_ast.Return):
            ret = items[-1].expr
            items = items[:-1]
        if ok:
            for it in items:
                for n in _walk(it):
                    if isinstance(n, (c_ast.Return, c_ast.Goto, c_ast.Label)) or \
                            (isinstance(n, c_ast.Decl) and "static" in (n.storage or [])):
                        ok = False
                        break
                if not ok:
                    break
        if ok and sum(_size(i) for i in items) + (_size(ret) if ret is not None else 0) > MAX_NODES:
            ok = False
        self.info[name] = _Info(ps, items, ret) if ok else None
        return self.info[name]

    def _expand(self, call, result):
        name = self._callee_name(call)
        if name is None or name == self._self:
            return None
        if name in self.const_funcs and not (call.args and call.args.exprs):
            return None                             # handled by constant folding
        info = self._info(name)
        args = call.args.exprs if call.args else []
        if info is None or len(args) != len(info.params):
            return None
        size = sum(_size(i) for i in info.items) + 8 * len(args)
        if size > self._budget:
            return None
        self._budget -= size
        self._uid += 1
        suffix = f"_i{self._uid}"
        items = [copy.deepcopy(i) for i in info.items]
        ret = copy.deepcopy(info.ret) if info.ret is not None else None
        rename: Dict[str, str] = {p.name: p.name + suffix for p in info.params}
        for it in items:
            for n in _walk(it):
                if isinstance(n, c_ast.Decl) and n.name:
                    rename.setdefault(n.name, n.name + suffix)

        def ren(n):
            if isinstance(n, c_ast.ID) and n.name in rename:
                n.name = rename[n.name]
            elif isinstance(n, c_ast.Decl) and n.name in rename:
                n.name = rename[n.name]
                _set_declname(n.type, n.name)
            return n

        holder = c_ast.Compound(block_items=items + ([c_ast.Return(expr=ret)] if ret is not None else []))
        for n in _walk_values(holder):
            ren(n)
        # `*p` / `p[0]` for a pointer parameter bound to `&x` is `x`
        subst: Dict[str, c_ast.Node] = {}
        binds: List[c_ast.Node] = []
        for p, a in zip(info.params, args):
            t = copy.deepcopy(p.type)
            if isinstance(t, c_ast.ArrayDecl):
                t = c_ast.PtrDecl(quals=[], type=t.type)
            new = rename[p.name]
            _set_declname(t, new)
            binds.append(c_ast.Decl(name=new, quals=list(p.quals or []), storage=[], funcspec=[], align=None, type=t,
                                    init=copy.deepcopy(a), bitsize=None, coord=p.coord))
            ua = _unwrap(a)
            if isinstance(t, c_ast.PtrDecl) and isinstance(ua, c_ast.UnaryOp) and ua.op == "&" \
                    and isinstance(_unwrap(ua.expr), c_ast.ID):
                subst[new] = _unwrap(ua.expr)
        if subst:
            for n in _walk(holder):          # `T **p2 = (T**)p;` makes p2 another name for `*p`
                if isinstance(n, c_ast.Decl) and n.name and isinstance(n.type, c_ast.PtrDecl) and n.init is not None                         and isinstance(_unwrap(n.init), c_ast.ID) and _unwrap(n.init).name in subst:
                    subst[n.name] = subst[_unwrap(n.init).name]
            def sub(n):
                if isinstance(n, c_ast.UnaryOp) and n.op == "*" and isinstance(_unwrap(n.expr), c_ast.ID) \
                        and _unwrap(n.expr).name in subst:
                    return copy.deepcopy(subst[_unwrap(n.expr).name])
                if isinstance(n, c_ast.ArrayRef) and isinstance(_unwrap(n.name), c_ast.ID) \
                        and _unwrap(n.name).name in subst and isinstance(n.subscript, c_ast.Constant) \
                        and n.subscript.value == "0":
                    return copy.deepcopy(subst[_unwrap(n.name).name])
                return n
            _replace(holder, sub)
        items = holder.block_items
        ret = None
        if info.ret is not None:
            ret = items.pop().expr
        self.consumed[name] = self.consumed.get(name, 0) + 1
        # count the callee's own references that were consumed when *it* was transformed
        block = list(binds) + items
        if result is None:
            if ret is not None:
                pass                                # value unused
            return [c_ast.Compound(block_items=block)]
        kind, node = result
        if ret is None:
            return None
        if kind == "assign":
            block.append(c_ast.Assignment(op="=", lvalue=node.lvalue, rvalue=ret, coord=node.coord))
            return [c_ast.Compound(block_items=block)]
        decl = copy.deepcopy(node)
        decl.init = None
        block.append(c_ast.Assignment(op="=", lvalue=c_ast.ID(name=node.name, coord=node.coord), rvalue=ret, coord=node.coord))
        return [decl, c_ast.Compound(block_items=block)]

    # ---------------------------------------------------------------------------------- constant folding
    def _fold(self, f):
        if not self.const_funcs:
            return

        def fold(n):
            if isinstance(n, c_ast.FuncCall) and isinstance(n.name, c_ast.ID) and n.name.name in self.const_funcs \
                    and not (n.args and n.args.exprs):
                self.consumed[n.name.name] = self.consumed.get(n.name.name, 0) + 1
                v = self.const_funcs[n.name.name]
                return c_ast.Constant(type="int", value=str(v), coord=n.coord) if v >= 0 else \
                    c_ast.UnaryOp(op="-", expr=c_ast.Constant(type="int", value=str(-v), coord=n.coord), coord=n.coord)
            return n
        _replace(f.body, fold)


def _is_fnptr(t) -> bool:
    return isinstance(t, c_ast.PtrDecl) and isinstance(t.type, c_ast.FuncDecl)


def inline_file(ast) -> Tuple[object, Set[str]]:
    """(analysis AST, names of static functions that are analysed only through their callers)."""
    if ast is None or not getattr(ast, "ext", None):
        return ast, set()
    try:
        return Inliner(ast).run()
    except RecursionError:
        return ast, set()
