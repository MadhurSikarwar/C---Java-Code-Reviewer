"""
IntelliReview - Java taint interpreter (javalang AST)
=====================================================
A small abstract interpreter over the parse tree.  It executes every entry method statement by statement and keeps, for
each variable, an abstract VALUE:

    danger     the sink categories this value is still dangerous for ("sql", "cmd", "path", "xss", "ldap", "xpath", "trust");
               a source produces all of them, a sanitiser removes the ones it neutralises (ESAPI.encodeForHTML -> "xss")
    assumed    danger that only comes from "a public method's parameters are untrusted", not from a proven source
    const      the compile-time constant (string / number / bool) when there is one, so `if (7*42 - num > 200)`,
               `switch (guess.charAt(1))` and `getProperty("alg", "AES/GCM/NoPadding")` are decided instead of guessed
    concat     built by string concatenation / format / append
    obj        a tracked container: list, map, array, StringBuilder, cookie, writer

Branches on constants are folded; on unknowns both sides run on copies and are joined.  Methods and constructors of the same
file are executed at their call sites with the actual argument values (a helper `bad_source()` that returns the request
parameter makes its caller's variable tainted; `Test.doSomething(param)` that drops the value makes it clean), including
classes nested in the file and several top-level classes glued into one source.  A private method that is called from
somewhere in the file is only analysed through its callers; everything else is an entry point.

Anything the interpreter cannot follow yields an UNKNOWN value; it never invents danger from ignorance, except that a method
call on a tainted receiver, or with tainted arguments, returns a tainted value (the conservative default).
"""
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    import javalang
    from javalang import tree as T
except ImportError:  # pragma: no cover
    javalang = None

ALL = frozenset(("sql", "cmd", "path", "xss", "ldap", "xpath", "trust"))
NONE: frozenset = frozenset()
NULL = object()          # the Java `null` constant
MAX_DEPTH = 8
MAX_STEPS = 60000

SOURCE_METHODS = {"getParameter", "getHeader", "getQueryString", "getRequestURI", "getRequestURL", "getPathInfo",
                  "getInputStream", "getReader", "readLine", "nextLine", "readUTF", "getRemoteUser", "getPathTranslated",
                  "getServletPath", "getContextPath", "getAuthType", "readObject", "getRemoteAddr", "getRemoteHost"}
SOURCE_COLLECTIONS = {"getParameterValues", "getParameterNames", "getHeaderNames", "getHeaders", "getParameterMap",
                      "getCookies", "getAttributeNames", "getParts"}
NUMBER_PARSERS = {"parseInt", "parseLong", "parseDouble", "parseFloat", "parseShort", "parseByte", "parseBoolean",
                  "valueOf_num"}
XSS_ENCODERS = {"encodeForHTML", "encodeForHTMLAttribute", "encodeForJavaScript", "encodeForCSS", "encodeForURL",
                "encodeForXML", "encodeForXMLAttribute", "forHtml", "forHtmlContent", "forHtmlAttribute", "forHtmlUnquotedAttribute",
                "forJavaScript", "forJavaScriptAttribute", "forJavaScriptBlock", "forUriComponent", "forXml", "forXmlContent",
                "forXmlAttribute", "forCssString", "escapeHtml", "escapeHtml4", "escapeHtml3", "escapeXml", "escapeXml10",
                "escapeXml11", "escapeJavaScript", "escapeEcmaScript", "htmlEscape", "htmlEscapeDecimal", "htmlEscapeHex",
                "sanitize", "escapeJson"}
OTHER_ENCODERS = {"encodeForSQL": "sql", "encodeForLDAP": "ldap", "encodeForDN": "ldap", "encodeForXPath": "xpath",
                  "encodeForOS": "cmd", "encodeForXPathAttribute": "xpath"}
SQL_SINKS = {"executeQuery", "executeUpdate", "executeLargeUpdate", "addBatch", "prepareStatement", "prepareCall",
             "createQuery", "createNativeQuery", "createSQLQuery", "queryForList", "queryForObject", "queryForRowSet",
             "queryForMap", "queryForLong", "queryForInt", "batchUpdate", "nativeQuery"}
SQL_AMBIGUOUS = {"execute", "query", "update"}
SQL_HINT = re.compile(r"(?i)statement|stmt|jdbc|template|connection|conn|session|query|sql|entitymanager|\bdb\b")
STR_TYPES = {"String", "Object", "CharSequence", "StringBuilder", "StringBuffer"}
WEAK_HASH = re.compile(r"(?i)^(MD2|MD4|MD5|SHA-?1)$")
WEAK_CIPHER = re.compile(r"(?i)^(DES|DESede|RC2|RC4|ARCFOUR|Blowfish|AES)(?:/.*)?$|/ECB(?:/|$)")


class Obj:
    """A tracked mutable object."""
    __slots__ = ("kind", "items", "entries", "any", "content", "flags")

    def __init__(self, kind, items=None, entries=None, any_=None, content=None, flags=None):
        self.kind = kind
        self.items: Optional[list] = items
        self.entries: Optional[dict] = entries
        self.any: Optional["V"] = any_
        self.content: Optional["V"] = content
        self.flags: dict = flags or {}

    def clone(self, memo) -> "Obj":
        if id(self) in memo:
            return memo[id(self)]
        o = Obj(self.kind, None, None, None, None, dict(self.flags))
        memo[id(self)] = o
        o.items = [x.clone(memo) for x in self.items] if self.items is not None else None
        o.entries = {k: x.clone(memo) for k, x in self.entries.items()} if self.entries is not None else None
        o.any = self.any.clone(memo) if self.any is not None else None
        o.content = self.content.clone(memo) if self.content is not None else None
        return o

    def union(self) -> "V":
        v = self.any or V()
        for x in (self.items or []):
            v = join(v, x, keep_const=False)
        for x in (self.entries or {}).values():
            v = join(v, x, keep_const=False)
        if self.content is not None:
            v = join(v, self.content, keep_const=False)
        return v


class V:
    __slots__ = ("danger", "assumed", "const", "concat", "unk", "obj", "cls")

    def __init__(self, danger=NONE, assumed=NONE, const=None, concat=False, unk=False, obj=None, cls=None):
        self.danger = danger
        self.assumed = assumed
        self.const = const
        self.concat = concat
        self.unk = unk
        self.obj = obj
        self.cls = cls          # class name of an object created from a class of this file

    def clone(self, memo) -> "V":
        return V(self.danger, self.assumed, self.const, self.concat, self.unk,
                 self.obj.clone(memo) if self.obj is not None else None, self.cls)

    def tainted(self, cat) -> bool:
        return cat in self.danger

    def assume(self, cat) -> bool:
        return cat in self.assumed


def join(a: "V", b: "V", keep_const=True) -> "V":
    if a is None:
        return b
    if b is None:
        return a
    obj = a.obj if a.obj is b.obj else (_merge_obj(a.obj, b.obj))
    const = a.const if (keep_const and a.const is not None and _same(a.const, b.const)) else None
    return V(a.danger | b.danger, a.assumed | b.assumed, const, a.concat or b.concat, a.unk or b.unk, obj, a.cls or b.cls)


def _same(x, y) -> bool:
    return x is y or (x is not NULL and y is not NULL and type(x) == type(y) and x == y)


def _merge_obj(a: Optional[Obj], b: Optional[Obj]) -> Optional[Obj]:
    if a is None or b is None:
        return a or b
    if a.kind != b.kind:
        return Obj("mixed", any_=join(a.union(), b.union(), keep_const=False))
    o = Obj(a.kind, flags={k: (a.flags.get(k) and b.flags.get(k)) for k in set(a.flags) | set(b.flags)})
    if a.items is not None and b.items is not None and len(a.items) == len(b.items):
        o.items = [join(x, y) for x, y in zip(a.items, b.items)]
    elif a.items is not None or b.items is not None:
        o.any = join(a.union(), b.union(), keep_const=False)
    if a.entries is not None and b.entries is not None:
        o.entries = {k: join(a.entries.get(k, V(const=NULL)), b.entries.get(k, V(const=NULL))) for k in set(a.entries) | set(b.entries)}
        if a.any or b.any:
            o.any = join(a.any, b.any, keep_const=False)
    if a.content is not None or b.content is not None:
        o.content = join(a.content, b.content) if (a.content and b.content) else (a.content or b.content)
    return o


class Env:
    __slots__ = ("vars", "types", "fields", "dead", "ret")

    def __init__(self, fields=None):
        self.vars: Dict[str, V] = {}
        self.types: Dict[str, str] = {}
        self.fields: Dict[str, V] = fields if fields is not None else {}
        self.dead = False
        self.ret: Optional[V] = None

    def clone(self) -> "Env":
        memo: dict = {}
        e = Env({k: v.clone(memo) for k, v in self.fields.items()})
        e.vars = {k: v.clone(memo) for k, v in self.vars.items()}
        e.types = dict(self.types)
        e.dead = self.dead
        e.ret = self.ret.clone(memo) if self.ret is not None else None
        return e

    def absorb(self, other: "Env"):
        """Take over the state of `other` (used after a branch that was cloned)."""
        self.vars, self.types, self.fields, self.dead, self.ret = other.vars, other.types, other.fields, other.dead, other.ret


def join_env(a: Env, b: Env) -> Env:
    if a.dead and not b.dead:
        return b
    if b.dead and not a.dead:
        return a
    out = Env({k: join(a.fields.get(k), b.fields.get(k)) for k in set(a.fields) | set(b.fields)})
    for k in set(a.vars) | set(b.vars):
        va, vb = a.vars.get(k), b.vars.get(k)
        out.vars[k] = join(va, vb) if va is not None and vb is not None else (va or vb)
    out.types = {**a.types, **b.types}
    out.dead = a.dead and b.dead
    out.ret = join(a.ret, b.ret, keep_const=False) if a.ret and b.ret else (a.ret or b.ret)
    return out


def _lit(node) -> V:
    val = node.value
    ops = node.prefix_operators or []
    if val == "null":
        return V(const=NULL)
    if val in ("true", "false"):
        v = val == "true"
        return V(const=(not v) if "!" in ops else v)
    if val.startswith('"'):
        return V(const=_unescape(val[1:-1]))
    if val.startswith("'"):
        return V(const=_unescape(val[1:-1]))
    try:
        txt = val.rstrip("lLfFdD")
        num = int(txt, 0) if not re.search(r"[.eE]", txt) or txt.lower().startswith("0x") else float(txt)
        return V(const=-num if "-" in ops else num)
    except ValueError:
        return V(unk=True)


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "r": "\r"}.get(m.group(1), m.group(1)), s)


def _is_str(x) -> bool:
    return isinstance(x, str)


class JavaTaint:
    def __init__(self, tree, source_text: str = ""):
        self.tree = tree
        self.source = source_text
        self.findings: List[dict] = []
        self._seen: Set[Tuple] = set()
        self.methods: Dict[str, List[Tuple[object, object]]] = {}    # name -> [(decl, class)]
        self.classes: Dict[str, object] = {}
        self.static_consts: Dict[str, V] = {}
        self.field_inits: Dict[str, List[Tuple[str, object, str]]] = {}   # class -> [(name, initializer, type)]
        self.stack: List[int] = []
        self.steps = 0
        self.line = 0
        self.webish = bool(re.search(r"javax\.servlet|jakarta\.servlet|java\.security|javax\.crypto|HttpServlet", source_text))
        self.called: Set[str] = set()
        self._collect(tree.types)

    # ------------------------------------------------------------------ collection
    def _collect(self, types):
        for t in types:
            if isinstance(t, (T.ClassDeclaration, T.InterfaceDeclaration, T.EnumDeclaration)):
                self._collect_class(t)
        for name, lst in self.methods.items():
            for decl, _cls in lst:
                for n in _walk(decl):
                    if isinstance(n, T.MethodInvocation):
                        self.called.add(n.member)

    def _collect_class(self, cls):
        self.classes[cls.name] = cls
        inits = []
        for m in (cls.body or []):
            if isinstance(m, T.MethodDeclaration) or isinstance(m, T.ConstructorDeclaration):
                self.methods.setdefault(m.name, []).append((m, cls))
            elif isinstance(m, T.FieldDeclaration):
                tname = getattr(m.type, "name", "")
                for d in m.declarators:
                    inits.append((d.name, d.initializer, tname))
                    if "static" in (m.modifiers or ()) and "final" in (m.modifiers or ()) and d.initializer is not None:
                        v = self._const_expr(d.initializer)
                        if v is not None:
                            self.static_consts[cls.name + "." + d.name] = v
                            self.static_consts.setdefault(d.name, v)
            elif isinstance(m, (T.ClassDeclaration, T.InterfaceDeclaration, T.EnumDeclaration)):
                self._collect_class(m)
        self.field_inits[cls.name] = inits

    def _const_expr(self, node) -> Optional[V]:
        try:
            v = self._ev(node, Env())
        except Exception:  # noqa: BLE001
            return None
        return v if v.const is not None else None

    # ------------------------------------------------------------------ driver
    def run(self) -> List[dict]:
        entries = []
        for name, lst in self.methods.items():
            for decl, cls in lst:
                if not isinstance(decl, T.MethodDeclaration) or decl.body is None:
                    continue
                private = "private" in (decl.modifiers or ())
                if private and name in self.called:
                    continue                                    # analysed through its callers
                entries.append((decl, cls))
        for decl, cls in entries:
            try:
                self._run_entry(decl, cls)
            except RecursionError:
                continue
        return self.findings

    def _fresh_fields(self, cls) -> Dict[str, V]:
        fields: Dict[str, V] = {}
        env = Env(fields)
        for c in self.classes.values():
            for name, init, tname in self.field_inits.get(c.name, []):
                if init is not None:
                    try:
                        fields[name] = self._ev(init, env)
                    except Exception:  # noqa: BLE001
                        fields[name] = V(unk=True)
                else:
                    fields[name] = V(const=NULL) if tname not in ("int", "long", "short", "byte") else V(const=0)
        return fields

    def _run_entry(self, decl, cls):
        env = Env(self._fresh_fields(cls))
        for p in decl.parameters:
            env.vars[p.name] = self._param_value(p, decl)
            env.types[p.name] = self._tname(p.type)
        self.steps = 0
        self._exec_block(decl.body, env)

    def _tname(self, t) -> str:
        if t is None:
            return ""
        n = getattr(t, "name", "") or ""
        if getattr(t, "dimensions", None):
            n += "[]" * len(t.dimensions)
        return n

    def _param_value(self, p, decl) -> V:
        tn = self._tname(p.type)
        private = "private" in (decl.modifiers or ())
        base = tn.replace("[]", "")
        if base.endswith("Response"):
            return V(obj=Obj("response"))
        if private:
            return V(unk=True)
        if base in STR_TYPES or base.endswith("Request") or base in ("Cookie", "Map", "List", "Object"):
            v = V(assumed=ALL, unk=True)
            if tn.endswith("[]") or base in ("Map", "List"):
                v.obj = Obj("array", any_=V(assumed=ALL, unk=True))
            if base.endswith("Request"):
                v = V(danger=ALL, unk=True)
            return v
        return V(unk=True)

    # ------------------------------------------------------------------ findings
    def _add(self, kind, sev, msg, fix, line=None):
        line = line or self.line
        key = (kind, line)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append({"type": kind, "severity": sev, "line": line, "function": kind.lower(), "message": msg,
                              "suggestion": fix})

    # ------------------------------------------------------------------ statements
    def _tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise RecursionError

    def _exec_block(self, stmts, env: Env):
        if stmts is None:
            return
        if not isinstance(stmts, list):
            stmts = [stmts]
        for s in stmts:
            if env.dead:
                return
            self._exec(s, env)

    def _cond(self, node, env: Env):
        """(value, truth) - truth is True/False when the condition is a compile-time constant."""
        v = self._ev(node, env)
        if isinstance(v.const, bool):
            return v, v.const
        return v, None

    def _exec(self, s, env: Env):
        self._tick()
        pos = getattr(s, "position", None)
        if pos is not None:
            self.line = pos.line
        if isinstance(s, T.LocalVariableDeclaration) or isinstance(s, T.VariableDeclaration):
            tname = self._tname(s.type)
            for d in s.declarators:
                env.types[d.name] = tname + ("[]" * len(d.dimensions or []))
                if d.initializer is None:
                    env.vars[d.name] = V(const=NULL) if tname not in ("int", "long", "short", "byte", "char", "boolean", "double") else V()
                else:
                    env.vars[d.name] = self._ev(d.initializer, env, tname)
        elif isinstance(s, T.StatementExpression):
            self._ev(s.expression, env)
        elif isinstance(s, T.IfStatement):
            _v, truth = self._cond(s.condition, env)
            if truth is True:
                self._exec_block(s.then_statement, env)
            elif truth is False:
                self._exec_block(s.else_statement, env)
            else:
                a, b = env.clone(), env.clone()
                self._exec_block(s.then_statement, a)
                self._exec_block(s.else_statement, b)
                env.absorb(join_env(a, b))
        elif isinstance(s, T.BlockStatement):
            self._exec_block(s.statements, env)
        elif isinstance(s, (T.WhileStatement, T.DoStatement)):
            _v, truth = self._cond(s.condition, env)
            if isinstance(s, T.WhileStatement) and truth is False:
                return
            body = env.clone()
            self._exec_block(s.body, body)
            body.dead = False
            env.absorb(join_env(env, body) if truth is not True else body)
        elif isinstance(s, T.ForStatement):
            self._for(s, env)
        elif isinstance(s, T.SwitchStatement):
            self._switch(s, env)
        elif isinstance(s, T.TryStatement):
            for r in (s.resources or []):
                env.vars[r.name] = self._ev(r.value, env) if getattr(r, "value", None) is not None else V(unk=True)
            body = env.clone()
            self._exec_block(s.block, body)
            merged = body
            for c in (s.catches or []):
                cenv = env.clone()
                for p in (c.parameter.types or []):
                    pass
                cenv.vars[c.parameter.name] = V(unk=True)
                self._exec_block(c.block, cenv)
                merged = join_env(merged, cenv)
            merged.dead = False
            env.absorb(merged)
            if s.finally_block:
                self._exec_block(s.finally_block, env)
        elif isinstance(s, T.ReturnStatement):
            v = self._ev(s.expression, env) if s.expression is not None else V()
            env.ret = join(env.ret, v, keep_const=False) if env.ret is not None else v
            env.dead = True
        elif isinstance(s, (T.ThrowStatement, T.BreakStatement, T.ContinueStatement)):
            env.dead = True
        elif isinstance(s, T.SynchronizedStatement):
            self._exec_block(s.block, env)
        # class declarations, asserts, empty statements: nothing to do

    def _for(self, s, env: Env):
        ctl = s.control
        loop = env.clone()
        if isinstance(ctl, T.EnhancedForControl):
            it = self._ev(ctl.iterable, loop)
            elem = it.obj.union() if it.obj is not None else V(danger=it.danger, assumed=it.assumed, unk=True)
            for d in ctl.var.declarators:
                loop.vars[d.name] = V(elem.danger, elem.assumed, None, False, True, elem.obj if elem.obj else None)
                loop.types[d.name] = self._tname(ctl.var.type)
        else:
            init = ctl.init
            if isinstance(init, (T.VariableDeclaration, T.LocalVariableDeclaration)):
                self._exec(init, loop)
            elif isinstance(init, list):
                for e in init:
                    self._ev(e, loop)
            if ctl.condition is not None:
                v = self._ev(ctl.condition, loop)
                if v.const is False:
                    return
            # loop counters are not constants after the first iteration
            for n in _walk(s.body):
                if isinstance(n, T.Assignment) and isinstance(n.expressionl, T.MemberReference):
                    loop.vars[n.expressionl.member] = V(unk=True)
            for e in (ctl.update or []):
                if isinstance(e, T.MemberReference):
                    loop.vars[e.member] = V(unk=True)
        self._exec_block(s.body, loop)
        loop.dead = False
        # values assigned inside the loop may or may not have happened
        env.absorb(join_env(env, loop))

    def _switch(self, s, env: Env):
        sel = self._ev(s.expression, env)
        cases = s.cases or []
        chosen = None
        if sel.const is not None:
            for i, c in enumerate(cases):
                for label in (c.case or []):
                    lv = self._ev(label, env)
                    if lv.const is not None and _same(_coerce(lv.const), _coerce(sel.const)):
                        chosen = i
                        break
                if chosen is not None:
                    break
            if chosen is None:
                chosen = next((i for i, c in enumerate(cases) if not c.case), None)
            if chosen is None:
                return
            for c in cases[chosen:]:
                self._exec_block(c.statements, env)
                if env.dead or any(isinstance(x, T.BreakStatement) for x in (c.statements or [])):
                    break
            env.dead = env.dead and any(isinstance(x, T.ReturnStatement) for c in cases[chosen:] for x in (c.statements or []))
            return
        outs = [env.clone()]
        for i, c in enumerate(cases):
            e = env.clone()
            for c2 in cases[i:]:
                self._exec_block(c2.statements, e)
                if e.dead or any(isinstance(x, T.BreakStatement) for x in (c2.statements or [])):
                    break
            e.dead = False
            outs.append(e)
        merged = outs[0]
        for e in outs[1:]:
            merged = join_env(merged, e)
        env.absorb(merged)

    # ------------------------------------------------------------------ expressions
    def _ev(self, node, env: Env, decl_type: str = "") -> V:
        self._tick()
        if node is None:
            return V()
        if isinstance(node, list):
            return V()
        if isinstance(node, T.Literal):
            v = _lit(node)
            return self._selectors(v, node, env)
        if isinstance(node, T.MemberReference):
            return self._member(node, env)
        if isinstance(node, T.MethodInvocation):
            return self._invoke(node, env)
        if isinstance(node, T.BinaryOperation):
            return self._binary(node, env)
        if isinstance(node, T.TernaryExpression):
            _v, truth = self._cond(node.condition, env)
            if truth is True:
                return self._ev(node.if_true, env)
            if truth is False:
                return self._ev(node.if_false, env)
            return join(self._ev(node.if_true, env), self._ev(node.if_false, env))
        if isinstance(node, T.Assignment):
            return self._assign(node, env)
        if isinstance(node, T.ClassCreator):
            return self._selectors(self._create(node, env), node, env)
        if isinstance(node, T.ArrayCreator):
            if node.initializer is not None:
                return self._ev(node.initializer, env)
            return V(obj=Obj("array", items=None, any_=V(const=NULL)))
        if isinstance(node, T.ArrayInitializer):
            return V(obj=Obj("array", items=[self._ev(x, env) for x in node.initializers]))
        if isinstance(node, T.Cast):
            return self._selectors(self._ev(node.expression, env), node, env)
        if isinstance(node, T.This):
            return self._selectors(V(unk=True), node, env)
        return V(unk=True)

    def _selectors(self, base: V, node, env: Env, name: str = "") -> V:
        cur = base
        for sel in (getattr(node, "selectors", None) or []):
            if isinstance(sel, T.MethodInvocation):
                args = [self._ev(a, env) for a in sel.arguments]
                cur = self._call(cur, None, sel.member, args, sel, env, None, name)
                name = sel.member
            elif isinstance(sel, T.ArraySelector):
                idx = self._ev(sel.index, env)
                cur = self._index(cur, idx)
            elif isinstance(sel, T.MemberReference):
                cur = V(cur.danger, cur.assumed, None, False, True)
            else:
                cur = V(cur.danger, cur.assumed, None, False, True)
        return cur

    def _index(self, arr: V, idx: V) -> V:
        o = arr.obj
        if o is not None and o.kind == "array":
            if o.items is not None and isinstance(idx.const, int) and 0 <= idx.const < len(o.items):
                return o.items[idx.const]
            return o.union() if (o.items or o.any) else V(danger=arr.danger, assumed=arr.assumed, unk=True)
        return V(danger=arr.danger, assumed=arr.assumed, unk=True)

    def _member(self, node, env: Env) -> V:
        ops = node.prefix_operators or []
        q, m = node.qualifier or "", node.member
        v = None
        if not q:
            if m in env.vars:
                v = env.vars[m]
            elif m in env.fields:
                v = env.fields[m]
            elif m in self.static_consts:
                v = self.static_consts[m]
            else:
                v = V(unk=True)
        else:
            first = q.split(".")[0]
            if first in env.vars:
                r = env.vars[first]
                v = V(r.danger, r.assumed, None, False, True) if m != "length" else V(unk=True)
                if m == "length" and r.obj is not None and r.obj.items is not None:
                    v = V(const=len(r.obj.items))
            elif first == "this" and m in env.fields:
                v = env.fields[m]
            elif q + "." + m in self.static_consts:
                v = self.static_consts[q + "." + m]
            elif q.split(".")[-1] + "." + m in self.static_consts:
                v = self.static_consts[q.split(".")[-1] + "." + m]
            elif q == "System" and m == "in":
                v = V(danger=ALL, unk=True)
            else:
                v = V(unk=True)
        v = self._selectors(v, node, env)
        if "!" in ops and isinstance(v.const, bool):
            v = V(const=not v.const)
        elif "-" in ops and isinstance(v.const, (int, float)) and not isinstance(v.const, bool):
            v = V(const=-v.const)
        if ("++" in ops or "--" in ops or node.postfix_operators) and not q and m in env.vars:
            c = env.vars[m].const
            if isinstance(c, int) and not isinstance(c, bool):
                d = 1 if "++" in ops or "++" in (node.postfix_operators or []) else -1
                env.vars[m] = V(const=c + d)
            else:
                env.vars[m] = V(unk=True)
        return v

    def _binary(self, node, env: Env) -> V:
        op = node.operator
        a = self._ev(node.operandl, env)
        if op == "&&" and a.const is False:
            return V(const=False)
        if op == "||" and a.const is True:
            return V(const=True)
        b = self._ev(node.operandr, env)
        danger, assumed = a.danger | b.danger, a.assumed | b.assumed
        if op == "instanceof":
            return V(unk=True)
        if a.const is not None and b.const is not None and a.const is not NULL and b.const is not NULL:
            x, y = a.const, b.const
            try:
                if op == "+" and (_is_str(x) or _is_str(y)):
                    return V(const=_tostr(x) + _tostr(y))
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    r = {"+": lambda: x + y, "-": lambda: x - y, "*": lambda: x * y,
                         "/": lambda: (x // y if isinstance(x, int) and isinstance(y, int) else x / y) if y else None,
                         "%": lambda: x % y if y else None, "<": lambda: x < y, ">": lambda: x > y, "<=": lambda: x <= y,
                         ">=": lambda: x >= y, "==": lambda: x == y, "!=": lambda: x != y,
                         "&": lambda: int(x) & int(y), "|": lambda: int(x) | int(y), "^": lambda: int(x) ^ int(y),
                         "<<": lambda: int(x) << int(y), ">>": lambda: int(x) >> int(y)}.get(op)
                    if r:
                        val = r()
                        if val is not None:
                            return V(const=val)
                if op in ("==", "!=") and type(x) == type(y):
                    return V(const=(x == y) if op == "==" else (x != y))
                if op == "&&" and isinstance(x, bool) and isinstance(y, bool):
                    return V(const=x and y)
                if op == "||" and isinstance(x, bool) and isinstance(y, bool):
                    return V(const=x or y)
            except Exception:  # noqa: BLE001
                pass
        if op in ("==", "!=", "<", ">", "<=", ">=", "&&", "||"):
            return V(unk=False)
        concat = op == "+" and (a.concat or b.concat or (a.const is None and (_is_str(b.const) or b.const is None))
                                or (b.const is None and _is_str(a.const)))
        return V(danger, assumed, None, concat, a.unk or b.unk)

    def _assign(self, node, env: Env) -> V:
        lv = node.expressionl
        val = self._ev(node.value, env)
        if node.type != "=":
            old = self._ev(lv, env)
            if node.type == "+=":
                val = V(old.danger | val.danger, old.assumed | val.assumed, None, True, old.unk or val.unk)
            else:
                val = V(old.danger | val.danger, old.assumed | val.assumed, None, False, old.unk or val.unk)
        if isinstance(lv, T.MemberReference):
            q, m = lv.qualifier or "", lv.member
            sels = lv.selectors or []
            if sels and isinstance(sels[0], T.ArraySelector):
                arr = env.vars.get(m) or env.fields.get(m)
                idx = self._ev(sels[0].index, env)
                if arr is not None and arr.obj is not None and arr.obj.kind == "array":
                    if arr.obj.items is not None and isinstance(idx.const, int) and 0 <= idx.const < len(arr.obj.items):
                        arr.obj.items[idx.const] = val
                    else:
                        arr.obj.any = join(arr.obj.any, val, keep_const=False) if arr.obj.any else val
                return val
            if not q:
                if m in env.vars or m not in env.fields:
                    env.vars[m] = val
                else:
                    env.fields[m] = val
            elif q == "this" or q.split(".")[0] == "this":
                env.fields[m] = val
            elif q.split(".")[0] in env.vars:
                r = env.vars[q.split(".")[0]]
                env.vars[q.split(".")[0]] = V(r.danger | val.danger, r.assumed | val.assumed, None, False, True, r.obj, r.cls)
            else:
                env.fields[m] = val
        return val

    # ------------------------------------------------------------------ object creation
    def _create(self, node, env: Env) -> V:
        tname = node.type.name if node.type is not None else ""
        sub = node.type.sub_type
        while sub is not None:
            tname = sub.name
            sub = sub.sub_type
        args = [self._ev(a, env) for a in (node.arguments or [])]
        danger = frozenset().union(*[a.danger for a in args]) if args else NONE
        assumed = frozenset().union(*[a.assumed for a in args]) if args else NONE
        unk = any(a.unk for a in args)
        if tname in ("ArrayList", "LinkedList", "Vector", "Stack", "ArrayDeque", "CopyOnWriteArrayList", "HashSet", "LinkedHashSet",
                     "TreeSet"):
            o = Obj("list", items=[])
            if args and args[0].obj is not None:
                o.items = [args[0].obj.union()] if args[0].obj.items is None else list(args[0].obj.items)
            return V(obj=o)
        if tname in ("HashMap", "LinkedHashMap", "TreeMap", "Hashtable", "ConcurrentHashMap", "Properties", "WeakHashMap"):
            return V(obj=Obj("map", entries={}))
        if tname in ("StringBuilder", "StringBuffer"):
            c = args[0] if args else V(const="")
            return V(obj=Obj("sb", content=V(c.danger, c.assumed, c.const if _is_str(c.const) else None, False, c.unk)))
        if tname == "Cookie" and len(args) >= 2:
            return V(danger=args[1].danger, assumed=args[1].assumed, obj=Obj("cookie", flags={"secure": False}))
        if tname in ("Random",) or tname.endswith(".Random"):
            if self.webish:
                self._add("WEAK_CRYPTO", "MEDIUM", "java.util.Random is predictable; do not use it for tokens, keys or session ids.",
                          "Use java.security.SecureRandom.")
            return V(obj=Obj("random"))
        if tname == "ProcessBuilder":
            self._check_cmd(args, node)
        if tname in ("File", "FileInputStream", "FileOutputStream", "FileReader", "FileWriter", "RandomAccessFile", "PrintWriter",
                     "FileSystemResource", "URI") and args:
            if not (tname == "PrintWriter" and not any(_is_path_like(a) for a in args)):
                self._check_path(args, node)
            return V(danger, assumed, const=None, unk=unk)
        if tname in self.classes:
            return V(unk=True, cls=tname, danger=danger, assumed=assumed)
        return V(danger, assumed, None, False, unk or not args)

    # ------------------------------------------------------------------ invocation
    def _invoke(self, node, env: Env) -> V:
        q, m = node.qualifier or "", node.member
        args = [self._ev(a, env) for a in node.arguments]
        pos = getattr(node, "position", None)
        if pos is not None:
            self.line = pos.line
        recv: Optional[V] = None
        qual = q
        if q:
            first = q.split(".")[0]
            if first in env.vars:
                recv, qual = env.vars[first], ""
            elif first in env.fields:
                recv, qual = env.fields[first], ""
            elif first == "this":
                recv, qual = V(unk=True), ""
        out = self._call(recv, qual, m, args, node, env, env.types.get(q.split(".")[0]) if q else None, q)
        return self._selectors(out, node, env, m)

    def _call(self, recv: Optional[V], qual: Optional[str], m: str, args: List[V], node, env: Env,
              rtype: Optional[str] = None, rname: str = "") -> V:
        qual = qual or ""
        qlast = qual.split(".")[-1] if qual else ""
        a0 = args[0] if args else None
        rd = recv.danger if recv else NONE
        ra = recv.assumed if recv else NONE
        adang = frozenset().union(*[a.danger for a in args]) if args else NONE
        aass = frozenset().union(*[a.assumed for a in args]) if args else NONE

        # ---------------- sources
        if m in SOURCE_METHODS and (recv is not None or qual):
            return V(danger=ALL, unk=True)
        if m in SOURCE_COLLECTIONS:
            return V(obj=Obj("array" if m in ("getParameterValues", "getCookies") else "list",
                             items=None, any_=V(danger=ALL, unk=True), entries={} if m == "getParameterMap" else None))
        if m == "getenv" and len(args) == 1:
            return V(danger=ALL, unk=True)
        if m == "getProperty":
            if len(args) >= 2 and args[1].const is not None and args[1].const is not NULL:
                return V(const=args[1].const)              # a configuration default we can see
            return V(danger=ALL, unk=True)

        # ---------------- sanitisers / parsers
        if qlast in ("Integer", "Long", "Double", "Float", "Short", "Byte", "Boolean") and m.startswith(("parse", "valueOf")):
            return V()
        if qlast == "Math":
            if m == "random" and self.webish:
                self._add("WEAK_CRYPTO", "MEDIUM", "Math.random() is predictable; do not use it for tokens, keys or session ids.",
                          "Use java.security.SecureRandom.")
            return V(unk=True)
        if m in XSS_ENCODERS:
            return V(danger=adang - {"xss"}, assumed=aass - {"xss"}, unk=True)
        if m in OTHER_ENCODERS:
            cat = OTHER_ENCODERS[m]
            d = (args[-1].danger if args else NONE) - {cat}
            return V(danger=d, assumed=(args[-1].assumed if args else NONE) - {cat}, unk=True)
        if m == "encode" and qlast == "URLEncoder":
            return V(danger=(adang - {"xss", "sql", "cmd", "path", "ldap", "xpath"}), assumed=NONE, unk=True)
        if m == "matches" and qlast == "Pattern":
            return V()

        # ---------------- sinks
        self._sinks(recv, qual, qlast, m, args, node, env, rtype, rname)

        # ---------------- containers / builders
        o = recv.obj if recv is not None else None
        if o is not None:
            r = self._container(recv, o, m, args)
            if r is not None:
                return r
        if qlast == "Collections" or qlast == "Arrays" or qlast == "Objects" or qlast == "String" and m in ("valueOf", "format", "join", "copyValueOf"):
            if m in ("asList", "unmodifiableList", "singletonList", "unmodifiableSet", "unmodifiableMap", "unmodifiableCollection",
                     "emptyList") and args:
                if m == "asList":
                    if len(args) == 1 and args[0].obj is not None and args[0].obj.kind == "array":
                        return V(obj=Obj("list", items=list(args[0].obj.items) if args[0].obj.items is not None else None,
                                         any_=args[0].obj.any))
                    return V(obj=Obj("list", items=list(args)))
                return args[0]
            if m in ("addAll",) and args and args[0].obj is not None and len(args) >= 2:
                for x in args[1:]:
                    args[0].obj.items = (args[0].obj.items or []) + [x if x.obj is None else x.obj.union()]
                return V()
            return V(adang, aass, None, m in ("format", "join"), any(a.unk for a in args))

        # ---------------- methods of this file
        if m in self.methods:
            cands = [(d, c) for d, c in self.methods[m] if isinstance(d, T.MethodDeclaration) and len(d.parameters) == len(args)
                     and d.body is not None]
            if recv is not None and recv.cls:
                narrowed = [(d, c) for d, c in cands if c.name == recv.cls]
                cands = narrowed or cands
            elif qual and qlast in self.classes:
                narrowed = [(d, c) for d, c in cands if c.name == qlast]
                cands = narrowed or cands
            if cands and len(self.stack) < MAX_DEPTH:
                res = None
                for decl, cls in cands[:3]:
                    if id(decl) in self.stack:
                        continue
                    v = self._call_method(decl, cls, args, env)
                    res = v if res is None else join(res, v, keep_const=False)
                if res is not None:
                    return res

        # ---------------- generic default: value derived from receiver and arguments
        if m in ("charAt",) and recv is not None and _is_str(recv.const) and a0 is not None and isinstance(a0.const, int) \
                and 0 <= a0.const < len(recv.const):
            return V(const=recv.const[a0.const])
        if m in ("length",) and recv is not None and _is_str(recv.const):
            return V(const=len(recv.const))
        if m in ("equals", "equalsIgnoreCase", "startsWith", "endsWith", "contains", "isEmpty", "matches") and recv is not None:
            if recv.const is not None and a0 is not None and a0.const is not None and m == "equals":
                return V(const=recv.const == a0.const)
            return V()
        if m in ("toString", "trim", "toLowerCase", "toUpperCase", "intern", "strip", "toCharArray", "getBytes", "clone") \
                and recv is not None:
            c = recv.const if (_is_str(recv.const) and m in ("toString", "trim", "intern", "strip")) else None
            if _is_str(recv.const) and m == "toLowerCase":
                c = recv.const.lower()
            if _is_str(recv.const) and m == "toUpperCase":
                c = recv.const.upper()
            return V(recv.danger, recv.assumed, c, recv.concat, recv.unk, recv.obj)
        if m in ("concat", "replace", "replaceAll", "replaceFirst", "substring", "format", "formatted", "repeat", "join", "split",
                 "append", "toString") and recv is not None:
            if recv.const is not None and all(a.const is not None for a in args) and m == "concat" and _is_str(recv.const):
                return V(const=recv.const + _tostr(a0.const))
            return V(recv.danger | adang, recv.assumed | aass, None, m in ("concat", "format", "formatted") or recv.concat,
                     recv.unk or any(a.unk for a in args))
        if recv is not None:
            return V(rd | adang, ra | aass, None, recv.concat, True, None)
        return V(adang, aass, None, False, (not args) or any(a.unk for a in args) or not adang)

    def _size(self, recv: V) -> V:
        o = recv.obj
        if o is not None and o.items is not None and o.any is None:
            return V(const=len(o.items))
        return V()

    def _call_method(self, decl, cls, args, caller: Env) -> V:
        env = Env(caller.fields)
        for p, a in zip(decl.parameters, args):
            env.vars[p.name] = a
            env.types[p.name] = self._tname(p.type)
        self.stack.append(id(decl))
        saved_line = self.line
        try:
            self._exec_block(decl.body, env)
        finally:
            self.stack.pop()
            self.line = saved_line
        return env.ret if env.ret is not None else V()

    # ------------------------------------------------------------------ containers
    def _container(self, recv: V, o: Obj, m: str, args: List[V]) -> Optional[V]:
        a0 = args[0] if args else None
        if o.kind == "response" or o.kind == "random":
            if o.kind == "response" and m in ("getWriter", "getOutputStream"):
                return V(obj=Obj("writer"))
            return None
        if o.kind == "sb":
            if m in ("append", "insert", "prepend"):
                x = args[-1] if args else V()
                c = o.content
                o.content = V(c.danger | x.danger, c.assumed | x.assumed,
                              (c.const + _tostr(x.const)) if (_is_str(c.const) and x.const is not None and x.const is not NULL) else None,
                              True, c.unk or x.unk)
                return recv
            if m == "toString":
                c = o.content
                return V(c.danger, c.assumed, c.const, c.concat, c.unk)
            if m in ("length",):
                return V()
            return None
        if o.kind == "cookie":
            if m == "setSecure" and a0 is not None:
                o.flags["secure"] = a0.const is True
                return V()
            if m in ("getValue", "getName"):
                return V(recv.danger, recv.assumed, None, False, True)
            return V()
        if o.kind == "list":
            items = o.items
            if m in ("add", "addElement", "addFirst", "addLast", "push", "offer"):
                x = args[-1] if args else V()
                if len(args) == 2 and items is not None and isinstance(a0.const, int) and 0 <= a0.const <= len(items):
                    items.insert(a0.const, x)
                elif items is not None and o.any is None:
                    items.append(x)
                else:
                    o.any = join(o.any, x, keep_const=False) if o.any else x
                return V(const=True)
            if m == "addAll" and args:
                src = args[-1].obj.union() if args[-1].obj is not None else args[-1]
                if items is not None and args[-1].obj is not None and args[-1].obj.items is not None and o.any is None:
                    items.extend(args[-1].obj.items)
                else:
                    o.any = join(o.any, src, keep_const=False) if o.any else src
                return V(const=True)
            if m in ("remove", "removeElementAt", "poll", "pop", "removeFirst") and items is not None and o.any is None:
                if m == "remove" and a0 is not None and isinstance(a0.const, int) and not isinstance(a0.const, bool):
                    if 0 <= a0.const < len(items):
                        return items.pop(a0.const)
                    return V(unk=True)
                if m in ("poll", "pop", "removeFirst") and items:
                    return items.pop(0)
                if a0 is not None and a0.const is not None:
                    for i, it in enumerate(items):
                        if _same(it.const, a0.const):
                            items.pop(i)
                            break
                    return V(const=True)
                return V(unk=True)
            if m in ("get", "elementAt", "getFirst", "peek", "firstElement", "getLast") and items is not None and o.any is None:
                if m in ("getFirst", "peek", "firstElement") and items:
                    return items[0]
                if m == "getLast" and items:
                    return items[-1]
                if a0 is not None and isinstance(a0.const, int) and 0 <= a0.const < len(items):
                    return items[a0.const]
                return o.union() if items else V(unk=True)
            if m in ("get", "getFirst", "peek", "next", "nextElement", "firstElement", "getLast", "elementAt", "iterator", "poll",
                     "pop", "remove", "toArray", "stream", "listIterator", "elements"):
                u = o.union()
                if m in ("iterator", "toArray", "stream", "listIterator", "elements"):
                    return V(obj=o) if m != "toArray" else V(obj=Obj("array", items=list(items) if items is not None else None, any_=o.any))
                return V(u.danger, u.assumed, None, False, True)
            if m in ("clear", "removeAll"):
                o.items, o.any = [], None
                return V()
            if m in ("size",):
                return V(const=len(items)) if items is not None and o.any is None else V()
            if m in ("hasNext", "hasMoreElements", "contains", "isEmpty", "containsAll", "equals"):
                return V()
            if m == "set" and items is not None and a0 is not None and isinstance(a0.const, int) and 0 <= a0.const < len(items):
                items[a0.const] = args[1]
                return V()
            return None
        if o.kind == "map":
            if m in ("put", "putIfAbsent") and len(args) == 2:
                k, v = args
                if o.entries is not None and k.const is not None and k.const is not NULL:
                    o.entries[k.const] = v
                else:
                    o.any = join(o.any, v, keep_const=False) if o.any else v
                return V(unk=True)
            if m == "setProperty" and len(args) == 2 and o.entries is not None and args[0].const is not None:
                o.entries[args[0].const] = args[1]
                return V()
            if m == "putAll" and args:
                src = args[0].obj.union() if args[0].obj is not None else args[0]
                o.any = join(o.any, src, keep_const=False) if o.any else src
                return V()
            if m in ("get", "getProperty", "getOrDefault") and a0 is not None:
                if o.entries is not None and a0.const is not None and a0.const is not NULL:
                    hit = o.entries.get(a0.const)
                    if hit is not None:
                        return join(hit, o.any, keep_const=False) if o.any else hit
                    if o.any is None:
                        return args[1] if (m in ("getOrDefault",) and len(args) == 2) else V(const=NULL)
                    return V(o.any.danger, o.any.assumed, None, False, True)
                u = o.union()
                return V(u.danger, u.assumed, None, False, True)
            if m == "remove" and a0 is not None:
                if o.entries is not None and a0.const is not None and a0.const in o.entries:
                    return o.entries.pop(a0.const)
                return V(unk=True)
            if m in ("keySet", "values", "entrySet", "keys", "elements", "propertyNames"):
                u = o.union()
                return V(obj=Obj("list", items=[u] if (o.entries or o.any) else [], any_=None))
            if m in ("clear",):
                o.entries, o.any = {}, None
                return V()
            if m in ("containsKey", "containsValue", "isEmpty", "size"):
                return V()
            return None
        if o.kind == "array":
            if m in ("length",):
                return V()
            return None
        return None

    # ------------------------------------------------------------------ sinks
    def _sinks(self, recv, qual, qlast, m, args, node, env: Env, rtype, rname):
        a0 = args[0] if args else None
        o = recv.obj if recv is not None else None
        rt = (rtype or "") + " " + (rname or "") + " " + qual
        # ---- SQL
        sql_arg = None
        if m in SQL_SINKS and a0 is not None:
            sql_arg = a0
        elif m in SQL_AMBIGUOUS and a0 is not None and (SQL_HINT.search(rt) or (_is_str(a0.const) and re.search(
                r"(?i)\b(select|insert|update|delete|call)\b", a0.const))):
            sql_arg = a0
        if sql_arg is not None and not (o is not None and o.kind in ("writer", "list", "map", "sb")):
            self._check_sql(sql_arg)
        # ---- command
        if m == "exec" and args and (rt.strip() != "" or True) and (qlast in ("Runtime", "") or "untime" in rt or "rt" in rt.lower()
                                                                      or (rname and rname.lower() in ("r", "rt", "runtime"))
                                                                      or not qual):
            self._check_cmd(args, node)
        if m in ("command",) and args and "ProcessBuilder" in rt + (rtype or ""):
            self._check_cmd(args, node)
        # ---- path
        if qlast in ("Paths", "Path") and m in ("get", "of") and args:
            self._check_path(args, node)
        if qlast == "Files" and args and m in ("newBufferedReader", "newBufferedWriter", "newInputStream", "newOutputStream",
                                               "readAllBytes", "readAllLines", "readString", "write", "writeString", "lines",
                                               "delete", "deleteIfExists", "copy", "move", "createFile", "createDirectory",
                                               "createDirectories", "list", "walk", "exists", "size"):
            self._check_path(args[:1], node)
        # ---- XSS
        if o is not None and o.kind == "writer" and m in ("print", "println", "write", "printf", "format", "append") and args:
            if any("xss" in a.danger or (a.obj is not None and "xss" in a.obj.union().danger) for a in args):
                self._add("XSS", "HIGH", "Cross-Site Scripting: untrusted data is written to the HTTP response unescaped.",
                          "HTML-encode all untrusted output (e.g. OWASP Java Encoder).")
        # ---- LDAP / XPath
        if m == "search" and len(args) >= 2 and re.search(r"(?i)ctx|context|ldap|dir", rt):
            if "ldap" in args[1].danger or (len(args) >= 1 and "ldap" in args[0].danger):
                self._add("LDAP_INJECTION", "HIGH", "LDAP Injection: untrusted data is used to build an LDAP search filter.",
                          "Escape filter values (e.g. ESAPI.encodeForLDAP) or use parameterised filters.")
        if m in ("evaluate", "compile", "selectNodes", "selectSingleNode") and a0 is not None and re.search(r"(?i)xp|xpath", rt):
            if "xpath" in a0.danger:
                self._add("XPATH_INJECTION", "HIGH", "XPath Injection: untrusted data is used to build an XPath expression.",
                          "Use XPath variables or escape the value (e.g. ESAPI.encodeForXPath).")
        # ---- trust boundary
        if m in ("setAttribute", "putValue") and len(args) == 2 and re.search(r"(?i)session|getSession", rt):
            if "trust" in args[0].danger or "trust" in args[1].danger:
                self._add("TRUST_BOUNDARY", "MEDIUM", "Trust boundary violation: untrusted input is stored in the session as if it were validated.",
                          "Validate the value before storing it in the session.")
        # ---- cookies
        if m == "addCookie" and a0 is not None and a0.obj is not None and a0.obj.kind == "cookie" and not a0.obj.flags.get("secure"):
            self._add("INSECURE_COOKIE", "MEDIUM", "Cookie is sent without the Secure flag (CWE-614).",
                      "Call cookie.setSecure(true).")
        # ---- weak crypto / hash (algorithm names may come from configuration defaults)
        if m == "getInstance" and a0 is not None and _is_str(a0.const):
            alg = a0.const.strip()
            if qlast == "MessageDigest" and WEAK_HASH.match(alg):
                self._add("WEAK_CRYPTO", "MEDIUM", f"Weak hash algorithm '{alg}' is broken for security purposes.",
                          "Use SHA-256 or stronger (and a salted KDF such as PBKDF2/bcrypt for passwords).")
            elif qlast == "Cipher" and WEAK_CIPHER.search(alg):
                if alg.upper() == "AES":
                    self._add("WEAK_CRYPTO", "MEDIUM", "Cipher.getInstance(\"AES\") defaults to ECB mode, which leaks plaintext patterns.",
                              "Use AES/GCM/NoPadding with a random IV.")
                else:
                    self._add("WEAK_CRYPTO", "MEDIUM", f"Weak or insecure cipher/mode '{alg}'.", "Use AES/GCM/NoPadding.")

    def _check_sql(self, a: V):
        if "sql" in a.danger or ("sql" in a.assumed and a.concat):
            self._add("SQL_INJECTION", "HIGH", "SQL Injection: untrusted data is concatenated into a SQL statement.",
                      "Use PreparedStatement with bound parameters instead of string concatenation.")
        elif a.unk and a.concat and not a.danger:
            self._add("SQL_INJECTION", "MEDIUM", "Possible SQL Injection: a SQL statement is built by concatenating non-constant values.",
                      "Use PreparedStatement with bound parameters.")

    def _check_cmd(self, args: List[V], node):
        vals = []
        for a in args:
            vals.append(a)
            if a.obj is not None:
                vals.append(a.obj.union())
        if any("cmd" in v.danger or "cmd" in v.assumed for v in vals):
            self._add("COMMAND_INJECTION", "HIGH", "OS Command Injection: untrusted data reaches a command execution API.",
                      "Avoid shelling out; if required use ProcessBuilder with a fixed argument list and a whitelist.")
        elif any(v.unk and v.const is None and v.concat for v in vals):
            self._add("COMMAND_INJECTION", "MEDIUM", "Possible command injection: a non-constant value is executed.",
                      "Validate the command against a whitelist.")

    def _check_path(self, args: List[V], node):
        vals = []
        for a in args:
            vals.append(a)
            if a.obj is not None:
                vals.append(a.obj.union())
        if any("path" in v.danger for v in vals):
            self._add("PATH_TRAVERSAL", "HIGH", "Path Traversal: untrusted data is used to build a file path.",
                      "Canonicalise the path and check it stays inside an allowed directory.")


def _is_path_like(v: V) -> bool:
    return _is_str(v.const) is False


def _tostr(x) -> str:
    if x is NULL:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float) and x == int(x):
        return str(x)
    return str(x)


def _coerce(x):
    if isinstance(x, str) and len(x) == 1:
        return x
    return x


def _walk(node):
    """All javalang nodes below `node`."""
    if isinstance(node, list):
        for x in node:
            yield from _walk(x)
        return
    if not hasattr(node, "children"):
        return
    yield node
    for c in node.children:
        if isinstance(c, (list, tuple)):
            for x in c:
                if hasattr(x, "children") or isinstance(x, list):
                    yield from _walk(x)
        elif hasattr(c, "children"):
            yield from _walk(c)


def analyze_java_taint(source: str) -> Optional[List[dict]]:
    """Findings of the taint interpreter, or None if the source cannot be parsed (the caller falls back to the text rules).
    `source` must be comment-free (line numbers are kept)."""
    if javalang is None:
        return None
    try:
        tree = javalang.parse.parse(source)
    except Exception:  # noqa: BLE001 - javalang has its own error types
        return None
    try:
        return JavaTaint(tree, source).run()
    except RecursionError:
        return None
