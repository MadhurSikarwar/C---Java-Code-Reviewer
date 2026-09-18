"""
IntelliReview — Java security & correctness rules
=================================================
The original Java analysis only knew three method names (exec / eval / readLine). This module adds a
small, dependency-free rule engine that works on comment/string-aware source text:

  SQL_INJECTION, COMMAND_INJECTION, PATH_TRAVERSAL, XSS, INSECURE_DESERIALIZATION,
  HARDCODED_CREDENTIAL, WEAK_CRYPTO, EMPTY_CATCH, RESOURCE_LEAK

Injection-style rules use light, per-method taint tracking:
  * a variable is TAINTED if it is assigned from an input source (readLine, getParameter, getenv, ...),
    from another tainted variable, or if it is a parameter of a non-private method (external input);
  * a variable is CONSTANT if it is only ever assigned string/number literals;
  * tainted data reaching a sink  -> HIGH,  non-constant unknown data reaching a sink -> MEDIUM,
    constants only -> no finding.
"""
import re
from typing import Dict, Any, List, Tuple, Set

from parsers.text_utils import sanitize
from analyzers.java_taint import analyze_java_taint

SOURCE_RE = re.compile(
    r"\.\s*readLine\s*\(|\bgetParameter\s*\(|\bgetParameterValues\s*\(|\bgetHeader\s*\(|\bgetCookies\s*\(|"
    r"\bgetQueryString\s*\(|\bgetenv\s*\(|\bgetProperty\s*\(|\bnextLine\s*\(|\bScanner\s*\(\s*System\.in|"
    r"\bSystem\.in\b|\bgetInputStream\s*\(|\bargs\s*\[|\bgetRequestURI\s*\(|\bgetPathInfo\s*\(|"
    r"\breadUTF\s*\(|\.\s*read\s*\(\s*\w+\s*\)")
SANITIZER_RE = re.compile(
    r"\bInteger\.parseInt\s*\(|\bLong\.parseLong\s*\(|\bDouble\.parseDouble\s*\(|\bBoolean\.parseBoolean\s*\(|"
    r"\bURLEncoder\.encode\s*\(|\bStringEscapeUtils\.|\bescapeHtml|\bencodeFor|\bsanitize\w*\s*\(|"
    r"\bHtmlUtils\.|\bESAPI\.|\bencodeForHTML\s*\(|\bPattern\.matches\s*\(")

METHOD_RE = re.compile(
    r"((?:public|protected|private|static|final|synchronized|abstract|native)\s+)*"
    r"[\w<>\[\],.?]+(?:\s+[\w<>\[\],.?]+)*?\s+(\w+)\s*\(([^()]*)\)\s*(?:throws\s+[\w.,\s]+)?\{")
NOT_METHODS = {"if", "for", "while", "switch", "catch", "synchronized", "try", "return", "new", "else", "do"}

SQL_SINKS = re.compile(r"\.\s*(executeQuery|executeUpdate|execute|addBatch|prepareStatement|prepareCall|"
                       r"createQuery|createNativeQuery|createSQLQuery)\s*\(")
CMD_SINKS = re.compile(r"\.\s*exec\s*\(|\bnew\s+ProcessBuilder\s*\(|\.\s*command\s*\(")
FILE_SINKS = re.compile(r"\bnew\s+(?:[\w.]+\.)?(?:File|FileInputStream|FileOutputStream|FileReader|FileWriter|"
                        r"RandomAccessFile|PrintWriter|Scanner)\s*\(|\bPaths\.get\s*\(|\bFiles\.\w+\s*\(")
XSS_SINKS = re.compile(r"\bgetWriter\s*\(\s*\)\s*\.\s*(?:print|println|write|printf|append)\s*\(|"
                       r"\bresponse\s*\.\s*getOutputStream\s*\(\s*\)\s*\.\s*(?:print|println|write)\s*\(")
DESER_SINKS = re.compile(r"\.\s*(readObject|readUnshared)\s*\(\s*\)")

# APIs that take a secret: regex -> index of the secret argument
CRED_SINKS = [
    (re.compile(r"\bnew\s+(?:[\w.]+\.)?PasswordAuthentication\s*\("), 1),
    (re.compile(r"\bnew\s+(?:[\w.]+\.)?KerberosKey\s*\("), 1),
    (re.compile(r"\bnew\s+(?:[\w.]+\.)?PBEKeySpec\s*\("), 0),
    (re.compile(r"\bnew\s+(?:[\w.]+\.)?SecretKeySpec\s*\("), 0),
    (re.compile(r"\bgetConnection\s*\("), 2),
    (re.compile(r"\bsetPassword\s*\("), 0),
]
CRED_NAME = re.compile(r"(?i)(pass(?:word|wd)?|pwd|secret|api_?key|token|credential|private_?key)")
WEAK_HASH = re.compile(r'\bMessageDigest\s*\.\s*getInstance\s*\(\s*"(MD2|MD4|MD5|SHA-?1)"', re.I)
WEAK_CIPHER = re.compile(r'\bCipher\s*\.\s*getInstance\s*\(\s*"(DES|DESede|RC2|RC4|Blowfish|[A-Za-z]+/ECB[^"]*|AES)"', re.I)
RESOURCE_TYPES = ("FileInputStream", "FileOutputStream", "FileReader", "FileWriter", "Socket", "ServerSocket",
                  "RandomAccessFile", "BufferedReader", "BufferedWriter", "InputStreamReader", "PrintWriter",
                  "ObjectInputStream", "ObjectOutputStream")


def _line(code: str, pos: int) -> int:
    return code.count("\n", 0, pos) + 1


def _match_paren(code: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(code)):
        c = code[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(code) - 1


def _brace_map(code: str) -> Dict[int, int]:
    stack, out = [], {}
    for i, c in enumerate(code):
        if c == "{":
            stack.append(i)
        elif c == "}" and stack:
            out[stack.pop()] = i
    return out


def _identifiers(expr: str) -> Set[str]:
    """Variable-like identifiers in an expression (skips method names, member selectors and keywords)."""
    ids = set()
    for m in re.finditer(r"(?<![\w.$])([A-Za-z_]\w*)\b(?!\s*\()", expr):
        w = m.group(1)
        if w not in ("new", "null", "true", "false", "this", "String", "int", "long", "char", "byte", "boolean",
                     "instanceof", "class"):
            ids.add(w)
    return ids


def _split_args(text: str):
    """Split call arguments at top-level commas (ignores commas inside parentheses and string literals)."""
    out, depth, cur, quote = [], 0, [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


def _issue(kind, sev, line, msg, fix):
    return {"type": kind, "severity": sev, "line": line, "function": kind.lower(), "message": msg, "suggestion": fix}


def detect_java_security(source: str) -> Dict[str, Any]:
    code = sanitize(source, keep_strings=True, java_text_blocks=True)
    nostr = sanitize(source, keep_strings=False, java_text_blocks=True)
    issues: List[Dict[str, Any]] = []
    seen = set()

    def add(kind, sev, line, msg, fix):
        key = (kind, line)
        if key not in seen:
            seen.add(key)
            issues.append(_issue(kind, sev, line, msg, fix))

    # ---- file level rules ------------------------------------------------------------------
    for m in re.finditer(r'\b(\w+)\s*=\s*"([^"\n]{1,200})"', code):
        name, val = m.group(1), m.group(2)
        if CRED_NAME.search(name) and val.strip() and not re.fullmatch(r"\s*|\*+|<.*>|\$\{.*\}|%s|null|none", val, re.I) \
                and "password" != val.lower():
            add("HARDCODED_CREDENTIAL", "MEDIUM", _line(code, m.start()),
                f"Hard-coded credential: '{name}' is assigned a string literal.",
                "Load secrets from environment variables or a secrets manager, never from source code.")
    for m in re.finditer(r'\bPasswordAuthentication\s*\(\s*[^,]+,\s*"([^"]+)"|\bgetConnection\s*\(\s*[^,]+,\s*[^,]+,\s*"([^"]+)"', code):
        add("HARDCODED_CREDENTIAL", "MEDIUM", _line(code, m.start()),
            "Hard-coded password passed directly to an authentication / connection API.",
            "Read the password from a secure configuration source.")
    for m in WEAK_HASH.finditer(code):
        add("WEAK_CRYPTO", "MEDIUM", _line(code, m.start()),
            f"Weak hash algorithm '{m.group(1)}' is broken for security purposes.",
            "Use SHA-256 or stronger (and a salted KDF such as PBKDF2/bcrypt for passwords).")
    for m in WEAK_CIPHER.finditer(code):
        alg = m.group(1)
        if alg.upper() == "AES":      # bare "AES" defaults to ECB in the JDK
            add("WEAK_CRYPTO", "MEDIUM", _line(code, m.start()),
                "Cipher.getInstance(\"AES\") defaults to ECB mode, which leaks plaintext patterns.",
                "Use AES/GCM/NoPadding with a random IV.")
        else:
            add("WEAK_CRYPTO", "MEDIUM", _line(code, m.start()),
                f"Weak or insecure cipher/mode '{alg}'.", "Use AES/GCM/NoPadding.")
    for m in re.finditer(r"catch\s*\([^)]*\)\s*\{\s*\}", nostr):
        add("EMPTY_CATCH", "MEDIUM", _line(nostr, m.start()),
            "Empty catch block silently swallows the exception.",
            "Log or handle the exception, or rethrow it.")
    static_consts = {m.group(1) for m in re.finditer(
        r"\bstatic\s+final\s+[\w<>\[\]]+\s+(\w+)\s*=\s*(?:\"[^\"]*\"|-?\d[\w.]*)\s*;", code)}

    # ---- taint rules: the AST interpreter (analyzers/java_taint.py); the text rules below are the fallback ---------
    ast_findings = analyze_java_taint(code)
    if ast_findings is not None:
        for f in ast_findings:
            add(f["type"], f["severity"], f["line"], f["message"], f["suggestion"])

    # ---- per-method taint rules -------------------------------------------------------------
    bmap = _brace_map(nostr)
    for mm in METHOD_RE.finditer(nostr):
        name = mm.group(2)
        if name in NOT_METHODS:
            continue
        open_idx = mm.end() - 1
        close_idx = bmap.get(open_idx)
        if close_idx is None:
            continue
        header = mm.group(0)
        is_private = bool(re.search(r"\bprivate\b", header))
        params = set(re.findall(r"(\w+)\s*(?:,|$)", mm.group(3).strip())) if mm.group(3).strip() else set()

        body = code[open_idx + 1:close_idx]
        body_ns = nostr[open_idx + 1:close_idx]
        base_line = _line(code, open_idx + 1)
        tainted: Set[str] = set()                                   # proven: derived from an input source
        ext_params: Set[str] = set() if is_private else set(params)  # untrusted-by-assumption (public API surface)
        const_vars: Set[str] = set(static_consts)

        # try-with-resources header ranges (their declarations are closed automatically)
        twr_ranges = []
        for tm in re.finditer(r"\btry\s*\(", body_ns):
            twr_ranges.append((tm.end() - 1, _match_paren(body_ns, tm.end() - 1)))

        def in_twr(pos):
            return any(a <= pos <= b for a, b in twr_ranges)

        # sequential statements
        # Statements are processed in order, but branches are tracked: taint introduced in one `if` branch must not be
        # cancelled by an assignment in the sibling `else` branch, so an "untainting" assignment only counts when it is
        # on the same chain of enclosing blocks (same block, or an ancestor/descendant of it).
        block_path: tuple = ()
        block_counter = 0
        taint_path: Dict[str, tuple] = {}
        nonconst: Set[str] = set()
        secret_literals: Set[str] = set()          # variables holding a non-empty string literal (candidate secrets)
        secret_path: Dict[str, tuple] = {}

        def comparable(a: tuple, b: tuple) -> bool:
            return a[:len(b)] == b or b[:len(a)] == a

        for sm in re.finditer(r"[^;{}]+|[{}]", body):
            tok = body[sm.start():sm.end()]
            if tok == "{":
                block_counter += 1
                block_path = block_path + (block_counter,)
                continue
            if tok == "}":
                block_path = block_path[:-1]
                continue
            stmt = tok
            stmt_ns = body_ns[sm.start():sm.end()]
            if not stmt.strip():
                continue
            line = base_line + body.count("\n", 0, sm.start()) + stmt[:len(stmt) - len(stmt.lstrip())].count("\n")

            _check_sinks(stmt, stmt_ns, line, tainted, ext_params, const_vars, add, secret_literals,
                         injection=ast_findings is None)

            core = re.sub(r"^\s*(?:else\s+)?(?:if\s*\((?:[^()]|\([^()]*\))*\)\s*)?", "", stmt)
            am = re.match(r"\s*(?:final\s+)?(?:[\w.<>\[\],?]+\s+)?(\w+)(?:\s*\[\s*\])*\s*(\+?=)\s*(?!=)(.+)$", core, re.S)
            cond_assign = re.search(r"\(\s*(\w+)\s*=\s*(?!=)([^;]*?)\)\s*(?:!=|==)", stmt)
            for var, op, rhs in ([(am.group(1), am.group(2), am.group(3))] if am else []) + \
                                ([(cond_assign.group(1), "=", cond_assign.group(2))] if cond_assign else []):
                rhs_t = SOURCE_RE.search(rhs) is not None or bool(_identifiers(rhs) & tainted)
                rhs_e = (not rhs_t) and bool(_identifiers(rhs) & ext_params)
                if SANITIZER_RE.search(rhs):
                    rhs_t = rhs_e = False
                rhs_const = not rhs_t and not rhs_e and (
                    re.fullmatch(r'\s*(?:"[^"]*"|-?\d[\w.]*|null|true|false|[\w.]+\.(?:toString)?\(\))\s*', rhs.strip()) is not None
                    or _identifiers(rhs) <= const_vars and "(" not in re.sub(r'"[^"]*"', "", rhs))
                if op == "=":
                    if re.fullmatch(r'\s*"[^"]+"\s*', rhs):
                        secret_literals.add(var)
                        secret_path[var] = block_path
                    elif not re.fullmatch(r"\s*null\s*", rhs) and comparable(block_path, secret_path.get(var, ())):
                        secret_literals.discard(var)
                if op == "+=":
                    if rhs_t:
                        tainted.add(var)
                        taint_path[var] = block_path
                    elif rhs_e:
                        ext_params.add(var)
                        taint_path[var] = block_path
                    if not rhs_const:
                        const_vars.discard(var)
                        nonconst.add(var)
                    continue
                if rhs_t:
                    tainted.add(var)
                    ext_params.discard(var)
                    taint_path[var] = block_path
                    const_vars.discard(var)
                    nonconst.add(var)
                elif rhs_e:
                    ext_params.add(var)
                    taint_path[var] = block_path
                    const_vars.discard(var)
                    nonconst.add(var)
                else:
                    if comparable(block_path, taint_path.get(var, ())):
                        tainted.discard(var)
                        ext_params.discard(var)
                    if rhs_const and var not in nonconst:
                        const_vars.add(var)
                    else:
                        const_vars.discard(var)
                        nonconst.add(var)

        # resource leaks
        stdin_vars = set(re.findall(r"(\w+)\s*=\s*new\s+(?:[\w.]+\.)?InputStreamReader\s*\(\s*System\.in", body_ns))
        for rm in re.finditer(r"(?:(\w+)\s*=\s*)?new\s+(?:[\w.]+\.)?(" + "|".join(RESOURCE_TYPES) + r")\s*\(", body_ns):
            var = rm.group(1)
            if in_twr(rm.start()) or not var:
                continue
            stmt_end = body_ns.find(";", rm.start())
            stmt_txt = body_ns[rm.start():stmt_end if stmt_end != -1 else len(body_ns)]
            if "System.in" in stmt_txt or any(re.search(rf"\b{re.escape(v)}\b", stmt_txt) for v in stdin_vars):
                continue          # wrapping stdin: closing it is not expected
            if re.search(rf"\b{re.escape(var)}\s*\.\s*close\s*\(", body_ns) or \
                    re.search(rf"\breturn\s+{re.escape(var)}\s*;|=\s*{re.escape(var)}\s*;|\(\s*{re.escape(var)}\s*[,)]", body_ns):
                continue
            # a wrapper constructed around another tracked resource is closed together with it
            add("RESOURCE_LEAK", "MEDIUM", base_line + body.count("\n", 0, rm.start()),
                f"Resource leak: '{var}' ({rm.group(2)}) is opened but never closed.",
                "Use try-with-resources or close it in a finally block.")

    return {"issues": issues, "unsafe_function_count": len(issues)}


def _args_of(stmt: str, stmt_ns: str, open_idx: int) -> str:
    end = _match_paren(stmt_ns, open_idx)
    return stmt[open_idx + 1:end]


def _classify(arg: str, tainted: Set[str], ext_params: Set[str], const_vars: Set[str]) -> str:
    """'tainted' (proven) | 'param' (public-API input) | 'unknown' | 'constant'"""
    if SANITIZER_RE.search(arg):
        return "constant"
    if SOURCE_RE.search(arg):
        return "tainted"
    ids = _identifiers(re.sub(r'"(?:[^"\\]|\\.)*"', '""', arg))
    if ids & tainted:
        return "tainted"
    if ids & ext_params:
        return "param"
    if ids - const_vars - {"System", "out", "Integer", "Long", "Math"}:
        return "unknown"
    return "constant"


def _check_sinks(stmt, stmt_ns, line, tainted, ext_params, const_vars, add, secret_literals=frozenset(), injection=True):
    def each(rx):
        for m in rx.finditer(stmt_ns):
            open_idx = stmt_ns.find("(", m.end() - 1 if m.group(0).endswith("(") else m.start())
            if open_idx == -1:
                continue
            yield m, _args_of(stmt, stmt_ns, open_idx)

    for rx, idx in CRED_SINKS:
        for m, arg in each(rx):
            parts = _split_args(arg)
            if len(parts) <= idx:
                continue
            value = re.sub(r"\.\s*(?:toCharArray|getBytes)\s*\([^)]*\)", "", parts[idx]).strip()
            if re.fullmatch(r'"[^"]+"', value) or value in secret_literals:
                add("HARDCODED_CREDENTIAL", "MEDIUM", line,
                    "Hard-coded credential: a fixed string literal is used as a password or key.",
                    "Read secrets from configuration or a secrets manager, never from source code.")
    if not injection:
        for m in DESER_SINKS.finditer(stmt_ns):
            untrusted = SOURCE_RE.search(stmt) or _identifiers(stmt) & (tainted | ext_params) or "Socket" in stmt \
                or "request" in stmt.lower()
            add("INSECURE_DESERIALIZATION", "HIGH" if untrusted else "MEDIUM", line,
                "Insecure deserialization: ObjectInputStream.readObject() on data that may be attacker controlled.",
                "Avoid native Java serialization for untrusted data; use a look-ahead ObjectInputFilter or JSON.")
        return
    for m, arg in each(SQL_SINKS):
        kind = _classify(arg, tainted, ext_params, const_vars)
        if kind in ("tainted", "param") and ("+" in arg or kind == "tainted"):
            add("SQL_INJECTION", "HIGH", line, "SQL Injection: untrusted data is concatenated into a SQL statement.",
                "Use PreparedStatement with bound parameters instead of string concatenation.")
        elif kind == "unknown" and "+" in arg:
            add("SQL_INJECTION", "MEDIUM", line,
                "Possible SQL Injection: a SQL statement is built by concatenating non-constant values.",
                "Use PreparedStatement with bound parameters.")
    for m, arg in each(CMD_SINKS):
        kind = _classify(arg, tainted, ext_params, const_vars)
        if kind in ("tainted", "param"):
            add("COMMAND_INJECTION", "HIGH", line, "OS Command Injection: untrusted data reaches a command execution API.",
                "Avoid shelling out; if required use ProcessBuilder with a fixed argument list and a whitelist.")
        elif kind == "unknown":
            add("COMMAND_INJECTION", "MEDIUM", line, "Possible command injection: a non-constant value is executed.",
                "Validate the command against a whitelist.")
    for m, arg in each(FILE_SINKS):
        if _classify(arg, tainted, ext_params, const_vars) == "tainted":
            add("PATH_TRAVERSAL", "HIGH", line, "Path Traversal: untrusted data is used to build a file path.",
                "Canonicalise the path and check it stays inside an allowed directory.")
    for m, arg in each(XSS_SINKS):
        if _classify(arg, tainted, ext_params, const_vars) == "tainted":
            add("XSS", "HIGH", line, "Cross-Site Scripting: untrusted data is written to the HTTP response unescaped.",
                "HTML-encode all untrusted output (e.g. OWASP Java Encoder).")
    for m in DESER_SINKS.finditer(stmt_ns):
        untrusted = SOURCE_RE.search(stmt) or _identifiers(stmt) & (tainted | ext_params) or "Socket" in stmt \
            or "request" in stmt.lower()
        add("INSECURE_DESERIALIZATION", "HIGH" if untrusted else "MEDIUM", line,
            "Insecure deserialization: ObjectInputStream.readObject() on data that may be attacker controlled.",
            "Avoid native Java serialization for untrusted data; use a look-ahead ObjectInputFilter or JSON.")
