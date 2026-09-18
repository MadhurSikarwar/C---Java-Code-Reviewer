"""
IntelliReview — labeled program generator
=========================================
The old training data was three Gaussian blobs of *feature vectors* (small / medium / large code) whose label was
simply the blob — so the models learned "bigger = riskier" and reached 100 % accuracy on data they had
effectively been told the answer to. This module generates real SOURCE CODE instead:

  * a program is a random number (0..N) of correct helper functions (incl. tricky-but-safe "near misses" such as
    strcpy of a short literal, NULL-checked malloc/free, PreparedStatement, try-with-resources) ...
  * ... plus zero or more injected defects, each with a known severity class,
  * the label is decided by WHICH defects were injected — never by how large the program is.

Program size is sampled independently of the label, so size carries no information about it. The generated code is
run through the real analysis pipeline (pipeline.static_analysis) to obtain features, so the features are exactly
what the models see in production.

Classes: 0 = Clean, 1 = Moderate Risk (resource leaks, uninitialised reads, weak crypto, swallowed exceptions ...),
         2 = High Risk (memory corruption, injection, insecure deserialization ...).
"""
import random
from typing import List, Tuple, Dict

# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
_WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta", "zeta", "lambda", "node", "item",
          "buf", "data", "value", "count", "total", "index", "temp", "result", "input", "output", "cache", "queue"]
_TYPES = ["int", "long", "unsigned int", "short"]
_DISTRACTORS_C = [
    "/* NOTE: never call gets(buf) here, use fgets instead */",
    "// strcpy(dst, src) is unsafe -- see snprintf below",
    '/* printf(user_input) would be a format string bug */',
    "// TODO: free(ptr) after the loop is done",
    "/* malloc(size) result must be checked */",
]
_DISTRACTORS_J = [
    "// never build SQL with + and user input",
    "/* Runtime.exec(cmd) is dangerous with untrusted cmd */",
    "// TODO: close the stream in a finally block",
    '/* password = "changeme" is only an example */',
]


def _name(rng: random.Random, prefix: str = "") -> str:
    return f"{prefix}{rng.choice(_WORDS)}_{rng.randint(0, 9999)}"


def _maybe_comment(rng, lang) -> str:
    if rng.random() < 0.35:
        return "    " + rng.choice(_DISTRACTORS_C if lang == "C" else _DISTRACTORS_J) + "\n"
    return ""


# --------------------------------------------------------------------------------------------------
# C : correct code
# --------------------------------------------------------------------------------------------------
def _c_sum(n, rng):
    t = rng.choice(_TYPES)
    k = rng.randint(2, 9)
    body = f"""/* Sums a filtered range, variant {n}. */
static {t} {n}({t} limit) {{
    {t} total = 0;
    for ({t} i = 0; i < limit; i++) {{
        if (i % {k} == 0) {{
            total += i;
        }} else if (i % {k + 1} == 0) {{
            total -= 1;
        }}
    }}
    return total;
}}
"""
    if rng.random() < 0.4:  # nested loops (deep but correct)
        body = body.replace("        if (i % ", "        for ({0} j = 0; j < 2; j++) {{ total += j; }}\n        if (i % ".format(t), 1)
    return body


def _c_alloc_free(n, rng):
    return f"""static int {n}(size_t len) {{
    int *buf = malloc(len * sizeof(int));
    if (buf == NULL) {{
        return -1;
    }}
    for (size_t i = 0; i < len; i++) {{
        buf[i] = (int)(i * {rng.randint(2, 7)});
    }}
    int last = len > 0 ? buf[len - 1] : 0;
    free(buf);
    return last;
}}
"""


def _c_strings(n, rng):
    size = rng.choice([32, 64, 128])
    return f"""static void {n}(const char *name) {{
    char line[{size}];
    snprintf(line, sizeof(line), "hello %s", name);
    line[sizeof(line) - 1] = '\\0';
    puts(line);
}}
"""


def _c_list(n, rng):
    return f"""struct {n}_node {{ int value; struct {n}_node *next; }};
static struct {n}_node *{n}_push(struct {n}_node *head, int v) {{
    struct {n}_node *n = malloc(sizeof(*n));
    if (!n) {{ return head; }}
    n->value = v;
    n->next = head;
    return n;
}}
static void {n}_free(struct {n}_node *head) {{
    while (head) {{
        struct {n}_node *next = head->next;
        free(head);
        head = next;
    }}
}}
"""


def _c_recursion(n, rng):
    if rng.random() < 0.5:
        return f"static int {n}(int n) {{ if (n <= 1) {{ return 1; }} return n * {n}(n - 1); }}\n"
    return f"static int {n}(int a, int b) {{ return b == 0 ? a : {n}(b, a % b); }}\n"


def _c_switch(n, rng):
    return f"""static int {n}(int op) {{
    switch (op) {{
        case 0: return {rng.randint(1, 9)};
        case 1: return {rng.randint(10, 99)};
        case 2: return -1;
        default: break;
    }}
    return 0;
}}
"""


def _c_bubble(n, rng):
    return f"""static void {n}(int *a, int len) {{
    for (int i = 0; i < len - 1; i++) {{
        for (int j = 0; j < len - i - 1; j++) {{
            if (a[j] > a[j + 1]) {{
                int t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }}
        }}
    }}
}}
"""


def _c_near_miss_strcpy(n, rng):     # provably safe strcpy of a short literal
    return f"""static void {n}(void) {{
    char dst[{rng.choice([16, 32])}];
    strcpy(dst, "ok");
    puts(dst);
}}
"""


def _c_realloc(n, rng):
    return f"""static int *{n}(int *p, size_t newlen) {{
    int *q = realloc(p, newlen * sizeof(int));
    if (q == NULL) {{
        free(p);
        return NULL;
    }}
    return q;
}}
"""


def _c_file(n, rng):
    return f"""static int {n}(const char *path) {{
    FILE *f = fopen(path, "r");
    if (f == NULL) {{ return -1; }}
    int c = fgetc(f);
    fclose(f);
    return c;
}}
"""


def _c_goto_cleanup(n, rng):
    return f"""static int {n}(int flag) {{
    char *a = malloc(16);
    int rc = 0;
    if (a == NULL) {{ return -1; }}
    if (flag) {{ rc = 1; goto out; }}
    a[0] = 'x';
out:
    free(a);
    return rc;
}}
"""


C_SAFE = [_c_sum, _c_alloc_free, _c_strings, _c_list, _c_recursion, _c_switch, _c_bubble, _c_near_miss_strcpy,
          _c_realloc, _c_file, _c_goto_cleanup]

# --------------------------------------------------------------------------------------------------
# C : injected defects   (name -> (class, template))
# --------------------------------------------------------------------------------------------------
def _d_gets(n, rng):
    return f"static void {n}(void) {{\n    char b[{rng.choice([8, 16, 64])}];\n    gets(b);\n    puts(b);\n}}\n"


def _d_strcpy(n, rng):
    return f"static void {n}(const char *src) {{\n    char b[{rng.choice([8, 16])}];\n    strcpy(b, src);\n    puts(b);\n}}\n"


def _d_strcpy_literal(n, rng):
    return f'static void {n}(void) {{\n    char b[4];\n    strcpy(b, "this literal is far too long");\n    puts(b);\n}}\n'


def _d_format(n, rng):
    return f"static void {n}(char *user) {{\n    printf(user);\n}}\n"


def _d_uaf(n, rng):
    return f"static int {n}(void) {{\n    int *p = malloc(sizeof(int));\n    if (!p) {{ return 0; }}\n    *p = {rng.randint(1, 9)};\n    free(p);\n    return *p;\n}}\n"


def _d_double_free(n, rng):
    return f"static void {n}(void) {{\n    char *p = malloc(8);\n    if (!p) {{ return; }}\n    free(p);\n    free(p);\n}}\n"


def _d_null_deref(n, rng):
    return f"static int {n}(void) {{\n    int *p = NULL;\n    *p = {rng.randint(1, 9)};\n    return 0;\n}}\n"


def _d_system(n, rng):
    return f'static void {n}(void) {{\n    char *cmd = getenv("CMD");\n    system(cmd);\n}}\n'


def _d_leak(n, rng):
    return f"static void {n}(void) {{\n    char *p = malloc({rng.randint(8, 128)});\n    if (p) {{ p[0] = 'a'; }}\n}}\n"


def _d_leak_early(n, rng):
    return f"static int {n}(int x) {{\n    char *b = malloc(32);\n    if (!b) {{ return -1; }}\n    if (x < 0) {{\n        return -2;\n    }}\n    free(b);\n    return 0;\n}}\n"


def _d_uninit(n, rng):
    return f"static int {n}(int flag) {{\n    int v;\n    if (flag) {{ v = 1; }}\n    return v + {rng.randint(1, 5)};\n}}\n"


def _d_scanf(n, rng):
    return f'static void {n}(void) {{\n    char name[16];\n    scanf("%s", name);\n    puts(name);\n}}\n'


def _d_memcpy(n, rng):
    return f"static void {n}(char *dst, const char *src, size_t n) {{\n    memcpy(dst, src, n);\n}}\n"


C_DEFECTS: Dict[str, Tuple[int, callable]] = {
    "gets": (2, _d_gets), "strcpy": (2, _d_strcpy), "strcpy_literal": (2, _d_strcpy_literal),
    "format_string": (2, _d_format), "use_after_free": (2, _d_uaf), "double_free": (2, _d_double_free),
    "null_deref": (2, _d_null_deref), "system": (2, _d_system),
    "leak": (1, _d_leak), "leak_early_return": (1, _d_leak_early), "uninit": (1, _d_uninit),
    "scanf": (1, _d_scanf), "memcpy": (1, _d_memcpy),
}

# --------------------------------------------------------------------------------------------------
# Java
# --------------------------------------------------------------------------------------------------
def _j_sum(n, rng):
    return f"""    /** Sums positive values, variant {n}. */
    static int {n}(java.util.List<Integer> xs) {{
        int total = 0;
        for (int x : xs) {{
            if (x > {rng.randint(0, 3)}) {{ total += x; }} else {{ total -= 1; }}
        }}
        return total;
    }}
"""


def _j_builder(n, rng):
    return f"""    static String {n}(String[] parts) {{
        StringBuilder sb = new StringBuilder();
        for (String p : parts) {{ sb.append(p).append(','); }}
        return sb.toString();
    }}
"""


def _j_recursion(n, rng):
    return f"    static int {n}(int n) {{ return n <= 1 ? 1 : n * {n}(n - 1); }}\n"


def _j_prepared(n, rng):
    return f"""    static int {n}(java.sql.Connection c, String name) throws java.sql.SQLException {{
        try (java.sql.PreparedStatement ps = c.prepareStatement("SELECT id FROM users WHERE name = ?")) {{
            ps.setString(1, name);
            try (java.sql.ResultSet rs = ps.executeQuery()) {{ return rs.next() ? rs.getInt(1) : -1; }}
        }}
    }}
"""


def _j_const_sql(n, rng):
    return f"""    static void {n}(java.sql.Connection c) throws java.sql.SQLException {{
        String table = "users";
        try (java.sql.Statement s = c.createStatement()) {{ s.executeQuery("SELECT * FROM " + table); }}
    }}
"""


def _j_twr(n, rng):
    return f"""    static String {n}(String path) throws java.io.IOException {{
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(path))) {{
            return r.readLine();
        }}
    }}
"""


def _j_exec_literal(n, rng):
    return f"""    static void {n}() throws java.io.IOException {{
        Runtime.getRuntime().exec("ls");
    }}
"""


def _j_sha256(n, rng):
    return f"""    static byte[] {n}(byte[] in) throws Exception {{
        return java.security.MessageDigest.getInstance("SHA-256").digest(in);
    }}
"""


def _j_switch(n, rng):
    return f"""    static int {n}(int op) {{
        switch (op) {{
            case 0: return {rng.randint(1, 9)};
            case 1: return {rng.randint(10, 99)};
            default: return 0;
        }}
    }}
"""


J_SAFE = [_j_sum, _j_builder, _j_recursion, _j_prepared, _j_const_sql, _j_twr, _j_exec_literal, _j_sha256, _j_switch]


def _jd_sqli(n, rng):
    return f"""    static void {n}(java.sql.Connection c) throws Exception {{
        java.io.BufferedReader r = new java.io.BufferedReader(new java.io.InputStreamReader(System.in));
        String u = r.readLine();
        java.sql.Statement s = c.createStatement();
        s.executeQuery("SELECT * FROM t WHERE n='" + u + "'");
    }}
"""


def _jd_cmd(n, rng):
    return f"""    static void {n}() throws Exception {{
        String d = System.getenv("CMD");
        Runtime.getRuntime().exec(d);
    }}
"""


def _jd_deser(n, rng):
    return f"""    static Object {n}(java.io.InputStream in) throws Exception {{
        java.io.ObjectInputStream o = new java.io.ObjectInputStream(in);
        return o.readObject();
    }}
"""


def _jd_path(n, rng):
    return f"""    static void {n}() throws Exception {{
        String p = System.getProperty("user.input");
        java.io.FileInputStream f = new java.io.FileInputStream(p);
        f.close();
    }}
"""


def _jd_password(n, rng):
    return f'    static final String PASSWORD_{rng.randint(0, 999)} = "{rng.choice(["hunter2", "s3cret!", "letmein", "P@ssw0rd"])}";\n'


def _jd_empty_catch(n, rng):
    return f"""    static void {n}() {{
        try {{ Integer.parseInt("x"); }} catch (Exception e) {{}}
    }}
"""


def _jd_md5(n, rng):
    return f"""    static byte[] {n}(byte[] in) throws Exception {{
        return java.security.MessageDigest.getInstance("MD5").digest(in);
    }}
"""


def _jd_resource_leak(n, rng):
    return f"""    static int {n}(String p) throws java.io.IOException {{
        java.io.FileInputStream f = new java.io.FileInputStream(p);
        return f.read();
    }}
"""


J_DEFECTS: Dict[str, Tuple[int, callable]] = {
    "sqli": (2, _jd_sqli), "cmd": (2, _jd_cmd), "deser": (2, _jd_deser), "path": (2, _jd_path),
    "password": (1, _jd_password), "empty_catch": (1, _jd_empty_catch), "md5": (1, _jd_md5),
    "resource_leak": (1, _jd_resource_leak),
}


# --------------------------------------------------------------------------------------------------
# program assembly
# --------------------------------------------------------------------------------------------------
def make_program(language: str, rng: random.Random, n_safe: int, defects: List[str]) -> Tuple[str, int, List[str]]:
    """Returns (source, label, injected_defect_names)."""
    language = language.upper()
    label = 0
    parts: List[str] = []
    if language == "C":
        header = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n"
        names = []
        for _ in range(n_safe):
            fn = rng.choice(C_SAFE)
            nm = _name(rng, "f")
            parts.append(_maybe_comment(rng, "C") + fn(nm, rng))
            names.append(nm)
        for d in defects:
            cls, fn = C_DEFECTS[d]
            label = max(label, cls)
            parts.append(fn(_name(rng, "g"), rng))
        rng.shuffle(parts)
        body = "\n".join(parts)
        main = "\nint main(void) {\n    return 0;\n}\n"
        return header + body + main, label, defects

    header = "import java.util.*;\n\npublic class Prog%d {\n" % rng.randint(0, 99999)
    for _ in range(n_safe):
        fn = rng.choice(J_SAFE)
        parts.append(_maybe_comment(rng, "JAVA") + fn(_name(rng, "m"), rng))
    for d in defects:
        cls, fn = J_DEFECTS[d]
        label = max(label, cls)
        parts.append(fn(_name(rng, "x"), rng))
    rng.shuffle(parts)
    return header + "\n".join(parts) + "}\n", label, defects


def sample_program(rng: random.Random, language: str = None) -> Dict:
    """One random labeled program. Size is drawn independently of the label."""
    language = language or rng.choice(["C", "JAVA"])
    n_safe = min(60, int(rng.expovariate(1 / 8)))          # heavy-tailed size, unrelated to label
    r = rng.random()
    table = C_DEFECTS if language == "C" else J_DEFECTS
    high = [k for k, (c, _) in table.items() if c == 2]
    mod = [k for k, (c, _) in table.items() if c == 1]
    if r < 0.40:
        defects: List[str] = []
    elif r < 0.72:
        defects = rng.sample(mod, k=rng.randint(1, min(3, len(mod))))
    else:
        defects = rng.sample(high, k=rng.randint(1, 2))
        if rng.random() < 0.4:
            defects += rng.sample(mod, k=1)
    src, label, defects = make_program(language, rng, n_safe, defects)
    return {"source": src, "language": language, "label": label, "defects": defects, "n_safe": n_safe}
