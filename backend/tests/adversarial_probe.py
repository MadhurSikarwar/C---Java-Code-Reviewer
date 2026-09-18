"""
Adversarial probe for the IntelliReview API.

Runs hostile / edge-case inputs against a RUNNING server (default http://127.0.0.1:8000)
and prints, per case, what every model predicted next to what a human would expect.

    python tests/adversarial_probe.py [--url http://127.0.0.1:8000]

Expectation codes:  C = Clean, M = Moderate Risk, H = High Risk, ERR = must be rejected cleanly (4xx, no 500)
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

MODELS = ["ensemble", "dl", "v3", "v4_dl"]
SHORT = {"Clean": "C", "Moderate Risk": "M", "High Risk": "H"}


def big_clean_c(n_funcs=40):
    """A large but perfectly correct C program: many small documented functions, no unsafe calls."""
    parts = ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", ""]
    for i in range(n_funcs):
        parts += [
            f"/* Returns the sum of the first n integers, variant {i}. */",
            f"static int sum_{i}(int n) {{",
            "    int total = 0;",
            "    for (int k = 1; k <= n; k++) {",
            "        if (k % 2 == 0) {",
            "            total += k;",
            "        } else {",
            "            total += k * 2;",
            "        }",
            "    }",
            "    return total;",
            "}",
            "",
            f"/* Allocates, fills and safely releases a buffer, variant {i}. */",
            f"static int buf_{i}(size_t len) {{",
            "    int *b = malloc(len * sizeof(int));",
            "    if (b == NULL) { return -1; }",
            "    for (size_t j = 0; j < len; j++) { b[j] = (int)j; }",
            "    int r = b[len - 1];",
            "    free(b);",
            "    return r;",
            "}",
            "",
        ]
    parts += ["int main(void) {", "    int acc = 0;"]
    parts += [f"    acc += sum_{i}({i + 3}) + buf_{i}({i + 2});" for i in range(n_funcs)]
    parts += ['    printf("%d\\n", acc);', "    return 0;", "}"]
    return "\n".join(parts)


def big_clean_java(n_funcs=40):
    parts = ["import java.util.*;", "", "public class Big {"]
    for i in range(n_funcs):
        parts += [
            f"    /** Safe helper number {i}. */",
            f"    static int helper{i}(List<Integer> xs) {{",
            "        int total = 0;",
            "        for (int x : xs) {",
            "            if (x > 0) { total += x; } else { total -= x; }",
            "        }",
            "        return total;",
            "    }",
            "",
        ]
    parts += ["    public static void main(String[] args) {", "        List<Integer> xs = Arrays.asList(1, 2, 3);"]
    parts += [f"        System.out.println(helper{i}(xs));" for i in range(n_funcs)]
    parts += ["    }", "}"]
    return "\n".join(parts)


DEEP = "int main(void){" + "if(1){" * 300 + "int x=1;" + "}" * 300 + "return 0;}"
MANY_FUNCS = "\n".join(f"int f{i}(int a){{return a+{i};}}" for i in range(3000)) + "\nint main(){return f1(1);}"

# (name, language, code, expected, note)
CASES = [
    # ---------- C: tiny but genuinely vulnerable -> must NOT be Clean ----------
    ("C tiny gets()", "C", "#include <stdio.h>\nint main(){char b[8];gets(b);return 0;}", "H", "classic overflow"),
    ("C tiny use-after-free", "C",
     "#include <stdlib.h>\nint main(){int*p=malloc(4);free(p);*p=1;return 0;}", "H", "UAF"),
    ("C tiny double free", "C",
     "#include <stdlib.h>\nint main(){int*p=malloc(4);free(p);free(p);return 0;}", "H", "double free"),
    ("C tiny leak", "C", "#include <stdlib.h>\nint main(){int*p=malloc(4);p[0]=1;return 0;}", "M", "leak only"),
    ("C strcpy overflow", "C",
     '#include <string.h>\nint main(){char b[4];strcpy(b,"way too long");return 0;}', "H", "strcpy"),
    ("C format string", "C",
     '#include <stdio.h>\nint main(int c,char**v){printf(v[1]);return 0;}', "H", "user-controlled format"),
    ("C null deref", "C", "int main(){int*p=0;*p=1;return 0;}", "H", "null deref"),
    # ---------- C: big but correct -> must NOT be High ----------
    ("C big correct (80 funcs)", "C", big_clean_c(40), "C", "size != risk"),
    # ---------- C: false-positive traps ----------
    ("C gets in comment", "C",
     "#include <stdio.h>\n/* never call gets(buf) */\n// gets(buf);\nint main(){puts(\"gets(x) is unsafe\");return 0;}",
     "C", "gets only in comment/string"),
    ("C malloc+free correct", "C",
     "#include <stdlib.h>\nint main(){int*p=malloc(4);if(!p)return 1;*p=1;free(p);return 0;}", "C", "correct"),
    ("C safe fgets", "C",
     "#include <stdio.h>\nint main(){char b[16];if(fgets(b,sizeof b,stdin))puts(b);return 0;}", "C", "safe"),
    ("C hello world", "C", '#include <stdio.h>\nint main(){printf("hi\\n");return 0;}', "C", "trivial"),
    # ---------- Java ----------
    ("Java SQL injection", "Java",
     'import java.sql.*;\npublic class A{ void q(Connection c,String u) throws Exception{ '
     'Statement s=c.createStatement(); s.executeQuery("SELECT * FROM t WHERE n=\'"+u+"\'"); } }', "H", "CWE-89"),
    ("Java Runtime.exec cmd injection", "Java",
     "public class A{ public static void main(String[] a) throws Exception{ Runtime.getRuntime().exec(a[0]); } }",
     "H", "CWE-78"),
    ("Java hardcoded password", "Java",
     'public class A{ static final String PASSWORD="hunter2"; }', "M", "CWE-259"),
    ("Java empty catch", "Java",
     "public class A{ void f(){ try{ Integer.parseInt(\"x\"); }catch(Exception e){} } }", "M", "swallowed exception"),
    ("Java insecure deserialization", "Java",
     "import java.io.*;\npublic class A{ Object f(InputStream i) throws Exception{ return new ObjectInputStream(i).readObject(); } }",
     "H", "CWE-502"),
    ("Java big correct (40 methods)", "Java", big_clean_java(40), "C", "size != risk"),
    ("Java hello world", "Java", 'public class A{ public static void main(String[] a){ System.out.println("hi"); } }',
     "C", "trivial"),
    # ---------- C++ (project name says Cpp) ----------
    ("C++ vector program", "C",
     "#include <vector>\n#include <iostream>\nint main(){ std::vector<int> v{1,2,3}; for(auto x: v) std::cout<<x; return 0; }",
     "C", "modern C++ submitted as C"),
    ("C++ raw new/delete leak", "C",
     "#include <iostream>\nclass A{ int* p; public: A(){p=new int[10];} };\nint main(){ A* a=new A(); return 0; }",
     "M", "C++ leak (new without delete)"),
    # ---------- Garbage / robustness ----------
    ("empty string", "C", "", "ERR", "no code"),
    ("whitespace only", "C", "   \n\n\t  ", "ERR", "no code"),
    ("only comments", "C", "// nothing here\n/* still nothing */", "C", "no code"),
    ("prose not code", "C", "The quick brown fox jumps over the lazy dog. " * 20, "ERR", "not code"),
    ("syntax error C", "C", "int main( { return ; }}}}", "ERR", "unparseable, must not 500"),
    ("Java given as C", "C", "public class A{ public static void main(String[] a){} }", "C", "wrong language"),
    ("C given as Java", "Java", "#include <stdio.h>\nint main(){return 0;}", "C", "wrong language"),
    ("unicode + BOM + CRLF", "C", "﻿#include <stdio.h>\r\nint main(){ /* héllo ☃ */ printf(\"ü\");\r\nreturn 0;}\r\n", "C", "encoding"),
    ("null bytes", "C", "int main(){\x00 return 0;}", "ERR", "binary junk"),
    ("300-deep nesting", "C", DEEP, "C", "parser recursion; style smell only"),
    ("3000 tiny functions", "C", MANY_FUNCS, "C", "scale / DoS"),
    ("huge single line (2MB)", "C", "int x;" * 350000, "ERR", "size limit -> 413"),
]

BAD_PAYLOADS = [
    ("unknown language", {"code": "int main(){}", "language": "Python"}),
    ("missing code", {"language": "C"}),
    ("wrong types", {"code": 123, "language": "C"}),
    ("bad model_type", {"code": "int main(){return 0;}", "language": "C", "model_type": "../../etc/passwd"}),
]


def post(url, payload, timeout=40):
    req = urllib.request.Request(
        url + "/api/analyze", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:200]
    except Exception as e:  # timeout / connection
        return -1, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    fails = 0
    print(f"{'case':38} {'exp':4} " + " ".join(f"{m:>8}" for m in MODELS) + "  issues  verdict")
    print("-" * 110)
    for name, lang, code, exp, note in CASES:
        row, issues, verdict = {}, "-", "ok"
        status, body = post(args.url, {"code": code, "language": lang, "model_type": "all"})
        if status != 200:
            row = {m: f"HTTP{status}" for m in MODELS}
            if exp != "ERR" or status >= 500 or status < 0:
                verdict = "FAIL"
        else:
            cmp_ = body["risk"].get("comparisons", {})
            row = {m: SHORT.get(cmp_.get(m, {}).get("risk_label", "?"), "?") for m in MODELS}
            issues = len(body["issues"])
            if exp == "ERR":
                verdict = "FAIL(accepted)"
            else:
                wrong = [m for m in MODELS if row[m] != exp]
                # allow one class of slack only for Moderate<->neighbours
                bad = [m for m in wrong if not (exp == "M" and row[m] in "CH" and False)]
                if bad:
                    verdict = "FAIL"
        fails += verdict.startswith("FAIL")
        print(f"{name:38} {exp:4} " + " ".join(f"{row[m]:>8}" for m in MODELS) + f"  {str(issues):>6}  {verdict}   # {note}")

    print("\n-- malformed payloads (must be 4xx, never 5xx) --")
    for name, payload in BAD_PAYLOADS:
        status, body = post(args.url, payload)
        ok = 400 <= status < 500
        fails += not ok
        print(f"{name:38} -> HTTP {status} {'ok' if ok else 'FAIL'}")

    print(f"\n{fails} failing checks")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
