"""Suppressions and one-line fixes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import static_analysis
from analyzers.suppress import apply_suppressions

H = "#include <stdio.h>\n#include <string.h>\n"


def fixes(code, lang="C"):
    r = static_analysis((H if lang == "C" else "") + code, lang)
    return {i["type"]: i.get("fix") for i in r["all_issues"]}, r


def test_gets_on_array_is_fixed():
    f, r = fixes("void f(void){\n  char b[8];\n  gets(b);\n}")
    assert f["UNSAFE_FUNCTION"]["after"].strip() == "fgets(b, sizeof(b), stdin);"


def test_gets_on_pointer_gets_no_unsafe_edit():
    f, _ = fixes("void f(char *b){\n  gets(b);\n}")
    assert f["UNSAFE_FUNCTION"] is None       # sizeof(pointer) would be a silent bug


def test_format_string_fix():
    f, _ = fixes("int main(int c,char**v){\n  printf(v[1]);\n  return 0;\n}")
    assert '"%s", v[1]' in f["FORMAT_STRING"]["after"]


def test_strcpy_to_array_is_bounded():
    f, _ = fixes("void f(const char*s){\n  char d[16];\n  strcpy(d, s);\n}")
    assert "snprintf(d, sizeof(d)" in f["UNSAFE_FUNCTION"]["after"]


def test_java_md5_and_empty_catch():
    f, _ = fixes('class A{ byte[] h(byte[] b) throws Exception{ return java.security.MessageDigest.getInstance("MD5").digest(b); }\n'
                 ' void g(){ try{ Integer.parseInt("x"); }catch(Exception e){} } }', "JAVA")
    assert '"SHA-256"' in f["WEAK_CRYPTO"]["after"]
    assert "printStackTrace" in f["EMPTY_CATCH"]["after"]


def test_suppress_same_line_and_next_line_and_typed_and_file():
    code = H + "void f(void){\n  char b[8];\n  gets(b); // intellireview: ignore\n}\n"
    assert not static_analysis(code, "C")["all_issues"]
    code = H + "void f(void){\n  char b[8];\n  // intellireview: ignore[UNSAFE_FUNCTION]\n  gets(b);\n}\n"
    assert not static_analysis(code, "C")["all_issues"]
    code = H + "void f(void){\n  char b[8];\n  // intellireview: ignore[USE_AFTER_FREE]\n  gets(b);\n}\n"
    assert static_analysis(code, "C")["all_issues"]                     # wrong type: still reported
    code = H + "/* intellireview: ignore-file */\nvoid f(void){ char b[8]; gets(b); }\n"
    r = static_analysis(code, "C")
    assert not r["all_issues"] and r["suppressed"]
    assert any("suppressed" in n for n in r["notes"])


def test_no_directive_is_a_noop():
    issues = [{"type": "X", "line": 3}]
    assert apply_suppressions("int main(){}", issues) == (issues, [])
