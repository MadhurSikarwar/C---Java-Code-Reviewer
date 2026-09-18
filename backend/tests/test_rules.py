"""
IntelliReview — rule-engine regression tests (no ML models, no server needed)

Each case is a tiny program plus what a human reviewer would conclude:
    0 = Clean, 1 = Moderate Risk, 2 = High Risk   (what the analyzers alone say; the API applies the same gate)
Run:  python -m pytest tests -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.issue_taxonomy import risk_level_from_issues
from pipeline import static_analysis

HDR = "#include <stdio.h>\n#include <string.h>\n#include <stdlib.h>\n#include <stdarg.h>\n"


def analyse(code, lang="C"):
    return static_analysis((HDR if lang == "C" else "") + code, lang)


def types(res):
    return {i["type"] for i in res["all_issues"]}


# ---------------------------------------------------------------- C: memory safety -----------------------------
C_LEVEL_CASES = [
    # (id, code, expected level, issue type that must be present or None)
    ("uaf", "int f(void){int*p=malloc(4); if(!p)return 0; *p=1; free(p); return *p;}", 2, "USE_AFTER_FREE"),
    ("double_free", "void f(void){char*p=malloc(8); if(!p)return; free(p); free(p);}", 2, "DOUBLE_FREE"),
    ("null_deref", "int f(void){int*p=0; *p=1; return 0;}", 2, "NULL_DEREF"),
    ("gets", "void f(void){char b[8]; gets(b);}", 2, "UNSAFE_FUNCTION"),
    ("strcpy_argv", "int main(int c,char**v){char b[8]; strcpy(b,v[1]); return 0;}", 2, "UNSAFE_FUNCTION"),
    ("format_string_argv", "int main(int c,char**v){printf(v[1]); return 0;}", 2, "FORMAT_STRING"),
    ("system_getenv", 'void f(void){char*c=getenv("X"); system(c);}', 2, "UNSAFE_FUNCTION"),
    ("free_stack", "void f(void){char b[8]; char*p=b; free(p);}", 2, "INVALID_FREE"),
    ("free_offset", "void f(void){char*p=malloc(9); if(!p)return; p++; free(p);}", 2, "INVALID_FREE"),
    ("oob_write_const", "void f(void){char b[10]; b[10]=1;}", 2, "BUFFER_OVERFLOW"),
    ("oob_write_loop", "void f(void){char b[50]; for(int i=0;i<100;i++) b[i]='A';}", 2, "BUFFER_OVERFLOW"),
    ("oob_negative", "void f(void){char b[10]; b[-1]='A';}", 2, "BUFFER_OVERFLOW"),
    ("memset_heap", "void f(void){char*d=malloc(50); if(!d)return; memset(d,'A',100); free(d);}", 2, "BUFFER_OVERFLOW"),
    ("strcpy_big_into_small", "void f(void){char d[50]; char s[100]; memset(s,'C',99); s[99]=0; strcpy(d,s);}", 2, "BUFFER_OVERFLOW"),
    ("fgets_too_big", "void f(void){char b[50]; fgets(b,100,stdin);}", 2, "BUFFER_OVERFLOW"),
    # moderate: reliability findings
    ("leak", "void f(void){char*p=malloc(9); if(!p)return; p[0]=1;}", 1, "MEMORY_LEAK"),
    ("leak_early_return", "int f(int x){char*b=malloc(9); if(!b)return -1; if(x<0)return -2; free(b); return 0;}", 1, "MEMORY_LEAK"),
    ("file_leak", 'int f(const char*p){FILE*f=fopen(p,"r"); if(!f)return -1; return fgetc(f);}', 1, "RESOURCE_LEAK"),
    ("fd_leak", "int f(const char*p){int fd=open(p,0); if(fd<0)return -1; return 1;}", 1, "RESOURCE_LEAK"),
    ("uninit", "int f(int a){int v; if(a){v=1;} return v;}", 1, "UNINITIALIZED_VARIABLE"),
    ("unchecked_alloc", "void f(void){char*p=malloc(9); p[0]=1; free(p);}", 1, "UNCHECKED_ALLOC"),
    ("scanf_s", 'void f(void){char n[16]; scanf("%s",n);}', 1, "UNSAFE_FUNCTION"),
    ("possible_null_paths", "void f(int a){int x; int*p=NULL; if(a){p=&x;} *p=1;}", 1, "NULL_DEREF"),
    # clean: correct code that a naive checker flags
    ("clean_malloc_free", "int f(void){int*p=malloc(4); if(!p)return 1; *p=1; free(p); return 0;}", 0, None),
    ("clean_both_branches", "void f(int a){char*p=malloc(3); if(!p)return; if(a) free(p); else free(p);}", 0, None),
    ("clean_goto_cleanup", "int f(int a){char*p=malloc(3); if(!p)return -1; if(a)goto out; a=2; out: free(p); return a;}", 0, None),
    ("clean_ownership_transfer", "struct S{char*d;};\nvoid f(struct S*s){char*p=malloc(3); if(!p)return; s->d=p;}", 0, None),
    ("clean_file", 'int f(const char*p){FILE*f=fopen(p,"r"); if(!f)return -1; int c=fgetc(f); fclose(f); return c;}', 0, None),
    ("clean_fd", "int f(const char*p){int fd=open(p,0); if(fd<0)return -1; close(fd); return 1;}", 0, None),
    ("clean_strcpy_literal", 'void f(void){char b[16]; strcpy(b,"ok"); puts(b);}', 0, None),
    ("clean_strcpy_sized", "void f(void){char d[100]; char s[50]; memset(s,'C',49); s[49]=0; strcpy(d,s);}", 0, None),
    ("clean_memcpy_sizeof", "void f(char*s){char d[20]; memcpy(d,s,sizeof(d));}", 0, None),
    ("clean_loop_in_bounds", "void f(void){char b[10]; for(int i=0;i<10;i++) b[i]=0;}", 0, None),
    ("clean_guarded_index", "void f(void){char b[10]; for(int i=0;i<100;i++){ if(i<10) b[i]='A'; }}", 0, None),
    ("clean_system_literal", 'void f(void){system("ls");}', 0, None),
    ("clean_scanf_addr", 'int f(void){int x; scanf("%d",&x); return x;}', 1, "UNSAFE_FUNCTION"),  # scanf itself is MEDIUM
    ("clean_recursion", "int fact(int n){ if(n<=1) return 1; return n*fact(n-1); }", 0, None),
    ("clean_while1_break", "int f(void){int i=0; while(1){ if(i>3) break; i++; } return i;}", 0, None),
    ("clean_const_branch", "static int off=0;\nvoid f(void){char*d=NULL; if(off){} else { d=malloc(9); if(!d)exit(1);} free(d);}", 0, None),
    ("clean_do_while0", "void f(void){char*d=malloc(9); if(!d)return; do{ free(d); }while(0);}", 0, None),
    ("clean_va_wrapper", "void logmsg(const char*fmt,...){va_list ap; va_start(ap,fmt); vprintf(fmt,ap); va_end(ap);}", 0, None),
    ("clean_comment_string", 'void f(void){ /* gets(b) */ puts("strcpy(a,b) is unsafe"); }', 0, None),
]


@pytest.mark.parametrize("name,code,level,must_have", C_LEVEL_CASES, ids=[c[0] for c in C_LEVEL_CASES])
def test_c_rule_level(name, code, level, must_have):
    res = analyse(code)
    assert res["parse_result"]["analysis_mode"] == "ast", res["parse_result"]["parse_errors"]
    got = risk_level_from_issues(res["all_issues"])
    assert got == level, f"{name}: expected level {level}, got {got}: " \
                         f"{[(i['type'], i['severity'], i['line']) for i in res['all_issues']]}"
    if must_have:
        assert must_have in types(res), f"{name}: expected {must_have}, got {types(res)}"


def test_line_numbers_are_original_lines():
    code = "\n\n/* c */\nint f(void){\n  int*p=malloc(4);\n  free(p);\n  free(p);\n  return 0;\n}\n"
    res = static_analysis(code, "C")
    df = [i for i in res["all_issues"] if i["type"] == "DOUBLE_FREE"]
    assert df and df[0]["line"] == 7, df


def test_unknown_typedefs_do_not_break_parsing():
    res = static_analysis("void f(HANDLE h, pthread_t t){ my_type_t x = 1; gets((char*)&x); }", "C")
    assert res["parse_result"]["analysis_mode"] == "ast"
    assert "UNSAFE_FUNCTION" in types(res)


def test_cpp_is_analysed_heuristically_with_note():
    res = static_analysis("#include <iostream>\nclass A{ int*p; public: A(){p=new int[10];} };\nint main(){A*a=new A(); return 0;}", "C")
    assert res["parse_result"]["analysis_mode"] == "heuristic"
    assert "MEMORY_LEAK" in types(res)


# ---------------------------------------------------------------- Java --------------------------------------------
JAVA_CASES = [
    ("sqli_readline", 'import java.sql.*;\nclass A{ void q(Connection c) throws Exception{ java.io.BufferedReader r=new java.io.BufferedReader(new java.io.InputStreamReader(System.in)); String u=r.readLine(); Statement s=c.createStatement(); s.executeQuery("SELECT * FROM t WHERE n=\'"+u+"\'"); } }', 2, "SQL_INJECTION"),
    ("sql_constant", 'import java.sql.*;\nclass A{ void q(Connection c) throws Exception{ String t="users"; Statement s=c.createStatement(); s.executeQuery("SELECT * FROM "+t); } }', 0, None),
    ("cmd_tainted", 'class A{ void f() throws Exception{ String d=System.getenv("C"); Runtime.getRuntime().exec(d); } }', 2, "COMMAND_INJECTION"),
    ("exec_literal", 'class A{ void f() throws Exception{ Runtime.getRuntime().exec("ls"); } }', 0, None),
    ("deser", "import java.io.*;\nclass A{ Object f(InputStream i) throws Exception{ return new ObjectInputStream(i).readObject(); } }", 2, "INSECURE_DESERIALIZATION"),
    ("xss_branch_taint", 'class A{ public void f(javax.servlet.http.HttpServletRequest request, javax.servlet.http.HttpServletResponse response) throws Exception{ String data; if(5==5){ data=request.getParameter("n"); } else { data=null; } if(data!=null){ response.getWriter().println("<b>"+data); } } }', 2, "XSS"),
    ("path_param_not_flagged", "class A{ String rd(String path) throws Exception{ try(java.io.BufferedReader r=new java.io.BufferedReader(new java.io.FileReader(path))){ return r.readLine(); } } }", 0, None),
    ("hardcoded_pw", 'class A{ static final String PASSWORD="hunter2"; }', 1, "HARDCODED_CREDENTIAL"),
    ("empty_catch", 'class A{ void f(){ try{ Integer.parseInt("x"); }catch(Exception e){} } }', 1, "EMPTY_CATCH"),
    ("md5", 'class A{ byte[] f(byte[] b) throws Exception{ return java.security.MessageDigest.getInstance("MD5").digest(b); } }', 1, "WEAK_CRYPTO"),
    ("resource_leak", "class A{ int f(String p) throws Exception{ java.io.FileInputStream f=new java.io.FileInputStream(p); return f.read(); } }", 1, "RESOURCE_LEAK"),
    ("twr_ok", "class A{ int f(String p) throws Exception{ try(java.io.FileInputStream f=new java.io.FileInputStream(p)){ return f.read(); } } }", 0, None),
    ("recursion_not_risky", "class A{ static int fact(int n){ return n<=1?1:n*fact(n-1); } }", 0, None),
    ("name_in_string_not_recursion", 'class A{ void bad(){ System.out.println("bad() called"); } }', 0, None),
]


@pytest.mark.parametrize("name,code,level,must_have", JAVA_CASES, ids=[c[0] for c in JAVA_CASES])
def test_java_rule_level(name, code, level, must_have):
    res = static_analysis(code, "JAVA")
    assert res["parse_result"]["analysis_mode"] == "ast", res["parse_result"]["parse_errors"]
    got = risk_level_from_issues(res["all_issues"])
    assert got == level, f"{name}: expected {level}, got {got}: {[(i['type'], i['severity']) for i in res['all_issues']]}"
    if must_have:
        assert must_have in types(res)
    if name in ("recursion_not_risky", "name_in_string_not_recursion"):
        assert "RECURSION" not in types(res) or name == "recursion_not_risky"


# ---------------------------------------------------------------- feature vector -----------------------------------
def test_feature_vector_shape_and_v4_features_are_computed():
    res = analyse("void f(void){char b[8]; gets(b);}")
    assert len(res["feature_vector"]) == 36
    assert res["feature_vector"][-2] >= 1            # dedup_path_occurrences (was always 0 at inference before)
    assert res["feature_vector"][-1] > 0             # scaled_severity_score


def test_large_correct_program_stays_clean():
    fns = "\n".join(f"static int f{i}(int n){{int t=0; for(int k=0;k<n;k++){{ if(k%2==0){{t+=k;}}else{{t-=1;}} }} return t;}}"
                    for i in range(300))
    res = analyse(fns)
    assert risk_level_from_issues(res["all_issues"]) == 0


def test_one_unchecked_allocation_is_one_finding():
    res = analyse("void f(int n){ int*d=malloc(40); for(int i=0;i<n;i++) d[i]=i; d[0]=1; free(d); }")
    assert [i["type"] for i in res["all_issues"]].count("UNCHECKED_ALLOC") == 1
