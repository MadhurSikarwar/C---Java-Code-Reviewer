"""
IntelliReview — Tests: C Analysis Pipeline
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.c_parser import parse_c_code
from analyzers.memory_leak import detect_memory_leaks
from analyzers.unsafe_functions import detect_unsafe_functions
from analyzers.complexity import analyze_complexity
from analyzers.recursion import detect_recursion

# ── Sample C code with known issues ──────────────────────────────
SAMPLE_LEAK = """
#include <stdlib.h>
int main() {
    int *p = malloc(100);
    return 0;
}
"""

SAMPLE_UNSAFE = """
#include <stdio.h>
#include <string.h>
void foo(char *buf) {
    gets(buf);
    strcpy(buf, "hello");
}
"""

SAMPLE_COMPLEX = """
void bigFunc(int a, int b, int c) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int k = 0; k < c; k++) {
                if (i == j) {
                    if (j == k) {
                        if (a > b) {}
                    }
                }
            }
        }
    }
}
"""

SAMPLE_RECURSION = """
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""


def test_c_parser_basic():
    result = parse_c_code("int main() { return 0; }")
    assert result["language"] == "C"
    assert result["lines_of_code"] >= 1
    print("✅ test_c_parser_basic passed")


def test_memory_leak_detected():
    result = parse_c_code(SAMPLE_LEAK)
    mem = detect_memory_leaks(result)
    assert mem["memory_leak_count"] >= 1, f"Expected leak, got {mem}"
    assert any(i["type"] == "MEMORY_LEAK" for i in mem["issues"])
    print("✅ test_memory_leak_detected passed")


def test_no_leak_when_freed():
    code = """
#include <stdlib.h>
int main() {
    int *p = malloc(100);
    free(p);
    return 0;
}
"""
    result = parse_c_code(code)
    mem = detect_memory_leaks(result)
    assert mem["memory_leak_count"] == 0, f"Expected no leak, got {mem['memory_leak_count']}"
    print("✅ test_no_leak_when_freed passed")


def test_unsafe_functions_detected():
    result = parse_c_code(SAMPLE_UNSAFE)
    unsafe = detect_unsafe_functions(result)
    assert unsafe["unsafe_function_count"] >= 1
    funcs = [i["function"] for i in unsafe["issues"]]
    assert any("gets" in f or "strcpy" in f for f in funcs), f"Expected gets/strcpy, got {funcs}"
    print("✅ test_unsafe_functions_detected passed")


def test_complexity_nesting():
    result = parse_c_code(SAMPLE_COMPLEX)
    cx = analyze_complexity(result, SAMPLE_COMPLEX)
    assert cx["max_nesting_depth"] >= 3, f"Expected depth>=3, got {cx['max_nesting_depth']}"
    assert cx["time_complexity"] in ("O(n²)", "O(n³)", "O(n⁴) — consider algorithm redesign")
    print("✅ test_complexity_nesting passed")


def test_recursion_detected():
    result = parse_c_code(SAMPLE_RECURSION)
    rec = detect_recursion(result, SAMPLE_RECURSION)
    assert rec["recursion_count"] >= 1, f"Expected recursion, got {rec}"
    print("✅ test_recursion_detected passed")


if __name__ == "__main__":
    test_c_parser_basic()
    test_memory_leak_detected()
    test_no_leak_when_freed()
    test_unsafe_functions_detected()
    test_complexity_nesting()
    test_recursion_detected()
    print("\n🎉 All C analysis tests passed!")
