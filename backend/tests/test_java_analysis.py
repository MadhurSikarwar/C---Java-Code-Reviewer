"""
IntelliReview — Tests: Java Analysis Pipeline
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.java_parser import parse_java_code
from analyzers.unsafe_functions import detect_unsafe_functions
from analyzers.complexity import analyze_complexity
from analyzers.recursion import detect_recursion

SAMPLE_JAVA_RECURSION = """
public class Fib {
    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n-1) + fibonacci(n-2);
    }
}
"""

SAMPLE_JAVA_COMPLEX = """
public class Sorter {
    public void process(int a, int b, int c) {
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                if (i > j) {
                    if (j > c) {
                        System.out.println(i + j);
                    }
                }
            }
        }
    }
}
"""

SAMPLE_JAVA_CLEAN = """
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
"""


def test_java_parser_basic():
    result = parse_java_code(SAMPLE_JAVA_CLEAN)
    assert result["language"] == "Java"
    assert result["num_pointer_uses"] == 0  # Java has no pointers
    assert result["num_mallocs"] == 0       # Java has no malloc
    print("✅ test_java_parser_basic passed")


def test_java_functions_detected():
    result = parse_java_code(SAMPLE_JAVA_RECURSION)
    assert result["num_functions"] >= 1, f"Expected functions, got {result['num_functions']}"
    print("✅ test_java_functions_detected passed")


def test_java_recursion_detected():
    result = parse_java_code(SAMPLE_JAVA_RECURSION)
    rec = detect_recursion(result, SAMPLE_JAVA_RECURSION)
    assert rec["recursion_count"] >= 1, f"Expected recursion, got {rec}"
    print("✅ test_java_recursion_detected passed")


def test_java_complexity_nesting():
    result = parse_java_code(SAMPLE_JAVA_COMPLEX)
    cx = analyze_complexity(result, SAMPLE_JAVA_COMPLEX)
    assert cx["max_nesting_depth"] >= 2
    print("✅ test_java_complexity_nesting passed")


def test_java_clean_code():
    result = parse_java_code(SAMPLE_JAVA_CLEAN)
    unsafe = detect_unsafe_functions(result)
    assert unsafe["unsafe_function_count"] == 0
    print("✅ test_java_clean_code passed")


if __name__ == "__main__":
    test_java_parser_basic()
    test_java_functions_detected()
    test_java_recursion_detected()
    test_java_complexity_nesting()
    test_java_clean_code()
    print("\n🎉 All Java analysis tests passed!")
