"""
IntelliReview — Java Source Code Parser
Uses javalang to build an AST from Java source code.
Falls back to regex-based analysis if javalang fails.
"""
import re
from typing import Dict, Any


def parse_java_code(source: str) -> Dict[str, Any]:
    """
    Parse Java source code and extract structural features.
    Returns a dict with counts and metadata.
    """
    result = {
        "language": "Java",
        "functions": [],
        "num_functions": 0,
        "num_loops": 0,
        "max_nesting_depth": 0,
        "num_conditionals": 0,
        "num_pointer_uses": 0,   # 0 for Java (no raw pointers)
        "num_mallocs": 0,         # 0 for Java (GC managed)
        "num_frees": 0,           # 0 for Java
        "function_calls": [],
        "function_call_count": 0,
        "lines_of_code": 0,
        "malloc_lines": [],
        "free_lines": [],
        "unsafe_calls": [],
        "parse_errors": [],
    }

    lines = source.splitlines()
    result["lines_of_code"] = sum(1 for l in lines if l.strip() and not l.strip().startswith("//"))

    try:
        result.update(_parse_with_javalang(source, lines))
    except Exception as e:
        result["parse_errors"].append(f"javalang: {str(e)}")
        result.update(_parse_java_with_regex(source, lines))

    return result


def _parse_with_javalang(source: str, lines) -> Dict[str, Any]:
    """Use javalang for accurate AST-based Java analysis."""
    import javalang

    tree = javalang.parse.parse(source)

    functions = []
    num_loops = 0
    num_conditionals = 0
    function_calls = set()
    unsafe_calls = []
    max_loop_depth = 0

    # Java unsafe patterns
    UNSAFE_JAVA = {
        "exec": "Runtime.exec() — OS command injection risk",
        "eval": "Potential code injection",
        "readLine": "Unvalidated input — potential injection",
    }

    def walk(node, loop_depth=0):
        nonlocal num_loops, num_conditionals, max_loop_depth

        if node is None:
            return

        if isinstance(node, javalang.tree.MethodDeclaration):
            line = node.position.line if node.position else 0
            functions.append({"name": node.name, "line": line})

        elif isinstance(node, (javalang.tree.ForStatement,
                               javalang.tree.WhileStatement,
                               javalang.tree.DoStatement,
                               javalang.tree.EnhancedForStatement)):
            num_loops += 1
            new_depth = loop_depth + 1
            max_loop_depth = max(max_loop_depth, new_depth)
            _walk_children(node, new_depth)
            return  # already walked children

        elif isinstance(node, javalang.tree.IfStatement):
            num_conditionals += 1

        elif isinstance(node, javalang.tree.MethodInvocation):
            if node.member:
                function_calls.add(node.member)
                if node.member in UNSAFE_JAVA:
                    line = node.position.line if node.position else 0
                    unsafe_calls.append({
                        "function": node.member,
                        "line": line,
                        "severity": "HIGH",
                        "reason": UNSAFE_JAVA[node.member],
                    })

                # String concatenation in loops (potential O(n²))
                if node.member == "concat" and loop_depth > 0:
                    unsafe_calls.append({
                        "function": "String.concat() inside loop",
                        "line": node.position.line if node.position else 0,
                        "severity": "MEDIUM",
                        "reason": "Use StringBuilder instead of string concatenation in loops",
                    })

        _walk_children(node, loop_depth)

    def _walk_children(node, loop_depth):
        """Iterate over all children of any javalang node."""
        if hasattr(node, 'children'):
            for child_group in node.children:
                if isinstance(child_group, list):
                    for child in child_group:
                        if isinstance(child, javalang.tree.Node):
                            walk(child, loop_depth)
                elif isinstance(child_group, javalang.tree.Node):
                    walk(child_group, loop_depth)

    walk(tree)

    return {
        "functions": functions,
        "num_functions": len(functions),
        "num_loops": num_loops,
        "max_nesting_depth": max_loop_depth,
        "num_conditionals": num_conditionals,
        "num_pointer_uses": 0,
        "num_mallocs": 0,
        "num_frees": 0,
        "function_calls": list(function_calls),
        "function_call_count": len(function_calls),
        "malloc_lines": [],
        "free_lines": [],
        "unsafe_calls": unsafe_calls,
    }


def _parse_java_with_regex(source: str, lines) -> Dict[str, Any]:
    """Fallback regex-based Java parser."""
    functions = []
    for i, line in enumerate(lines, 1):
        if re.search(r'\b(public|private|protected|static).*\w+\s+\w+\s*\(', line):
            name_match = re.search(r'(\w+)\s*\(', line)
            if name_match and name_match.group(1) not in ('if', 'for', 'while', 'catch', 'switch'):
                functions.append({"name": name_match.group(1), "line": i})

    num_loops = sum(1 for l in lines if re.search(r'\b(for|while|do)\b\s*[\(\{]', l))

    depth = 0
    max_depth = 0
    for l in lines:
        depth += l.count('{') - l.count('}')
        max_depth = max(max_depth, depth)

    num_conditionals = sum(1 for l in lines if re.search(r'\b(if|switch)\b', l))

    all_calls = set(re.findall(r'\.(\w+)\s*\(', source))
    unsafe_funcs = {"exec", "eval", "readLine"}
    unsafe_calls = []
    for i, l in enumerate(lines, 1):
        for f in unsafe_funcs:
            if re.search(rf'\.{f}\s*\(', l):
                unsafe_calls.append({"function": f, "line": i, "severity": "HIGH"})

    return {
        "functions": functions,
        "num_functions": len(functions),
        "num_loops": num_loops,
        "max_nesting_depth": max_depth,
        "num_conditionals": num_conditionals,
        "num_pointer_uses": 0,
        "num_mallocs": 0,
        "num_frees": 0,
        "function_calls": list(all_calls),
        "function_call_count": len(all_calls),
        "malloc_lines": [],
        "free_lines": [],
        "unsafe_calls": unsafe_calls,
    }
