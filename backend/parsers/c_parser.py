"""
IntelliReview — C Source Code Parser
Uses pycparser to build an AST from C source code.
Falls back to regex-based analysis if pycparser fails (e.g., complex macros).
"""
import re
from typing import Dict, Any


def parse_c_code(source: str) -> Dict[str, Any]:
    """
    Parse C source code and extract structural features.
    Returns a dict with counts and metadata.
    """
    result = {
        "language": "C",
        "functions": [],
        "num_functions": 0,
        "num_loops": 0,
        "max_nesting_depth": 0,
        "num_conditionals": 0,
        "num_pointer_uses": 0,
        "num_mallocs": 0,
        "num_frees": 0,
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

    # Try pycparser first, fall back to regex
    try:
        result.update(_parse_with_pycparser(source, lines))
    except Exception as e:
        result["parse_errors"].append(f"pycparser: {str(e)}")
        result.update(_parse_with_regex(source, lines))

    return result


def _parse_with_pycparser(source: str, lines) -> Dict[str, Any]:
    """Use pycparser for accurate AST-based analysis."""
    import pycparser
    from pycparser import c_ast, c_generator

    # pycparser needs clean source — prepend common typedefs
    fake_libc_includes = """
typedef unsigned int size_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef int int32_t;
typedef long long int64_t;
typedef void FILE;
"""
    # Remove comments since pycparser fails on them without GCC
    clean_source = re.sub(r'//.*', '', source)
    clean_source = re.sub(r'/\*[\s\S]*?\*/', '', clean_source)
    
    # Remove #include directives (pycparser can't handle them without gcc)
    clean_source = re.sub(r'#include\s*[<"][^>"]*[>"]', '', clean_source)
    clean_source = re.sub(r'#define\s+\w+.*', '', clean_source)
    clean_source = fake_libc_includes + clean_source

    parser = pycparser.CParser()
    ast = parser.parse(clean_source, filename='<code>')

    visitor = _CASTVisitor(lines)
    visitor.visit(ast)

    return {
        "ast": ast,
        "functions": visitor.functions,
        "num_functions": len(visitor.functions),
        "num_loops": visitor.num_loops,
        "max_nesting_depth": visitor.max_nesting_depth,
        "num_conditionals": visitor.num_conditionals,
        "num_pointer_uses": visitor.num_pointer_uses,
        "num_mallocs": visitor.num_mallocs,
        "num_frees": visitor.num_frees,
        "function_calls": list(visitor.function_calls),
        "function_call_count": len(visitor.function_calls),
        "malloc_lines": visitor.malloc_lines,
        "free_lines": visitor.free_lines,
        "unsafe_calls": visitor.unsafe_calls,
    }


class _CASTVisitor(object):
    """pycparser AST visitor to extract features."""

    UNSAFE_C_FUNCS = {"gets", "strcpy", "strcat", "sprintf", "scanf",
                      "vsprintf", "memcpy", "memmove", "strncpy"}

    def __init__(self, lines):
        self.lines = lines
        self.functions = []
        self.num_loops = 0
        self.max_nesting_depth = 0
        self.num_conditionals = 0
        self.num_pointer_uses = 0
        self.num_mallocs = 0
        self.num_frees = 0
        self.function_calls = set()
        self.malloc_lines = []
        self.free_lines = []
        self.unsafe_calls = []
        self._current_depth = 0
        self._loop_depth = 0

    def visit(self, node):
        from pycparser import c_ast
        if node is None:
            return

        # Function Definitions
        if isinstance(node, c_ast.FuncDef):
            func_name = node.decl.name if node.decl else "unknown"
            coord = node.coord.line if node.coord else 0
            self.functions.append({"name": func_name, "line": coord})
            self._visit_children(node)

        # Loops
        elif isinstance(node, (c_ast.For, c_ast.While, c_ast.DoWhile)):
            self.num_loops += 1
            self._loop_depth += 1
            self.max_nesting_depth = max(self.max_nesting_depth, self._loop_depth)
            self._visit_children(node)
            self._loop_depth -= 1

        # Conditionals
        elif isinstance(node, c_ast.If):
            self.num_conditionals += 1
            self._visit_children(node)

        # Function Calls
        elif isinstance(node, c_ast.FuncCall):
            if isinstance(node.name, c_ast.ID):
                call_name = node.name.name
                self.function_calls.add(call_name)
                line = node.coord.line if node.coord else 0

                if call_name in ("malloc", "calloc", "realloc"):
                    self.num_mallocs += 1
                    self.malloc_lines.append(line)
                elif call_name == "free":
                    self.num_frees += 1
                    self.free_lines.append(line)
                elif call_name in self.UNSAFE_C_FUNCS:
                    self.unsafe_calls.append({
                        "function": call_name,
                        "line": line,
                        "severity": "HIGH" if call_name in {"gets", "strcpy"} else "MEDIUM",
                    })
            self._visit_children(node)

        # Pointer declarations
        elif isinstance(node, c_ast.PtrDecl):
            self.num_pointer_uses += 1
            self._visit_children(node)

        else:
            self._visit_children(node)

    def _visit_children(self, node):
        for _, child in node.children():
            self.visit(child)


def _parse_with_regex(source: str, lines) -> Dict[str, Any]:
    """Fallback regex-based parser when pycparser fails."""
    functions = []
    func_pattern = re.compile(
        r'(?:^|\n)\s*(?:[\w\*]+\s+)+(\w+)\s*\((?:[^)]*)\)\s*\{',
        re.MULTILINE
    )
    for i, line in enumerate(lines, 1):
        # Detect function definitions (simple heuristic)
        if re.match(r'\s*(?:int|void|char|float|double|long|unsigned|static|struct\s+\w+)\s+\w+\s*\(', line):
            name_match = re.search(r'(\w+)\s*\(', line)
            if name_match:
                functions.append({"name": name_match.group(1), "line": i})

    # Count loops
    num_loops = sum(1 for l in lines if re.search(r'\b(for|while|do)\b\s*[\(\{]', l))

    # Nesting depth via brace counting
    depth = 0
    max_depth = 0
    for l in lines:
        depth += l.count('{') - l.count('}')
        max_depth = max(max_depth, depth)

    # Conditionals
    num_conditionals = sum(1 for l in lines if re.search(r'\b(if|switch|else\s+if)\b', l))

    # Pointer uses
    num_pointer_uses = source.count('*')

    # Malloc/free
    malloc_lines = [i+1 for i, l in enumerate(lines) if re.search(r'\bmalloc\b|\bcalloc\b|\brealloc\b', l)]
    free_lines = [i+1 for i, l in enumerate(lines) if re.search(r'\bfree\b', l)]

    # Unsafe functions
    unsafe_funcs = {"gets", "strcpy", "strcat", "sprintf", "vsprintf"}
    unsafe_calls = []
    for i, l in enumerate(lines, 1):
        for f in unsafe_funcs:
            if re.search(rf'\b{f}\s*\(', l):
                unsafe_calls.append({
                    "function": f,
                    "line": i,
                    "severity": "HIGH" if f in {"gets", "strcpy"} else "MEDIUM",
                })

    # Function calls
    all_calls = set(re.findall(r'\b(\w+)\s*\(', source))

    return {
        "functions": functions,
        "num_functions": len(functions),
        "num_loops": num_loops,
        "max_nesting_depth": max_depth,
        "num_conditionals": num_conditionals,
        "num_pointer_uses": num_pointer_uses,
        "num_mallocs": len(malloc_lines),
        "num_frees": len(free_lines),
        "function_calls": list(all_calls),
        "function_call_count": len(all_calls),
        "malloc_lines": malloc_lines,
        "free_lines": free_lines,
        "unsafe_calls": unsafe_calls,
    }
