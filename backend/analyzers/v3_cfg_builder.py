"""
IntelliReview V3 — Core Control Flow Graph (CFG) Builder
=========================================================
Converts a pycparser AST into a true Control Flow Graph.
Each block contains sequential statements, ending in a branch or return.
Nodes: Basic Blocks
Edges: True/False/Unconditional jumps

This forms the required foundation for Data Flow Analysis and Path-Sensitive checks.
"""
from typing import List, Dict, Any
from pycparser import c_ast

class CFGNode:
    def __init__(self, node_id: int, name: str = "Block"):
        self.id = node_id
        self.name = name
        self.statements = []      # List of raw PycParser AST nodes in this block
        self.out_edges = []       # List of (target_node_id, edge_condition)
        self.is_return = False
    
    def add_stmt(self, stmt):
        self.statements.append(stmt)
        
    def add_edge(self, target_id: int, condition: str = "unconditional"):
        self.out_edges.append((target_id, condition))

    def to_dict(self):
        allocs = []
        frees = []
        assigns = []
        uses = []
        returns = []
        decls = []

        def get_min_line(stmt_list):
            min_idx = -1
            for st in stmt_list:
                if hasattr(st, 'coord') and getattr(st, 'coord', None) is not None:
                    try:
                        line_no = int(st.coord.line)
                        if min_idx == -1 or line_no < min_idx:
                            min_idx = line_no
                    except: pass
            return min_idx

        class NodeInspector(c_ast.NodeVisitor):
            def visit_FuncCall(self, n):
                if isinstance(n.name, c_ast.ID):
                    if n.name.name == "free":
                        if n.args and n.args.exprs:
                            arg = n.args.exprs[0]
                            if isinstance(arg, c_ast.ID):
                                frees.append(arg.name)
                for _, child in n.children():
                    self.visit(child)

            def visit_Assignment(self, n):
                if isinstance(n.lvalue, c_ast.ID):
                    assigned_var = n.lvalue.name
                    assigns.append(assigned_var)
                    if isinstance(n.rvalue, c_ast.FuncCall) and getattr(n.rvalue.name, 'name', '') in ("malloc", "calloc"):
                        allocs.append(assigned_var)
                self.visit(n.rvalue)
                
            def visit_Decl(self, n):
                decls.append(n.name)
                if n.init:
                    assigns.append(n.name)
                    if isinstance(n.init, c_ast.FuncCall) and getattr(n.init.name, 'name', '') in ("malloc", "calloc"):
                        allocs.append(n.name)
                    self.visit(n.init)
                    
            def visit_ID(self, n):
                uses.append(n.name)
                
            def visit_Return(self, n):
                if n.expr:
                    if isinstance(n.expr, c_ast.ID):
                        returns.append(n.expr.name)
                    self.visit(n.expr)

        inspector = NodeInspector()
        for stmt in self.statements:
            inspector.visit(stmt)

        return {
            "id": self.id,
            "line": get_min_line(self.statements),
            "name": self.name,
            "num_stmts": len(self.statements),
            "out_edges": self.out_edges,
            "is_return": self.is_return,
            "allocs": list(set(allocs)),
            "frees": list(set(frees)),
            "assigns": list(set(assigns)),
            "uses": list(set(uses)),
            "returns": list(set(returns)),
            "decls": list(set(decls))
        }

class CFGBuilder(c_ast.NodeVisitor):
    def __init__(self):
        self.nodes: Dict[int, CFGNode] = {}
        self.node_counter = 0
        self.current_node_id = None
        self.entry_node_id = None
        
        # Keep track of loop targets for 'break' and 'continue'
        self.loop_break_targets = []
        self.loop_continue_targets = []
        
        self.function_calls = []

    def _create_node(self, name: str = "Block") -> int:
        nid = self.node_counter
        self.node_counter += 1
        self.nodes[nid] = CFGNode(nid, name)
        return nid

    def _add_edge(self, src: int, dst: int, condition: str = "unconditional"):
        if src is not None and dst is not None:
            self.nodes[src].add_edge(dst, condition)
            
    def _extract_func_calls(self, node):
        """Helper to find all function calls inside a statement."""
        if isinstance(node, c_ast.FuncCall):
            if isinstance(node.name, c_ast.ID):
                self.function_calls.append(node.name.name)
        for _, child in node.children():
            self._extract_func_calls(child)

    def build_from_func(self, func_ast: c_ast.FuncDef) -> Dict[str, Any]:
        """Main entry point. Builds a CFG for a single C function."""
        self.nodes = {}
        self.node_counter = 0
        self.function_calls = []
        
        entry = self._create_node(name=f"Entry: {func_ast.decl.name}")
        self.entry_node_id = entry
        self.current_node_id = entry
        
        if func_ast.body:
            self.visit(func_ast.body)
            
        # Add an implicit exit node if the last node doesn't return
        exit_node = self._create_node(name="Exit")
        if self.current_node_id is not None and not self.nodes[self.current_node_id].is_return:
            self._add_edge(self.current_node_id, exit_node)
            
        # Hook up all loose returns to the Exit node
        for nid, node in self.nodes.items():
            if node.is_return and nid != exit_node:
                self._add_edge(nid, exit_node, "return")

        return {
            "function": func_ast.decl.name,
            "entry_id": self.entry_node_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "function_calls": self.function_calls
        }

    def visit_Compound(self, node):
        if node.block_items:
            for item in node.block_items:
                self.visit(item)

    def visit_If(self, node):
        # Current block evaluates the condition
        cond_node = self.current_node_id
        if node.cond:
            self.nodes[cond_node].add_stmt(node.cond)
            self._extract_func_calls(node.cond)

        # Create branches
        true_block = self._create_node(name="If True")
        false_block = self._create_node(name="If False")
        merge_block = self._create_node(name="If Merge")

        # True path
        self._add_edge(cond_node, true_block, "True")
        self.current_node_id = true_block
        if node.iftrue:
            self.visit(node.iftrue)
        if self.current_node_id is not None and not self.nodes[self.current_node_id].is_return:
            self._add_edge(self.current_node_id, merge_block)

        # False path
        self._add_edge(cond_node, false_block, "False")
        self.current_node_id = false_block
        if node.iffalse:
            self.visit(node.iffalse)
        if self.current_node_id is not None and not self.nodes[self.current_node_id].is_return:
            self._add_edge(self.current_node_id, merge_block)

        self.current_node_id = merge_block

    def visit_While(self, node):
        cond_blk = self._create_node(name="While Cond")
        body_blk = self._create_node(name="While Body")
        exit_blk = self._create_node(name="While Exit")

        self.loop_break_targets.append(exit_blk)
        self.loop_continue_targets.append(cond_blk)

        # Enter loop
        self._add_edge(self.current_node_id, cond_blk)
        self.current_node_id = cond_blk
        if node.cond:
            self.nodes[cond_blk].add_stmt(node.cond)
            self._extract_func_calls(node.cond)

        self._add_edge(cond_blk, body_blk, "True")
        self._add_edge(cond_blk, exit_blk, "False")

        # Body
        self.current_node_id = body_blk
        if node.stmt:
            self.visit(node.stmt)
        
        # Back edge
        if self.current_node_id is not None and not self.nodes[self.current_node_id].is_return:
            self._add_edge(self.current_node_id, cond_blk, "loop_back")

        self.loop_break_targets.pop()
        self.loop_continue_targets.pop()
        self.current_node_id = exit_blk

    def visit_For(self, node):
        init_blk = self._create_node(name="For Init")
        cond_blk = self._create_node(name="For Cond")
        body_blk = self._create_node(name="For Body")
        next_blk = self._create_node(name="For Next")
        exit_blk = self._create_node(name="For Exit")

        self.loop_break_targets.append(exit_blk)
        self.loop_continue_targets.append(next_blk)

        # Init
        self._add_edge(self.current_node_id, init_blk)
        self.current_node_id = init_blk
        if node.init:
            self.nodes[init_blk].add_stmt(node.init)
            self._extract_func_calls(node.init)

        # Cond
        self._add_edge(init_blk, cond_blk)
        self.current_node_id = cond_blk
        if node.cond:
            self.nodes[cond_blk].add_stmt(node.cond)
            self._extract_func_calls(node.cond)
        
        self._add_edge(cond_blk, body_blk, "True")
        self._add_edge(cond_blk, exit_blk, "False")

        # Body
        self.current_node_id = body_blk
        if node.stmt:
            self.visit(node.stmt)
        if self.current_node_id is not None and not self.nodes[self.current_node_id].is_return:
            self._add_edge(self.current_node_id, next_blk)

        # Next (e.g. i++)
        self.current_node_id = next_blk
        if node.next:
            self.nodes[next_blk].add_stmt(node.next)
            self._extract_func_calls(node.next)
        self._add_edge(next_blk, cond_blk, "loop_back")

        self.loop_break_targets.pop()
        self.loop_continue_targets.pop()
        self.current_node_id = exit_blk

    def visit_Break(self, node):
        if self.loop_break_targets:
            self._add_edge(self.current_node_id, self.loop_break_targets[-1], "break")
            self.current_node_id = None # Unreachable after break
            
    def visit_Continue(self, node):
        if self.loop_continue_targets:
            self._add_edge(self.current_node_id, self.loop_continue_targets[-1], "continue")
            self.current_node_id = None

    def visit_Return(self, node):
        if self.current_node_id is not None:
            self.nodes[self.current_node_id].add_stmt(node)
            self._extract_func_calls(node)
            self.nodes[self.current_node_id].is_return = True
            self.current_node_id = None

    def generic_visit(self, node):
        # For standard linear statements (assignment, decl, function call)
        if self.current_node_id is not None:
            self.nodes[self.current_node_id].add_stmt(node)
            self._extract_func_calls(node)


def build_cfg_for_file(ast) -> List[Dict[str, Any]]:
    """Builds a list of CFGs, one for each function in the C file."""
    if ast is None:
        return []

    cfgs = []
    
    class FuncFinder(c_ast.NodeVisitor):
        def visit_FuncDef(self, node):
            builder = CFGBuilder()
            cfgs.append(builder.build_from_func(node))
            
    FuncFinder().visit(ast)
    return cfgs
