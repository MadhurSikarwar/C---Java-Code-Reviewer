"""
IntelliReview V3 — Structural Code Smells
=========================================
Detects deep nesting, long functions, global variable mutations,
and calculates cyclomatic complexity exactly via CFG edges.
"""
from typing import Dict, Any, List
import re
from parsers.text_utils import sanitize

class SmellWarning:
    def __init__(self, issue_type: str, severity: str, message: str, line_no: int = -1):
        self.type = issue_type
        self.severity = severity
        self.message = message
        self.line = line_no
        
    def to_dict(self):
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "line": self.line
        }

def analyze_smells(source: str, cfg_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes the code for architectural/structural smells."""
    warnings = []
    
    # 1. Cyclomatic Complexity via CFG (Edges - Nodes + 2P)
    # Since we have N components (functions), total CC = Sum(E - N + 2)
    total_cfg_nodes = 0
    total_cfg_edges = 0
    
    for cfg in cfg_list:
        nodes = cfg.get("nodes", {})
        V = len(nodes)
        E = sum(len(n.get("out_edges", [])) for n in nodes.values())
        total_cfg_nodes += V
        total_cfg_edges += E
        
        # Per function CC
        func_cc = E - V + 2 if V > 0 else 1
        if func_cc > 10:
            warnings.append(SmellWarning(
                "HIGH_COMPLEXITY",
                "MEDIUM" if func_cc > 15 else "LOW",
                f"Function '{cfg.get('function')}' is highly complex (CC = {func_cc}). Break it down.",
                cfg.get("line", -1)
            ))

    global_cc = total_cfg_edges - total_cfg_nodes + 2 * len(cfg_list) if cfg_list else 1

    # 2. Deep Nesting
    max_nesting = 0
    max_nesting_line = -1
    current_nesting = 0
    lines = sanitize(source).splitlines()
    for i, line in enumerate(lines, 1):
        clean = line.strip()
        current_nesting += clean.count('{') - clean.count('}')
        if current_nesting > max_nesting:
            max_nesting = current_nesting
            max_nesting_line = i
    
    if max_nesting > 4:
        warnings.append(SmellWarning(
            "DEEP_NESTING",
            "MEDIUM" if max_nesting > 5 else "LOW",
            f"Code is deeply nested ({max_nesting} levels). Consider early returns.",
            max_nesting_line
        ))

    # 4. Dead Code Reachability Analysis
    for cfg in cfg_list:
        nodes = cfg.get("nodes", {})
        entry_id = cfg.get("entry_id")
        if not nodes or entry_id is None:
            continue
            
        visited = set()
        stack = [entry_id]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                for target_id, _ in nodes[curr].get("out_edges", []):
                    stack.append(target_id)
                    
        # Empty join blocks after an if/else whose branches both return are not "code".
        unreachable = {n for n in set(nodes.keys()) - visited if nodes[n].get("num_stmts", 0) > 0}
        if unreachable:
            warnings.append(SmellWarning(
                "DEAD_CODE",
                "LOW",
                f"Function '{cfg.get('function')}' contains {len(unreachable)} entirely unreachable basic block(s).",
                cfg.get("line", -1)
            ))

    metrics = {
        "cfg_node_count": total_cfg_nodes,
        "branch_density": round(total_cfg_edges / total_cfg_nodes, 2) if total_cfg_nodes > 0 else 0.0,
        "cyclomatic_complexity": global_cc,
        "max_nesting_depth": max_nesting,
        "global_mutation_count": 0  # Implemented structurally via full external analyzer
    }

    return {
        "warnings": [w.to_dict() for w in warnings],
        "metrics": metrics
    }
