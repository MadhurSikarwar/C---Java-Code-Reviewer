"""
IntelliReview V3 — Data Flow Analysis (DFA)
===========================================
Tracks variable states and mutations across the CFG strictly via graph evaluation.
Detects Uninitialized Variable Usage and Infinite Loops natively on paths constraints.
"""
from typing import Dict, Any, List
from copy import deepcopy

class DataFlowWarning:
    def __init__(self, issue_type: str, severity: str, message: str, node_id: int, line_no: int = -1):
        self.type = issue_type
        self.severity = severity
        self.message = message
        self.node_id = node_id
        self.line = line_no

    def to_dict(self):
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "line": self.line,
            "node_id": self.node_id
        }

def analyze_data_flow(source: str, cfg_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    warnings = []
    metrics = {
        "infinite_loop_risks": 0,
        "uninitialized_vars_used": 0
    }

    for cfg in cfg_list:
        nodes = cfg.get("nodes", {})
        entry_id = cfg.get("entry_id")
        
        if not nodes or entry_id is None:
            continue
            
        stack = [(entry_id, {}, {n: 0 for n in nodes})]
        
        # Determine all local variables declared inside the function body 
        # (excludes parameters which are implicitly initialized)
        local_vars = set()
        for nid, node in nodes.items():
            local_vars.update(node.get("decls", []))
            
        while stack:
            curr_id, state_in, visits = stack.pop()
            if visits[curr_id] >= 3:
                continue
                
            visits[curr_id] += 1
            node = nodes[curr_id]
            node_line = node.get("line", -1)
            state_out = deepcopy(state_in)
            
            assigns = node.get("assigns", [])
            uses = node.get("uses", [])
            
            # Uninitialized variable check
            for u in uses:
                if u in local_vars and state_out.get(u) != "INITIALIZED":
                    metrics["uninitialized_vars_used"] += 1
                    warnings.append(DataFlowWarning(
                        "UNINITIALIZED_VARIABLE", "CRITICAL",
                        f"Uninitialized Variable: Local variable '{u}' is read before assignment on this execution path.",
                        curr_id, node_line
                    ))
                    # Prevent duplicate report logging on same path
                    state_out[u] = "INITIALIZED"
            
            for a in assigns:
                state_out[a] = "INITIALIZED"
                
            # Infinite Loop heuristic over AST branches
            if node.get("name") == "While Cond":
                if not uses:
                    metrics["infinite_loop_risks"] += 1
                    warnings.append(DataFlowWarning(
                        "INFINITE_LOOP", "HIGH", 
                        f"Infinite Loop Risk: loop condition is hardcoded or has no tracked scalar bounds variables.", 
                        curr_id, node_line
                    ))
            
            out_edges = node.get("out_edges", [])
            for target_id, _ in out_edges:
                stack.append((target_id, deepcopy(state_out), deepcopy(visits)))
                
    return {
        "warnings": [w.to_dict() for w in warnings],
        "metrics": metrics
    }
