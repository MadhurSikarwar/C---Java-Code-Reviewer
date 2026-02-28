"""
IntelliReview V3 — Path-Sensitive Pointer State Tracker
=========================================================
Performs Data Flow Analysis (DFA) on the Control Flow Graph.
Tracks pointer variables through UNINITIALIZED -> ALLOCATED -> FREED -> NULL.
Detects advanced vulnerabilities:
 - Use-After-Free
 - Double Free (on specific branches)
 - Conditional Memory Leaks (malloc'd but not freed on ALL exit paths)
"""
from typing import Dict, List, Set, Any
from copy import deepcopy

# Pointer States
STATE_UNINIT   = "UNINITIALIZED"
STATE_ALLOC    = "ALLOCATED"
STATE_FREED    = "FREED"
STATE_NULL     = "NULL"
STATE_UNKNOWN  = "UNKNOWN"

class PathWarning:
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

class PointerStateTracker:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.nodes = cfg.get("nodes", {})
        self.entry_id = cfg.get("entry_id")
        self.warnings: List[PathWarning] = []
        
        self.leaked_on_exits: Dict[str, Set[int]] = {}
        self.exit_nodes: Set[int] = set()
        self.max_visits_per_node = 3

    def analyze(self) -> Dict[str, Any]:
        """
        Simulates execution. Tracks allocations, frees, and usages over all possible execution paths.
        """
        metrics = {
            "use_after_free_count": 0,
            "double_free_count": 0,
            "path_leak_probability": 0.0,
            "pointer_state_transitions": 0,
            "invalid_free_count": 0,
            "leak_count": 0
        }
        
        if not self.nodes or self.entry_id is None:
            return {"warnings": [], "metrics": metrics}

        # DFS stack: (current_node_id, current_pointer_states, visit_counts)
        stack = [(self.entry_id, {}, {n: 0 for n in self.nodes})]
        
        total_paths = 0
        leaking_paths = 0
        
        while stack:
            curr_id, state_in, visits = stack.pop()
            
            if visits[curr_id] >= self.max_visits_per_node:
                continue
                
            visits[curr_id] += 1
            node = self.nodes[curr_id]
            node_line = node.get("line", -1)
            state_out = deepcopy(state_in)
            
            allocs = node.get("allocs", [])
            frees = node.get("frees", [])
            uses = node.get("uses", [])
            returns = node.get("returns", [])
            
            for var in allocs:
                state_out[var] = STATE_ALLOC
                metrics["pointer_state_transitions"] += 1
                
            for var in frees:
                current_st = state_out.get(var, STATE_UNINIT)
                if current_st == STATE_FREED:
                    metrics["double_free_count"] += 1
                    self.warnings.append(PathWarning("DOUBLE_FREE", "CRITICAL", f"Double Free: Pointer '{var}' is freed twice on this execution path.", curr_id, node_line))
                elif current_st != STATE_ALLOC:
                    metrics["invalid_free_count"] += 1
                    self.warnings.append(PathWarning("INVALID_FREE", "CRITICAL", f"Invalid Free: Pointer '{var}' is freed before being safely allocated.", curr_id, node_line))
                
                state_out[var] = STATE_FREED
                metrics["pointer_state_transitions"] += 1
                
            for var in uses:
                current_st = state_out.get(var, None)
                if current_st == STATE_FREED:
                    metrics["use_after_free_count"] += 1
                    self.warnings.append(PathWarning("USE_AFTER_FREE", "CRITICAL", f"Use-After-Free: Pointer '{var}' is dereferenced after being freed.", curr_id, node_line))
                
            out_edges = node.get("out_edges", [])
            if node.get("is_return") or not out_edges:
                total_paths += 1
                path_leaks = []
                for var, st in state_out.items():
                    if st == STATE_ALLOC and var not in returns:
                        path_leaks.append(var)
                        
                if path_leaks:
                    leaking_paths += 1
                    for var in path_leaks:
                        if var not in self.leaked_on_exits:
                            self.leaked_on_exits[var] = set()
                        self.leaked_on_exits[var].add(curr_id)
            else:
                for target_id, _ in out_edges:
                    stack.append((target_id, deepcopy(state_out), deepcopy(visits)))
                    
        if total_paths > 0:
            metrics["path_leak_probability"] = round(leaking_paths / total_paths, 2)
            
        # Update totals
        if len(self.leaked_on_exits) > 0:
            metrics["leak_count"] += len(self.leaked_on_exits)
            
        for var, exit_nodes in self.leaked_on_exits.items():
            exit_node_id = list(exit_nodes)[0]
            exit_line = self.nodes[exit_node_id].get("line", -1)
            self.warnings.append(PathWarning(
                "MEMORY_LEAK", "HIGH", 
                f"Memory Leak: Pointer '{var}' is allocated but not freed before execution exits.",
                exit_node_id, exit_line
            ))

        return {
            "warnings": [w.to_dict() for w in self.warnings],
            "metrics": metrics
        }


def analyze_pointers(cfg_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs data flow over all CFGs in the file."""
    all_warnings = []
    total_metrics = {
        "use_after_free_count": 0,
        "double_free_count": 0,
        "path_leak_probability": 0.0,
        "pointer_state_transitions": 0,
        "invalid_free_count": 0,
        "leak_count": 0
    }
    
    if not cfg_list:
        return {"warnings": [], "metrics": total_metrics}
        
    probs = []
    for cfg in cfg_list:
        tracker = PointerStateTracker(cfg)
        res = tracker.analyze()
        all_warnings.extend(res["warnings"])
        
        m = res["metrics"]
        total_metrics["use_after_free_count"] += m.get("use_after_free_count", 0)
        total_metrics["double_free_count"] += m.get("double_free_count", 0)
        total_metrics["invalid_free_count"] += m.get("invalid_free_count", 0)
        total_metrics["pointer_state_transitions"] += m.get("pointer_state_transitions", 0)
        total_metrics["leak_count"] += m.get("leak_count", 0)
        if "path_leak_probability" in m:
            probs.append(m["path_leak_probability"])
        
    if probs:
        total_metrics["path_leak_probability"] = round(sum(probs) / len(probs), 2)
        
    return {
        "warnings": all_warnings,
        "metrics": total_metrics
    }
