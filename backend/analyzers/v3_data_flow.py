"""
IntelliReview V3 — Data Flow Analysis (DFA)
===========================================
Definite-assignment analysis over the CFG:

  * a forward "must" analysis — the set of variables that are initialised on *every* path reaching a node
    (intersection at joins, iterated to a fix-point, so loops are handled correctly);
  * only plain scalar / pointer locals that were declared WITHOUT an initialiser are tracked
    (parameters, arrays, structs, statics and anything whose address is taken are excluded, because those
    cannot be judged this way and would only create noise);
  * a read of a tracked variable that is not definitely assigned is reported once per (variable, line).

Also flags loops whose condition is a non-zero constant and whose body has no way out
(no `break`, `return`, `goto` or `exit`) as potential infinite loops.
"""
from typing import Dict, Any, List, Optional, Set, FrozenSet


class DataFlowWarning:
    def __init__(self, issue_type: str, severity: str, message: str, node_id: int, line_no: int = -1,
                 suggestion: str = ""):
        self.type = issue_type
        self.severity = severity
        self.message = message
        self.node_id = node_id
        self.line = line_no
        self.suggestion = suggestion

    def to_dict(self):
        d = {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "line": self.line,
            "node_id": self.node_id,
        }
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


def _transfer(state: Set[str], node: Dict[str, Any], tracked: Set[str], report: Optional[list]):
    """Apply the node's ordered ops to `state` (mutated). When `report` is a list, uninitialised reads are appended."""
    for op, var, line in node.get("ordered_ops", []):
        if var not in tracked:
            continue
        if op == "ASSIGN":
            state.add(var)
        elif op == "KILL":
            state.discard(var)
        elif op == "USE" and var not in state:
            if report is not None:
                report.append((var, line, node["id"]))
            state.add(var)                 # do not cascade the same variable down the path


def _analyze_uninitialised(cfg: Dict[str, Any]):
    nodes = cfg.get("nodes", {})
    entry = cfg.get("entry_id")
    tracked = set(cfg.get("tracked_scalars", []))
    if not nodes or entry is None or not tracked:
        return []

    in_state: Dict[Any, Optional[FrozenSet[str]]] = {nid: None for nid in nodes}
    in_state[entry] = frozenset()
    work = [entry]
    steps = 0
    while work and steps < 200000:
        steps += 1
        nid = work.pop()
        state = set(in_state[nid] or ())
        _transfer(state, nodes[nid], tracked, None)
        out = frozenset(state)
        for target, _ in nodes[nid].get("out_edges", []):
            cur = in_state[target]
            new = out if cur is None else cur & out
            if new != cur:
                in_state[target] = new
                work.append(target)

    reports: list = []
    for nid, node in nodes.items():
        if in_state[nid] is None:
            continue                       # unreachable
        _transfer(set(in_state[nid]), node, tracked, reports)
    return reports


def _loop_can_exit(cfg: Dict[str, Any], cond_id) -> bool:
    """Is there any way out of the loop whose condition node is `cond_id` other than its False edge?"""
    nodes = cfg["nodes"]
    body_targets = [t for t, c in nodes[cond_id].get("out_edges", []) if c in ("True", "loop_back")]
    seen = set()
    stack = list(body_targets)
    while stack:
        nid = stack.pop()
        if nid in seen or nid == cond_id:
            continue
        seen.add(nid)
        node = nodes[nid]
        if node.get("is_return") or node.get("noreturn"):
            return True
        for target, cond in node.get("out_edges", []):
            if cond in ("break", "goto"):
                return True
            stack.append(target)
    return False


def analyze_data_flow(source: str, cfg_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    warnings: List[DataFlowWarning] = []
    metrics = {"infinite_loop_risks": 0, "uninitialized_vars_used": 0}
    seen_keys = set()

    for cfg in cfg_list or []:
        for var, line, nid in _analyze_uninitialised(cfg):
            key = ("U", var, line)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            metrics["uninitialized_vars_used"] += 1
            warnings.append(DataFlowWarning(
                "UNINITIALIZED_VARIABLE", "MEDIUM",
                f"Uninitialized Variable: '{var}' may be read before it is assigned a value.",
                nid, line, f"Initialise '{var}' where it is declared."))

        for nid, node in cfg.get("nodes", {}).items():
            if node.get("name") in ("While Cond", "DoWhile Cond") and node.get("cond_is_const_true"):
                if not _loop_can_exit(cfg, nid):
                    key = ("L", cfg.get("function"), nid)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    metrics["infinite_loop_risks"] += 1
                    warnings.append(DataFlowWarning(
                        "INFINITE_LOOP", "MEDIUM",
                        "Infinite Loop Risk: the loop condition is always true and the body contains no "
                        "break, return or exit.", nid, node.get("line", -1),
                        "Add a terminating condition or a break/return inside the loop."))

    return {"warnings": [w.to_dict() for w in warnings], "metrics": metrics}
