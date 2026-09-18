"""
IntelliReview V3 — Path-Sensitive Pointer / Resource State Tracker
===================================================================
Forward data-flow over the CFG. The abstract state of every tracked variable is (kind, line) with kind one of

    A   heap block allocated, result NOT yet NULL-checked        AC  allocated and NULL-checked
    AP  allocated, handed to a function that may own it now      O   allocated pointer moved by arithmetic (p++, p+=n)
    F   freed                                                    N   NULL
    S   points to stack/array memory (must never be free()d)
    R   OS resource open (FILE*, fd, DIR*, socket)               RP  resource handed to another function

A variable that is not in the map is *unknown* (parameter, global, escaped...). States are kept per CFG node as a
bounded *set of distinct abstract states*, so the analysis is path-sensitive without enumerating paths — it terminates
on loops and does not explode on branchy code.

`if (!p) return;`, `if (p == NULL)`, `if (fd < 0)`, `if (fd == -1)` are honoured: on the failure edge nothing is owned, so
the classic "allocation failed, return early" path is not reported as a leak.

Detects: use-after-free, double free, free of non-heap / moved pointers, NULL dereference, unchecked allocation,
memory leak (definite / possible) and resource leak.
"""
from typing import Dict, List, Any, Tuple

MAX_STATES_PER_NODE = 24
MAX_STEPS = 60000

_SEV = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
HEAP = ("A", "AC", "AP", "O")


class PathWarning:
    def __init__(self, issue_type: str, severity: str, message: str, node_id: int, line_no: int = -1,
                 suggestion: str = ""):
        self.type = issue_type
        self.severity = severity
        self.message = message
        self.node_id = node_id
        self.line = line_no
        self.suggestion = suggestion

    def to_dict(self):
        d = {"type": self.type, "severity": self.severity, "message": self.message, "line": self.line,
             "node_id": self.node_id}
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


class PointerStateTracker:
    def __init__(self, cfg: Dict[str, Any], nonowning=frozenset(), defined=frozenset()):
        self.defined = defined            # names of the functions defined in this file
        self.nonowning = nonowning        # {(callee, arg index)} of same-file functions that never take ownership
        self.cfg = cfg
        self.nodes = cfg.get("nodes", {})
        self.entry_id = cfg.get("entry_id")
        self.locals = set(cfg.get("locals", []))
        self.ptr_vars = set(cfg.get("ptr_vars", []))
        self.warnings: Dict[Tuple, PathWarning] = {}
        self.null_hits: Dict[Tuple, int] = {}     # (node, var, line) -> #states that saw NULL there
        self.processed: Dict[int, int] = {}       # node -> #distinct states processed
        self.truncated = False
        self.metrics = {
            "use_after_free_count": 0, "double_free_count": 0, "null_deref_count": 0, "path_leak_probability": 0.0,
            "pointer_state_transitions": 0, "invalid_free_count": 0, "leak_count": 0,
        }

    # ---- reporting -----------------------------------------------------
    def _warn(self, kind, sev, msg, node_id, line, suggestion="", key_extra=None):
        # one unchecked allocation is one finding, reported at its first use, however many paths/lines use it afterwards
        key = (kind, None, key_extra) if kind == "UNCHECKED_ALLOC" else (kind, line, key_extra)
        existing = self.warnings.get(key)
        if existing is None or _SEV[sev] > _SEV[existing.severity] or                 (kind == "UNCHECKED_ALLOC" and line < existing.line):
            self.warnings[key] = PathWarning(kind, sev, msg, node_id, line, suggestion)

    # ---- transfer function ---------------------------------------------
    def _apply(self, state: Dict[str, Tuple[str, int]], node: Dict[str, Any]):
        nid = node["id"]
        for op, var, line in node.get("ptr_ops", []):
            st = state.get(var)
            kind = st[0] if st else None
            if op.startswith("PASS@"):
                _, callee, idx = op.split("@")
                if (callee, int(idx)) in self.nonowning:
                    op = "USEARG"
                elif callee in self.defined:
                    op = "ESCAPE"         # a function we can see frees/stores it: ownership provably moves there
                else:
                    op = "PASS"

            if op == "ALLOC":
                if var not in self.locals:
                    state.pop(var, None)                         # global / static: not ours to track
                    continue
                if kind in ("A", "AC"):
                    self._warn("MEMORY_LEAK", "MEDIUM",
                               f"Memory Leak: '{var}' (allocated at line {st[1]}) is overwritten by a new allocation "
                               f"at line {line} without being freed.", nid, st[1],
                               "Call free() on the old pointer before reassigning it.", var)
                state[var] = ("A", line)
                self.metrics["pointer_state_transitions"] += 1

            elif op == "REALLOC":
                state[var] = ("A", st[1] if kind in HEAP else line)

            elif op == "RES_ALLOC":
                if var not in self.locals:
                    state.pop(var, None)
                    continue
                if kind == "R":
                    self._warn("RESOURCE_LEAK", "MEDIUM",
                               f"Resource Leak: '{var}' (opened at line {st[1]}) is overwritten by a new handle at line "
                               f"{line} without being closed.", nid, st[1],
                               "Close the old file/descriptor before reusing the variable.", var)
                state[var] = ("R", line)
                self.metrics["pointer_state_transitions"] += 1

            elif op == "RES_FREE":
                state.pop(var, None)                             # closed

            elif op == "NULL":
                if var not in self.ptr_vars:
                    continue
                if kind in ("A", "AC"):
                    self._warn("MEMORY_LEAK", "MEDIUM",
                               f"Memory Leak: '{var}' (allocated at line {st[1]}) is set to NULL at line {line} "
                               f"without being freed.", nid, st[1], "Call free() before overwriting the pointer.", var)
                state[var] = ("N", line)

            elif op == "STACKPTR":
                if var in self.locals or var in self.ptr_vars:
                    state[var] = ("S", line)

            elif op == "OFFSET":
                if kind in ("A", "AC"):
                    state[var] = ("O", st[1])

            elif op == "BADFREE":
                what = ("free() is called on the address of a variable — that is not heap memory (CWE-590)."
                        if var == "stack" else
                        "free() is called on a pointer expression (p + n) instead of the pointer returned by the "
                        "allocator (CWE-761).")
                self._warn("INVALID_FREE", "HIGH", f"Invalid Free: {what}", nid, line,
                           "Only pass the exact pointer returned by malloc/calloc/realloc to free().", var)

            elif op == "FREE":
                if kind == "F":
                    self._warn("DOUBLE_FREE", "CRITICAL",
                               f"Double Free: Pointer '{var}' is freed twice on this execution path "
                               f"(first free at line {st[1]}).", nid, line,
                               "Set the pointer to NULL right after free() and free it only once.", var)
                elif kind == "S":
                    self._warn("INVALID_FREE", "HIGH",
                               f"Invalid Free: '{var}' points to stack/array memory and must not be passed to free() "
                               f"(CWE-590).", nid, line, "Only free() memory obtained from malloc/calloc/realloc.", var)
                    state[var] = ("F", line)
                elif kind == "O":
                    self._warn("INVALID_FREE", "HIGH",
                               f"Invalid Free: '{var}' was moved by pointer arithmetic and no longer points to the "
                               f"start of the allocated block (CWE-761).", nid, line,
                               "Free the original pointer, or keep a separate copy of it for free().", var)
                    state[var] = ("F", line)
                elif kind != "N":
                    state[var] = ("F", line)
                    self.metrics["pointer_state_transitions"] += 1

            elif op in ("DEREF", "USEARG"):
                if kind == "F":
                    sev = "CRITICAL" if op == "DEREF" else "HIGH"
                    self._warn("USE_AFTER_FREE", sev,
                               f"Use-After-Free: Pointer '{var}' is {'dereferenced' if op == 'DEREF' else 'passed to a function'} "
                               f"after being freed (freed at line {st[1]}).", nid, line,
                               "Do not access memory after free(); set the pointer to NULL after freeing.", var)
                elif kind == "N" and op == "DEREF":
                    self.null_hits[(nid, var, line)] = self.null_hits.get((nid, var, line), 0) + 1
                    self._warn("NULL_DEREF", "HIGH",
                               f"Null Pointer Dereference: '{var}' is NULL on this path when it is dereferenced.",
                               nid, line, "Check the pointer against NULL before using it.", var)
                elif kind == "A":
                    self._warn("UNCHECKED_ALLOC", "MEDIUM" if op == "DEREF" else "LOW",
                               f"Unchecked Allocation: the result of the allocation stored in '{var}' (line {st[1]}) "
                               f"is used without checking for NULL (CWE-690).", nid, line,
                               "Check `if (ptr == NULL)` right after malloc/calloc/realloc.", var)
                    state[var] = ("AC", st[1])                    # report once per allocation

            elif op == "PASS":
                if kind == "F":
                    self._warn("USE_AFTER_FREE", "HIGH",
                               f"Use-After-Free: Pointer '{var}' is passed to a function after being freed "
                               f"(freed at line {st[1]}).", nid, line, "Do not use a pointer after free().", var)
                elif kind in ("A", "AC"):
                    state[var] = ("AP", st[1])
                elif kind == "R":
                    state[var] = ("RP", st[1])

            elif op.startswith("ALIAS@"):
                src = op.split("@", 1)[1]
                sst = state.get(src)
                if sst and sst[0] in ("N", "F", "S"):
                    state[var] = sst                 # both names now refer to NULL / freed / stack memory
                else:
                    state.pop(src, None)             # ownership may have moved to `var`: stop tracking both
                    state.pop(var, None)

            elif op in ("ESCAPE", "ASSIGN_OTHER"):
                state.pop(var, None)

    def _refine(self, state, node, edge_cond):
        """Narrow the state on the True/False edge of a NULL / failure check."""
        chk = node.get("null_check")
        if not chk or edge_cond not in ("True", "False"):
            return state
        var = chk["var"]
        st = state.get(var)
        if var not in self.ptr_vars and st is None:
            return state                      # `if (x == 0)` on a plain integer says nothing about NULL
        kind = st[0] if st else None
        is_fail_edge = (edge_cond == "True") == chk["true_is_null"]
        if (not is_fail_edge and kind == "N") or (is_fail_edge and kind == "S"):
            return None                       # e.g. `if (p != NULL)` while p is known NULL: this path cannot happen
        new = dict(state)
        if is_fail_edge:
            if kind in ("R", "RP"):
                new.pop(var, None)            # open failed: there is nothing to close
            elif kind is None or kind in ("A", "AC", "AP", "N", "O"):
                new[var] = ("N", node.get("line", -1))   # an allocation that came back NULL owns nothing
            else:
                return state
            return new
        if kind == "A":
            new[var] = ("AC", st[1])          # NULL-checked
            return new
        return state

    # ---- driver ---------------------------------------------------------
    def analyze(self) -> Dict[str, Any]:
        if not self.nodes or self.entry_id is None:
            return {"warnings": [], "metrics": self.metrics}

        seen: Dict[int, set] = {self.entry_id: {frozenset()}}
        work: List[Tuple[int, frozenset]] = [(self.entry_id, frozenset())]
        exit_states = 0
        leaking_exit_states = 0
        steps = 0

        while work:
            steps += 1
            if steps > MAX_STEPS:
                self.truncated = True
                break
            nid, fstate = work.pop()
            node = self.nodes[nid]
            self.processed[nid] = self.processed.get(nid, 0) + 1
            state = dict(fstate)
            self._apply(state, node)

            out_edges = node.get("out_edges", [])
            terminal = node.get("is_return") or not out_edges
            if terminal:
                if not node.get("noreturn"):
                    exit_states += 1
                    leaked = False
                    for var, (kind, aline) in state.items():
                        if var not in self.locals:
                            continue
                        if kind in ("A", "AC"):
                            leaked = True
                            self._warn("MEMORY_LEAK", "MEDIUM",
                                       f"Memory Leak: '{var}' allocated at line {aline} is not freed on every path "
                                       f"before the function returns.", nid, aline,
                                       "Ensure every allocation is freed (or ownership is returned) on all paths, "
                                       "including early returns.", var)
                        elif kind == "AP":
                            self._warn("MEMORY_LEAK", "LOW",
                                       f"Possible Memory Leak: '{var}' allocated at line {aline} was handed to another "
                                       f"function and is not freed here; verify that the callee takes ownership.",
                                       nid, aline, "Document ownership transfer or free the pointer here.", var)
                        elif kind == "R":
                            leaked = True
                            self._warn("RESOURCE_LEAK", "MEDIUM",
                                       f"Resource Leak: '{var}' opened at line {aline} is not closed on every path "
                                       f"before the function returns.", nid, aline,
                                       "Close files/sockets/descriptors on every path (fclose/close), including errors.",
                                       var)
                        elif kind == "RP":
                            self._warn("RESOURCE_LEAK", "LOW",
                                       f"Possible Resource Leak: '{var}' opened at line {aline} was handed to another "
                                       f"function and is not closed here.", nid, aline,
                                       "Verify that the callee closes it.", var)
                    if leaked:
                        leaking_exit_states += 1
                continue

            for target_id, cond in out_edges:
                nxt = self._refine(state, node, cond)
                if nxt is None:
                    continue
                fs = frozenset(nxt.items())
                bucket = seen.setdefault(target_id, set())
                if fs in bucket:
                    continue
                if len(bucket) >= MAX_STATES_PER_NODE:
                    self.truncated = True
                    continue
                bucket.add(fs)
                work.append((target_id, fs))

        # A NULL dereference on only *some* of the states reaching a node is often an infeasible path (correlated
        # conditions), so report it as MEDIUM; on all of them it is a definite bug.
        for (nid, var, line), hits in self.null_hits.items():
            if hits < self.processed.get(nid, 0):
                w = self.warnings.get(("NULL_DEREF", line, var))
                if w:
                    w.severity = "MEDIUM"
                    w.message = (f"Possible Null Pointer Dereference: '{var}' may be NULL on some paths "
                                 f"when it is dereferenced.")

        if exit_states:
            self.metrics["path_leak_probability"] = round(leaking_exit_states / exit_states, 2)
        self.metrics["leak_count"] = len({(w.type, w.message.split("'")[1]) for w in self.warnings.values()
                                          if w.type in ("MEMORY_LEAK", "RESOURCE_LEAK") and w.severity != "LOW"})
        for k, t in (("use_after_free_count", "USE_AFTER_FREE"), ("double_free_count", "DOUBLE_FREE"),
                     ("null_deref_count", "NULL_DEREF"), ("invalid_free_count", "INVALID_FREE")):
            self.metrics[k] = sum(1 for w in self.warnings.values() if w.type == t)

        fn = self.cfg.get("function")
        return {"warnings": [dict(w.to_dict(), function=fn) for w in self.warnings.values()], "metrics": self.metrics,
                "truncated": self.truncated}


def analyze_pointers(cfg_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs the pointer analysis over all CFGs in the file."""
    all_warnings: List[Dict[str, Any]] = []
    total = {
        "use_after_free_count": 0, "double_free_count": 0, "null_deref_count": 0,
        "path_leak_probability": 0.0, "pointer_state_transitions": 0, "invalid_free_count": 0, "leak_count": 0,
    }
    probs: List[float] = []
    truncated = False

    # Which parameters of functions defined in THIS file may take ownership of a pointer? A parameter that is
    # freed, stored, returned or passed on might; one that is only read or written through does not.
    owning = set()
    for cfg in cfg_list or []:
        order = cfg.get("param_order", [])
        for node in cfg.get("nodes", {}).values():
            for op, var, _line in node.get("ptr_ops", []):
                if op.split("@")[0] in ("FREE", "ESCAPE", "PASS", "BADFREE") and var in order:
                    owning.add((cfg.get("function"), order.index(var)))
                elif op.startswith("ALIAS@") and op.split("@", 1)[1] in order:
                    owning.add((cfg.get("function"), order.index(op.split("@", 1)[1])))
    nonowning = frozenset((cfg.get("function"), i) for cfg in (cfg_list or [])
                          for i in range(len(cfg.get("param_order", []))) if (cfg.get("function"), i) not in owning)

    defined = frozenset(c.get("function") for c in (cfg_list or []))
    for cfg in cfg_list or []:
        res = PointerStateTracker(cfg, nonowning, defined).analyze()
        all_warnings.extend(res["warnings"])
        truncated = truncated or res.get("truncated", False)
        m = res["metrics"]
        for k in ("use_after_free_count", "double_free_count", "null_deref_count", "invalid_free_count",
                  "pointer_state_transitions", "leak_count"):
            total[k] += m.get(k, 0)
        if m.get("path_leak_probability") is not None:
            probs.append(m["path_leak_probability"])
    if probs:
        total["path_leak_probability"] = round(sum(probs) / len(probs), 2)
    return {"warnings": all_warnings, "metrics": total, "truncated": truncated}
