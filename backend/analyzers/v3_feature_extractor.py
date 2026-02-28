"""
IntelliReview V3 — Feature Extractor
=====================================
Aggregates all V3 Path-Sensitive metrics into a flat feature dictionary
used for Random Forest validation and DL training.
"""
from typing import Dict, Any

def extract_v3_features(
    pointer_res: Dict[str, Any],
    data_flow_res: Dict[str, Any],
    smells_res: Dict[str, Any],
    complexity_res: Dict[str, Any]
) -> Dict[str, float]:
    
    # Defaults in case an analyzer failed
    p_m = pointer_res.get("metrics", {})
    d_m = data_flow_res.get("metrics", {})
    s_m = smells_res.get("metrics", {})
    c_m = complexity_res.get("metrics", {})

    features = {
        # 1. Pointer & Path Metrics
        "use_after_free_count": float(p_m.get("use_after_free_count", 0)),
        "double_free_count": float(p_m.get("double_free_count", 0)),
        "path_leak_probability": float(p_m.get("path_leak_probability", 0.0)),
        "pointer_state_transitions": float(p_m.get("pointer_state_transitions", 0)),
        
        # 2. Data Flow Metrics
        "infinite_loop_risks": float(d_m.get("infinite_loop_risks", 0)),
        "uninitialized_vars_used": float(d_m.get("uninitialized_vars_used", 0)),
        
        # 3. Structural Smell Metrics
        "cfg_node_count": float(s_m.get("cfg_node_count", 0)),
        "branch_density": float(s_m.get("branch_density", 0.0)),
        "cyclomatic_complexity": float(s_m.get("cyclomatic_complexity", 1)),
        "global_mutation_count": float(s_m.get("global_mutation_count", 0)),
        
        # 4. Complexity Engine Metrics
        "loop_count": float(c_m.get("loop_count", 0)),
        "max_loop_depth": float(c_m.get("max_loop_depth", 0)),
        "recursion_count": float(c_m.get("recursion_count", 0)),
    }
    
    return features
