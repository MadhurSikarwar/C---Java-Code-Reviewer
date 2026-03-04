"""
IntelliReview — /api/analyze route
Accepts code + language, runs full analysis pipeline, returns JSON report.
"""
import traceback
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.c_parser import parse_c_code
from parsers.java_parser import parse_java_code
from analyzers.memory_leak import detect_memory_leaks
from analyzers.unsafe_functions import detect_unsafe_functions
from analyzers.complexity import analyze_complexity
from analyzers.recursion import detect_recursion
from analyzers.feature_extractor import extract_features
from analyzers.v3_cfg_builder import build_cfg_for_file
from analyzers.v3_pointer_state import analyze_pointers
from analyzers.v3_data_flow import analyze_data_flow
from analyzers.v3_code_smells import analyze_smells
from analyzers.v3_complexity import analyze_time_complexity
from analyzers.v3_feature_extractor import extract_v3_features
from suggestions.generator import generate_suggestions
from ml.predict import predict_risk, load_model

router = APIRouter()


class CodeRequest(BaseModel):
    code: str
    language: str  # "C" or "Java"
    model_type: str = "dl"  # "dl", "v3", or "all"

@router.post("/analyze")
async def analyze_code(request: CodeRequest):
    print(f"📊 Analysis request received:")
    print(f"   Language: {request.language}")
    print(f"   Model Type: {request.model_type}")
    print(f"   Code length: {len(request.code)} characters")
    
    try:
        print("🔄 Starting analysis...")
        result = _run_analysis(request.code, request.language, request.model_type)
        print("✅ Analysis completed successfully")
        return result
    except Exception as e:
        error_msg = str(e)
        if "traceback" in str(type(e)):
            error_msg = f"{e.__class__.__name__}: {error_msg}"
        import traceback as tb
        print(f"❌ Analysis failed: {error_msg}")
        tb.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

def _run_analysis(source: str, language: str, model_type: str = "dl") -> dict:
    """Core analysis pipeline."""
    language = language.upper().strip()
    if language not in ("C", "JAVA"):
        raise ValueError(f"Unsupported language '{language}'. Use 'C' or 'Java'.")

    # Normalize model_type
    model_type = model_type.lower().strip()
    valid_types = ("dl", "v3", "ensemble", "v4_dl", "all")
    if model_type not in valid_types:
        # Accept legacy names and convert
        if model_type == "v3_path_sensitive":
            model_type = "v3"
        else:
            model_type = "v3"  # Default to V3 if unknown

    # 1. Parse
    if language == "C":
        parse_result = parse_c_code(source)
    else:
        parse_result = parse_java_code(source)

    # 2. Static Analysis Fundamentals (Always Run)
    memory_result = detect_memory_leaks(parse_result)
    unsafe_result = detect_unsafe_functions(parse_result)
    complexity_result = analyze_complexity(parse_result, source)
    recursion_result = detect_recursion(parse_result, source)

    # 3. Extract Classic Features (21 features)
    feature_data = extract_features(
        parse_result, memory_result, unsafe_result,
        complexity_result, recursion_result,
        source=source,
    )
    
    # NEW: Run V3 Path-Sensitive Analytics if Requested
    v3_issues = []
    cfgs = []
    v3_leak_count = 0
    
    if model_type in ("v3", "dl", "ensemble", "v4_dl", "all"):
        if language == "C":
            try:
                # Build AST properly handles the dict
                cfgs = build_cfg_for_file(parse_result.get("ast"))
                
                # Re-evaluate recursion with the strict Inter-procedural V3 Call Graph
                recursion_result = detect_recursion(parse_result, source, cfgs=cfgs)
                p_res = analyze_pointers(cfgs)
                v3_leak_count = p_res.get("metrics", {}).get("leak_count", 0)
                
                d_res = analyze_data_flow(source, cfgs)
                s_res = analyze_smells(source, cfgs)
                c_res = analyze_time_complexity(cfgs)
                
                v3_feats = extract_v3_features(p_res, d_res, s_res, c_res)
                
                # Combine 21 + 13 = 34 features
                # Order must match V3_FEATURES list in training exactly
                v3_order = [
                    "use_after_free_count", "double_free_count", "path_leak_probability",
                    "pointer_state_transitions", "infinite_loop_risks", "uninitialized_vars_used",
                    "cfg_node_count", "branch_density", "cyclomatic_complexity", 
                    "global_mutation_count", "loop_count", "max_loop_depth", "recursion_count"
                ]
                for f in v3_order:
                    feature_data["feature_vector"].append(float(v3_feats.get(f, 0.0)))
                    
                v3_issues.extend(p_res.get("warnings", []))
                v3_issues.extend(d_res.get("warnings", []))
                v3_issues.extend(s_res.get("warnings", []))
            except Exception as e:
                # If V3 analysis fails, log but continue with basic analysis
                print(f"⚠️ V3 analysis failed: {str(e)}")
        else:
            # For Java, use basic features but still try to extract from basic analysis
            # Pad feature vector for compatibility with V3 model if needed
            pass

    # If path-sensitive V3 analysis fired and found memory leaks, suppress the naive regex memory leaks to prevent duplicates
    has_v3_memory_issues = any(i.get("type") in ("MEMORY_LEAK", "DOUBLE_FREE") for i in v3_issues)
    basic_memory_issues = memory_result.get("issues", [])
    if has_v3_memory_issues:
        basic_memory_issues = [i for i in basic_memory_issues if i.get("type") not in ("MEMORY_LEAK", "DOUBLE_FREE")]

    # Collect all issues for the dashboard FIRST so ML has context for Hard Overrides!
    raw_issues = (
        basic_memory_issues
        + unsafe_result.get("issues", [])
        + complexity_result.get("issues", [])
        + recursion_result.get("issues", [])
        + v3_issues
    )
    
    # 3.5 Deduplicate Issues (Stop Path Explosion)
    from analyzers.v4_issue_aggregator import aggregate_issues
    all_issues = aggregate_issues(raw_issues)

    # 4. ML Prediction with Hard Override Support
    ml_result = predict_risk(feature_data["feature_vector"], model_type=model_type, issues=all_issues)

    # 5. Suggestions (Using aggregated issues to avoid duplicate noise)
    suggestion_result = generate_suggestions(
        features=feature_data["features"],
        memory_issues=aggregate_issues(memory_result.get("issues", []) + v3_issues),
        unsafe_issues=aggregate_issues(unsafe_result.get("issues", [])),
        complexity_issues=aggregate_issues(complexity_result.get("issues", [])),
        recursion_issues=aggregate_issues(recursion_result.get("issues", [])),
    )

    # Build highlighted lines map
    highlighted_lines = {}
    for issue in all_issues:
        # Some new V3 issues have node_id fallback, we need to map to line roughly
        # For our mock, we just skip line map if missing
        line = issue.get("line", 0)
        if hasattr(issue, 'get'):
            severity = issue.get("severity", "LOW")
        else:
            severity = "LOW"
            
        if line > 0:
            highlighted_lines[line] = max(
                highlighted_lines.get(line, "LOW"),
                severity,
                key=lambda s: {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(s, 1)
            )

    return {
        "success": True,
        "language": parse_result.get("language", "C"),
        "metrics": {
            "lines_of_code": int(parse_result.get("lines_of_code", 0)),
            "num_functions": int(parse_result.get("num_functions", 0)),
            "num_loops": int(parse_result.get("num_loops", 0)),
            "num_conditionals": int(parse_result.get("num_conditionals", 0)),
            "max_nesting_depth": int(parse_result.get("max_nesting_depth", 0)),
            "cyclomatic_complexity": int(complexity_result.get("cyclomatic_complexity", 1)),
            "time_complexity": str(complexity_result.get("time_complexity", "O(n)")),
            "memory_leak_count": int(memory_result.get("memory_leak_count", 0) + v3_leak_count),
            "unsafe_function_count": int(unsafe_result.get("unsafe_function_count", 0)),
            "recursion_count": int(recursion_result.get("recursion_count", 0)),
            "v3_active": model_type in ("v3", "all", "dl")
        },
        "risk": {
            "score": int(ml_result.get("risk_score", 5)),
            "label": str(ml_result.get("risk_label", "Unknown")),
            "confidence": int(ml_result.get("confidence", 0)),
            "explanations": ml_result.get("explanations", []),
            "probabilities": ml_result.get("probabilities", {"Clean": 33, "Moderate Risk": 34, "High Risk": 33}),
            "comparisons": ml_result.get("comparisons", {}),
            "categories": {
                "memory": len(memory_result.get("issues", [])) + len([i for i in v3_issues if i.get("type", "") in ["MEMORY_LEAK", "USE_AFTER_FREE", "DOUBLE_FREE", "INVALID_FREE"]]),
                "data_flow": len([i for i in v3_issues if i.get("type", "") in ["UNINITIALIZED_VARIABLE", "INFINITE_LOOP"]]),
                "architecture": len([i for i in v3_issues if i.get("type", "") in ["HIGH_COMPLEXITY", "DEEP_NESTING", "DEAD_CODE"]]),
                "recursion": int(recursion_result.get("recursion_count", 0)),
                "unsafe_headers": int(unsafe_result.get("unsafe_function_count", 0))
            }
        },
        "issues": [
            {
                "severity": str(issue.get("severity", "LOW")),
                "message": str(issue.get("message", "")),
                "line": int(issue.get("line", 0)) if issue.get("line") else 0,
                "type": str(issue.get("type", "")),
                "suggestion": str(issue.get("suggestion", ""))
            }
            for issue in all_issues
        ],
        "highlighted_lines": highlighted_lines,
        "function_complexity": [
            {
                "function": str(fc.get("function", fc.get("name", "__main__"))),
                "cyclomatic_complexity": int(fc.get("cyclomatic_complexity", 1))
            }
            for fc in (complexity_result.get("function_complexity", []) or [])
        ],
        "functions": [
            {
                "name": str(fn.get("name", "")),
                "line": int(fn.get("line", 0)) if fn.get("line") else 0
            }
            for fn in (parse_result.get("functions", []) or [])
        ],
        "cfgs": cfgs if model_type in ("v3", "dl") else [],
        "suggestions": suggestion_result,
        "parse_errors": parse_result.get("parse_errors", []),
        "features": feature_data.get("features", []),
    }
