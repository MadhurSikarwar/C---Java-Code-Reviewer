"""
IntelliReview — /api/analyze route
Accepts code + language, runs the full analysis pipeline, returns a JSON report.

Hardening (the previous version returned "Clean" for empty input, prose and syntax errors, froze the
whole server on a 2MB single-line input, and returned 500 for bad payloads):
  * strict request validation -> 4xx with a helpful message, never 500
  * size limits (characters, lines, line length)
  * the CPU-bound analysis runs in a worker thread (the event loop keeps answering /health)
    with a timeout
"""
import asyncio
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.text_utils import sanitize, braces_balanced, looks_like_code
from pipeline import static_analysis
from suggestions.generator import generate_suggestions
from analyzers.v4_issue_aggregator import aggregate_issues
from ml.predict import predict_risk

router = APIRouter()

MAX_CODE_CHARS = 200_000
MAX_LINES = 6_000
MAX_LINE_LENGTH = 4_000
ANALYSIS_TIMEOUT_S = 60

VALID_MODELS = ("dl", "v3", "ensemble", "v4_dl", "all")
MODEL_ALIASES = {"v3_path_sensitive": "v3"}
LANGUAGE_ALIASES = {"C": "C", "C++": "C", "CPP": "C", "CXX": "C", "JAVA": "JAVA"}


class CodeRequest(BaseModel):
    code: str
    language: str  # "C", "C++" or "Java"
    model_type: str = "dl"  # "dl", "v3", "ensemble", "v4_dl" or "all"

    @field_validator("language")
    @classmethod
    def _language(cls, v: str) -> str:
        key = v.upper().strip()
        if key not in LANGUAGE_ALIASES:
            raise ValueError(f"Unsupported language '{v}'. Use 'C', 'C++' or 'Java'.")
        return key

    @field_validator("model_type")
    @classmethod
    def _model(cls, v: str) -> str:
        key = MODEL_ALIASES.get(v.lower().strip(), v.lower().strip())
        if key not in VALID_MODELS:
            raise ValueError(f"Unknown model_type '{v}'. Use one of {', '.join(VALID_MODELS)}.")
        return key


def validate_source(code: str, language: str) -> None:
    """Raise HTTPException(4xx) if `code` cannot meaningfully be analysed."""
    if "\x00" in code:
        raise HTTPException(422, "The input contains NUL bytes — it looks like a binary file, not source code.")
    if not code.strip():
        raise HTTPException(422, "No code to analyze: the input is empty.")
    if len(code) > MAX_CODE_CHARS:
        raise HTTPException(413, f"Input too large ({len(code):,} characters). Limit is {MAX_CODE_CHARS:,}.")
    lines = code.splitlines()
    if len(lines) > MAX_LINES:
        raise HTTPException(413, f"Input has {len(lines):,} lines. Limit is {MAX_LINES:,}.")
    longest = max((len(l) for l in lines), default=0)
    if longest > MAX_LINE_LENGTH:
        raise HTTPException(413, f"A line is {longest:,} characters long (limit {MAX_LINE_LENGTH:,}) — "
                                 "is this minified or generated code?")

    clean = sanitize(code, java_text_blocks=(language == "JAVA"))
    if not clean.strip():
        return  # only comments: valid, analysed as an empty program
    if not looks_like_code(clean):
        raise HTTPException(422, f"This does not look like {'Java' if language == 'JAVA' else 'C/C++'} source code.")
    ok, reason = braces_balanced(clean)
    if not ok:
        raise HTTPException(422, f"Syntax error: {reason}. Fix the brackets and analyze again.")


@router.post("/analyze")
async def analyze_code(request: CodeRequest):
    language = LANGUAGE_ALIASES[request.language]
    validate_source(request.code, language)
    try:
        return await asyncio.wait_for(
            run_in_threadpool(_run_analysis, request.code, language, request.model_type),
            timeout=ANALYSIS_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, f"Analysis took longer than {ANALYSIS_TIMEOUT_S}s and was abandoned. "
                                 "Try a smaller file.")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(500, f"Internal analysis error ({type(e).__name__}). This is a bug in IntelliReview, "
                                 "not in your code.")


@router.post("/analyze/stream")
async def analyze_stream(request: CodeRequest):
    """Same analysis as /analyze, but as newline-delimited JSON: one `{"type":"trace",...}` line per stage as it finishes,
    then a final `{"type":"result","data":{...}}` (or `{"type":"error","detail":...}`)."""
    language = LANGUAGE_ALIASES[request.language]
    validate_source(request.code, language)          # bad input is still a proper 4xx, before any streaming starts
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def put(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    def worker():
        try:
            put({"type": "result", "data": _run_analysis(request.code, language, request.model_type, trace=put)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            put({"type": "error", "detail": f"Internal analysis error ({type(e).__name__}). This is a bug in IntelliReview, "
                                            "not in your code."})
        finally:
            put(None)

    loop.run_in_executor(None, worker)

    async def lines():
        deadline = time.monotonic() + ANALYSIS_TIMEOUT_S
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=max(0.1, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                yield json.dumps({"type": "error", "detail": f"Analysis took longer than {ANALYSIS_TIMEOUT_S}s and was abandoned."}) + "\n"
                return
            if ev is None:
                return
            yield json.dumps(ev) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


def _call_graph(cfgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact per-function call graph for the UI (the raw CFGs are far too large to ship)."""
    defined = {c.get("function") for c in cfgs}
    out = []
    for c in cfgs:
        calls = sorted({x for x in c.get("function_calls", []) if x in defined})
        out.append({"func_name": c.get("function"), "function": c.get("function"), "calls": calls,
                    "num_blocks": len(c.get("nodes", {}))})
    return out


def _run_analysis(source: str, language: str, model_type: str = "dl", trace=None) -> dict:
    """Core analysis pipeline. `language` is 'C' or 'JAVA'.

    Every stage reports what it is doing through `tr`; the events are kept in the result (`trace`) and, when a `trace`
    callback is given (the streaming endpoint), forwarded live so the UI can show the engine thinking."""
    t0 = time.perf_counter()
    events: list = []

    def tr(e: dict) -> None:
        e = dict(e, t=int((time.perf_counter() - t0) * 1000))
        events.append(e)
        if trace:
            trace(dict(e, type="trace"))

    sa = static_analysis(source, language, trace=tr)
    parse_result = sa["parse_result"]
    memory_result = sa["memory_result"]
    unsafe_result = sa["unsafe_result"]
    complexity_result = sa["complexity_result"]
    recursion_result = sa["recursion_result"]
    feature_data = sa["feature_data"]
    v3_issues = sa["v3_issues"]
    all_issues = sa["all_issues"]

    ml_result = predict_risk(sa["feature_vector"], model_type=model_type, issues=all_issues, trace=tr)
    tr({"stage": "verdict", "kind": "verdict",
        "text": f"Verdict: {ml_result.get('risk_label', '?')}, {int(ml_result.get('risk_score', 0))} out of 100."})

    suggestion_result = generate_suggestions(
        features=feature_data["features"],
        memory_issues=aggregate_issues(memory_result.get("issues", []) + v3_issues),
        unsafe_issues=aggregate_issues(unsafe_result.get("issues", [])),
        complexity_issues=aggregate_issues(complexity_result.get("issues", [])),
        recursion_issues=aggregate_issues(recursion_result.get("issues", [])),
    )

    weight = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    highlighted_lines: Dict[int, str] = {}
    for issue in all_issues:
        line = int(issue.get("line", 0) or 0)
        severity = str(issue.get("severity", "LOW"))
        if line > 0:
            prev = highlighted_lines.get(line, "LOW")
            highlighted_lines[line] = max(prev, severity, key=lambda s: weight.get(s, 1))

    notes = list(sa["notes"])
    if parse_result.get("analysis_mode") == "heuristic":
        notes.append(
            "This code could not be fully parsed"
            + (" (looks like C++)" if parse_result.get("cpp_like") else "")
            + "; a heuristic analysis was used, so findings are less precise and path-sensitive checks "
              "(use-after-free, double free, uninitialized variables) were skipped."
        )

    mem_types = ("MEMORY_LEAK", "USE_AFTER_FREE", "DOUBLE_FREE", "INVALID_FREE", "NULL_DEREF")
    return {
        "success": True,
        "language": parse_result.get("language", "C"),
        "analysis_mode": parse_result.get("analysis_mode", "ast"),
        "notes": notes,
        "metrics": {
            "lines_of_code": int(parse_result.get("lines_of_code", 0)),
            "num_functions": int(parse_result.get("num_functions", 0)),
            "num_loops": int(parse_result.get("num_loops", 0)),
            "num_conditionals": int(parse_result.get("num_conditionals", 0)),
            "max_nesting_depth": int(parse_result.get("max_nesting_depth", 0)),
            "cyclomatic_complexity": int(complexity_result.get("cyclomatic_complexity", 1)),
            "time_complexity": str(complexity_result.get("time_complexity", "O(n)")),
            "memory_leak_count": int(memory_result.get("memory_leak_count", 0)),
            "unsafe_function_count": int(unsafe_result.get("unsafe_function_count", 0)),
            "recursion_count": int(recursion_result.get("recursion_count", 0)),
            "v3_active": bool(sa["cfgs"]),
        },
        "risk": {
            "score": int(ml_result.get("risk_score", 5)),
            "label": str(ml_result.get("risk_label", "Unknown")),
            "confidence": int(ml_result.get("confidence", 0)),
            "explanations": list(ml_result.get("explanations", [])) + notes,
            "probabilities": ml_result.get("probabilities", {"Clean": 33, "Moderate Risk": 34, "High Risk": 33}),
            "comparisons": ml_result.get("comparisons", {}),
            "categories": {
                "memory": sum(1 for i in all_issues if i.get("type") in mem_types),
                "data_flow": sum(1 for i in all_issues if i.get("type") in ("UNINITIALIZED_VARIABLE", "INFINITE_LOOP")),
                "architecture": sum(1 for i in all_issues if i.get("type") in ("HIGH_COMPLEXITY", "DEEP_NESTING", "DEAD_CODE")),
                "recursion": int(recursion_result.get("recursion_count", 0)),
                "unsafe_headers": int(unsafe_result.get("unsafe_function_count", 0)),
            },
        },
        "issues": [
            {
                "severity": str(issue.get("severity", "LOW")),
                "message": str(issue.get("message", "")),
                "line": int(issue.get("line", 0)) if issue.get("line") else 0,
                "type": str(issue.get("type", "")),
                "suggestion": str(issue.get("suggestion", "")),
                "fix": issue.get("fix"),            # {title, line, before, after, note} or null
            }
            for issue in all_issues
        ],
        "suppressed": [{"line": int(i.get("line") or 0), "type": str(i.get("type", "")), "message": str(i.get("message", ""))}
                       for i in sa["suppressed"]],
        "trace": events,
        "highlighted_lines": highlighted_lines,
        "function_complexity": [
            {
                "function": str(fc.get("function", fc.get("name", "__main__"))),
                "cyclomatic_complexity": int(fc.get("cyclomatic_complexity", 1)),
            }
            for fc in (complexity_result.get("function_complexity", []) or [])
        ],
        "functions": [
            {"name": str(fn.get("name", "")), "line": int(fn.get("line", 0)) if fn.get("line") else 0}
            for fn in (parse_result.get("functions", []) or [])
        ],
        "cfgs": _call_graph(sa["cfgs"]),
        "suggestions": suggestion_result,
        "parse_errors": parse_result.get("parse_errors", []),
        "features": feature_data.get("features", []),
    }
