#!/usr/bin/env python
"""
IntelliReview command line — use it in a pre-commit hook or a CI pipeline.

    python cli.py src/ main.c Account.java                 # human-readable report
    python cli.py src/ --format sarif -o report.sarif      # GitHub code scanning / VS Code SARIF viewer
    python cli.py src/ --format json
    python cli.py src/ --fail-on high                      # exit code 1 if any High/Critical finding (default)
    python cli.py src/ --fail-on medium | --fail-on none

Runs the rule engine only (no ML models, no server), so it is fast and needs no training. Findings can be silenced with
`// intellireview: ignore` comments (see analyzers/suppress.py). Exit codes: 0 ok, 1 findings at/above --fail-on, 2 usage error.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzers.issue_taxonomy import risk_level_from_issues  # noqa: E402
from pipeline import static_analysis  # noqa: E402

EXT_LANG = {".c": "C", ".h": "C", ".cpp": "C", ".cc": "C", ".cxx": "C", ".hpp": "C", ".java": "JAVA"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", "target", "out"}
SEV_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
SARIF_LEVEL = {"CRITICAL": "error", "HIGH": "error", "MEDIUM": "warning", "LOW": "note"}
LEVEL_TEXT = ["Clean", "Moderate risk", "High risk"]
MAX_BYTES = 200_000


def collect(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                files += [os.path.join(root, n) for n in sorted(names) if os.path.splitext(n)[1].lower() in EXT_LANG]
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"intellireview: no such file or directory: {p}", file=sys.stderr)
    return files


def review(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    lang = EXT_LANG.get(ext)
    if lang is None:
        return {"path": path, "error": f"unsupported file type '{ext}'"}
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return {"path": path, "error": f"skipped: larger than {MAX_BYTES:,} bytes"}
        with open(path, encoding="utf-8", errors="replace") as f:
            code = f.read()
        r = static_analysis(code, lang)
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
        return {"path": path, "error": f"{type(e).__name__}: {e}"}
    issues = sorted(r["all_issues"], key=lambda i: (-SEV_RANK.get(i["severity"], 0), i.get("line", 0)))
    return {"path": path, "language": lang, "level": risk_level_from_issues(issues), "issues": issues,
            "suppressed": len(r["suppressed"]), "notes": r["notes"], "mode": r["parse_result"].get("analysis_mode")}


def to_sarif(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rules: Dict[str, Dict[str, Any]] = {}
    out = []
    for r in results:
        for i in r.get("issues", []):
            rid = i["type"]
            rules.setdefault(rid, {"id": rid, "name": rid.title().replace("_", ""),
                                   "shortDescription": {"text": rid.replace("_", " ").capitalize()},
                                   "help": {"text": i.get("suggestion") or "See the message."}})
            res = {"ruleId": rid, "level": SARIF_LEVEL.get(i["severity"], "warning"),
                   "message": {"text": i["message"] + (f" Fix: {i['suggestion']}" if i.get("suggestion") else "")},
                   "locations": [{"physicalLocation": {"artifactLocation": {"uri": r["path"].replace(os.sep, "/")},
                                                       "region": {"startLine": max(1, int(i.get("line") or 1))}}}]}
            out.append(res)
    return {"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
            "runs": [{"tool": {"driver": {"name": "IntelliReview", "informationUri": "https://github.com/MadhurSikarwar/C---Java-Code-Reviewer",
                                          "rules": list(rules.values())}}, "results": out}]}


def to_text(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for r in results:
        if "error" in r:
            lines.append(f"{r['path']}: {r['error']}")
            continue
        head = f"{r['path']}  —  {LEVEL_TEXT[r['level']]}"
        if r["suppressed"]:
            head += f"  ({r['suppressed']} suppressed)"
        lines.append(head)
        for i in r["issues"]:
            lines.append(f"  {r['path']}:{i.get('line') or 1}: {i['severity'].lower():8} {i['type'].lower().replace('_', ' ')}: {i['message']}")
            if i.get("suggestion"):
                lines.append(f"      fix: {i['suggestion']}")
        if r["mode"] == "heuristic":
            lines.append("  note: could not be fully parsed; heuristic analysis only")
    total = sum(len(r.get("issues", [])) for r in results)
    high = sum(1 for r in results if r.get("level") == 2)
    lines.append(f"\n{len(results)} file(s), {total} finding(s), {high} file(s) at High risk")
    return "\n".join(lines)


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(prog="intellireview", description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="files and/or directories")
    ap.add_argument("--format", choices=["text", "json", "sarif"], default="text")
    ap.add_argument("--fail-on", choices=["none", "medium", "high"], default="high",
                    help="exit with status 1 if a finding of this severity or worse exists (default: high)")
    ap.add_argument("-o", "--output", help="write the report to this file instead of stdout")
    args = ap.parse_args(argv)

    files = collect(args.paths)
    if not files:
        print("intellireview: nothing to review", file=sys.stderr)
        return 2
    results = [review(f) for f in files]

    report = {"text": to_text, "json": lambda r: json.dumps(r, indent=2, default=str),
              "sarif": lambda r: json.dumps(to_sarif(r), indent=2)}[args.format](results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
    else:
        print(report)

    threshold = {"none": 99, "medium": 2, "high": 3}[args.fail_on]
    worst = max((SEV_RANK.get(i["severity"], 0) for r in results for i in r.get("issues", [])
                 if i["type"] not in ("HIGH_COMPLEXITY", "DEEP_NESTING", "DEAD_CODE", "RECURSION")), default=0)
    return 1 if worst >= threshold else 0


if __name__ == "__main__":
    sys.exit(main())
