"""
IntelliReview — how good is the RULE ENGINE on independent, expert-labeled data?

Reads ml/data/features.csv (built by ml/build_features.py) and reports, for the Juliet function-level samples:
  * parse coverage (real AST vs heuristic fallback)
  * per-CWE recall  : fraction of vulnerable functions the rule engine flags at the right level (or any level)
  * false-alarm rate: fraction of PATCHED (good) functions that are flagged at all / flagged High Risk

Run from backend/:  python ml/evaluate_rules.py [--src juliet]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "features.csv")

CWE_NAMES = {
    121: "Stack buffer overflow", 122: "Heap buffer overflow", 124: "Buffer underwrite", 126: "Buffer over-read",
    127: "Buffer under-read", 134: "Format string", 242: "Dangerous function (gets)", 415: "Double free",
    416: "Use after free", 476: "NULL deref", 562: "Return stack address", 590: "Free non-heap", 676: "Dangerous function",
    690: "Unchecked ret -> NULL deref", 761: "Free not at buffer start", 78: "OS command injection", 401: "Memory leak",
    457: "Uninitialized variable", 775: "Missing file close", 773: "Missing FD close", 89: "SQL injection", 80: "XSS",
    83: "XSS (attribute)", 23: "Relative path traversal", 36: "Absolute path traversal", 259: "Hard-coded password",
    327: "Broken crypto", 328: "Weak hash", 772: "Missing resource release", 404: "Improper resource shutdown",
    390: "Error without action",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="juliet")
    args = ap.parse_args()
    df = pd.read_csv(CSV)
    df = df[df["src"] == args.src].copy()
    print(f"{len(df)} '{args.src}' samples\n")

    print("Parse mode (ast = full path-sensitive analysis, heuristic = regex fallback):")
    print(df.groupby(["language", "analysis_mode"]).size().unstack(fill_value=0).to_string(), "\n")

    for lang in ("C", "JAVA"):
        d = df[df["language"] == lang]
        if d.empty:
            continue
        good = d[d["kind"] == "good"]
        bad = d[d["kind"] == "bad"]
        print(f"=== {lang}  ({len(bad)} vulnerable, {len(good)} patched functions)")
        print(f"  false alarms on patched code:  any finding {100 * (good.rule_label > 0).mean():5.1f}%   "
              f"High Risk {100 * (good.rule_label == 2).mean():5.1f}%")
        print(f"  recall on vulnerable code:     flagged at all {100 * (bad.rule_label > 0).mean():5.1f}%   "
              f"right level {100 * (bad.rule_label == bad.label).mean():5.1f}%   "
              f"High-Risk recall {100 * (bad[bad.label == 2].rule_label == 2).mean():5.1f}%")
        print(f"  {'CWE':>5}  {'name':32} {'n':>4}  {'flagged':>8}  {'right lvl':>9}  patched-FP")
        for cwe, g in bad.groupby("cwe"):
            gg = good[good.cwe == cwe]
            fp = f"{100 * (gg.rule_label > 0).mean():5.1f}%" if len(gg) else "   n/a"
            print(f"  {int(cwe):>5}  {CWE_NAMES.get(int(cwe), '?'):32} {len(g):>4}  "
                  f"{100 * (g.rule_label > 0).mean():7.1f}%  {100 * (g.rule_label == g.label).mean():8.1f}%  {fp:>8}")
        print()


if __name__ == "__main__":
    main()
