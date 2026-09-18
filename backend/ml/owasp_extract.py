"""
IntelliReview - OWASP Benchmark (Java) extractor  ->  EXTERNAL TEST SET (never trained on)
==========================================================================================
The OWASP Benchmark v1.2 has ~2,740 small servlets. `expectedresults-1.2.csv` says, for every one, whether it really contains
the vulnerability (`true`) or only looks like it (`false`: a constant reaches the sink, the tainted value is dropped, the
branch is dead, a safe API is used ...).  Those labels were assigned by the benchmark's authors, not by this project's rules
or models, so they are independent ground truth.

Mapping to our classes (documented so it can be challenged):
    injection style   sqli, cmdi, xss, pathtraver, ldapi, xpathi   real=true -> 2 (High)
    weak-by-design    crypto, hash, weakrand, securecookie, trustbound   real=true -> 1 (Moderate)   [same as Juliet CWE 327/328 -> 1]
    real=false                                                   -> 0 (Clean)
`covered` marks the categories the analyser has a rule family for; the rest are reported but flagged as out of scope.

Anti-leakage: comments are stripped, the @WebServlet annotation (it contains the category name) and every `/<category>-NN/`
URL fragment are removed, the class name (BenchmarkTest00042) is renamed.  Only the whole servlet file is a sample, so no
function-level split can hide a sink from a source.

USAGE (from backend/):
    python ml/owasp_extract.py --root "C:/.../BenchmarkJava"
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.text_utils import sanitize

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "owasp_samples.jsonl")
HIGH_CATS = {"sqli", "cmdi", "xss", "pathtraver", "ldapi", "xpathi"}
MODERATE_CATS = {"crypto", "hash", "weakrand", "securecookie", "trustbound"}
COVERED = {"sqli", "cmdi", "xss", "pathtraver", "crypto", "hash"}


def clean(source: str, name: str) -> str:
    code = sanitize(source, keep_strings=True, java_text_blocks=True)
    code = re.sub(r"(?m)^[ \t]*@WebServlet\([^\n]*\)[ \t]*\n", "", code)
    code = re.sub(r"/[a-z]+-\d\d/", "/x/", code)
    code = code.replace(name, "T")
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="checkout of OWASP-Benchmark/BenchmarkJava")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    tdir = os.path.join(args.root, "src", "main", "java", "org", "owasp", "benchmark", "testcode")
    rows = []
    with open(os.path.join(args.root, "expectedresults-1.2.csv"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) < 4 or r[0].startswith("#"):
                continue
            rows.append((r[0].strip(), r[1].strip(), r[2].strip().lower() == "true", int(r[3])))
    out, missing = [], 0
    for name, cat, real, cwe in rows:
        p = os.path.join(tdir, name + ".java")
        if not os.path.exists(p):
            missing += 1
            continue
        src = clean(open(p, encoding="utf-8", errors="replace").read(), name)
        label = 0 if not real else (2 if cat in HIGH_CATS else 1)
        out.append({"source": src, "language": "JAVA", "src": "owasp", "cwe": cwe, "label": label,
                    "kind": "bad" if real else "good", "file": name + ".java", "group": f"JAVA-owasp-{cat}",
                    "category": cat, "covered": cat in COVERED})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    bad = sum(1 for s in out if s["kind"] == "bad")
    print(f"wrote {len(out)} samples ({bad} vulnerable / {len(out) - bad} safe, {missing} listed but missing) -> {args.out}")


if __name__ == "__main__":
    main()
