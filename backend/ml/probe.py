"""
IntelliReview - fast rule-engine probe (no CSV, no models)
===========================================================
Runs the analysis pipeline on a JSONL sample file and prints, per CWE / category, recall and false alarms of the RULE ENGINE,
and optionally the source of misses (a vulnerable sample with no finding) or false alarms (a safe sample with a finding).

    python ml/probe.py --data juliet --lang C --cwe 78                 # table for one CWE
    python ml/probe.py --data juliet --lang C --cwe 78 --show good-fp  # print 2 false alarms with their findings
    python ml/probe.py --data owasp                                    # external OWASP Benchmark table
    python ml/probe.py --data juliet --lang JAVA --per-cwe 25          # quick sub-sample of every CWE
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _work(s):
    from pipeline import static_analysis
    from analyzers.issue_taxonomy import risk_level_from_issues
    try:
        r = static_analysis(s["source"], s["language"])
        return s, risk_level_from_issues(r["all_issues"]), r["all_issues"], r["parse_result"].get("analysis_mode")
    except Exception as e:  # noqa: BLE001
        return s, -1, [{"type": "CRASH", "severity": "?", "line": 0, "message": f"{type(e).__name__}: {e}"}], "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="juliet", choices=["juliet", "owasp"])
    ap.add_argument("--lang", choices=["C", "JAVA"])
    ap.add_argument("--cwe", type=int)
    ap.add_argument("--category")
    ap.add_argument("--per-cwe", type=int, default=0, help="random sub-sample per CWE/category (0 = all)")
    ap.add_argument("--show", choices=["bad-miss", "good-fp", "bad-wrong"], help="print sources of such cases")
    ap.add_argument("--n-show", type=int, default=2)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--by", choices=["cwe", "variant"], default="cwe", help="row key (variant = flow-variant number of the Juliet file)")
    ap.add_argument("--fp-types", action="store_true", help="count the finding types behind false alarms / hits")
    ap.add_argument("--match", help="regex on the file name")
    ap.add_argument("--file", help="explicit JSONL path (default: ml/data/<data>_samples.jsonl)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    args = ap.parse_args()

    path = args.file or os.path.join(DATA, "juliet_samples.jsonl" if args.data == "juliet" else "owasp_samples.jsonl")
    samples = [json.loads(l) for l in open(path, encoding="utf-8")]
    if args.lang:
        samples = [s for s in samples if s["language"] == args.lang]
    if args.cwe is not None:
        samples = [s for s in samples if s["cwe"] == args.cwe]
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
    if args.match:
        import re as _re2
        samples = [s for s in samples if _re2.search(args.match, s["file"])]
    key = (lambda s: s.get("category") or s["cwe"]) if args.data == "owasp" else (lambda s: (s["language"], s["cwe"]))
    if args.by == "variant":
        import re as _re
        def key(s):
            m = _re.search(r"_(\d\d)[a-z]?\.(?:c|java)$", s["file"])
            return (s["language"], m.group(1) if m else "??")
    if args.per_cwe:
        rng = random.Random(7)
        by = defaultdict(list)
        for s in samples:
            by[(key(s), s["kind"])].append(s)
        samples = [x for v in by.values() for x in rng.sample(v, min(args.per_cwe, len(v)))]

    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(_work, samples, chunksize=16)

    stats = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])   # bad n, flagged, right level, high recall den, high recall num, good n, good fp
    shown = 0
    for s, lvl, issues, mode in res:
        st = stats[key(s)]
        if s["kind"] == "bad":
            st[0] += 1
            st[1] += lvl > 0
            st[2] += lvl == s["label"]
            if s["label"] == 2:
                st[3] += 1
                st[4] += lvl == 2
        else:
            st[5] += 1
            st[6] += lvl > 0
        want = (args.show == "bad-miss" and s["kind"] == "bad" and lvl == 0) or \
               (args.show == "bad-wrong" and s["kind"] == "bad" and lvl != s["label"]) or \
               (args.show == "good-fp" and s["kind"] == "good" and lvl > 0)
        if want and shown < args.n_show + args.skip:
            shown += 1
            if shown <= args.skip:
                continue
            print("=" * 100)
            print(f"{s['file']}  cwe={s['cwe']}  label={s['label']}  rule_level={lvl}  parse={mode}")
            body = s.get("body") or s["source"]
            if s["language"] == "C" and s.get("body"):
                body = "\n".join(s.get("decls", [])) + "\n" + body
            print(body)
            print("-- findings:")
            for i in issues:
                print(f"   {i['severity']:8} {i['type']:22} line {i.get('line')}: {i['message'][:150]}")
    print(f"\n{'key':>22} {'bad':>5} {'flagged':>8} {'right':>7} {'HighRec':>8} | {'good':>5} {'FP':>6}")
    tot = [0] * 7
    for k in sorted(stats, key=str):
        st = stats[k]
        for i in range(7):
            tot[i] += st[i]
        pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  n/a"
        print(f"{str(k):>22} {st[0]:>5} {pct(st[1], st[0]):>8} {pct(st[2], st[0]):>7} {pct(st[4], st[3]):>8} | {st[5]:>5} {pct(st[6], st[5]):>6}")
    pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  n/a"
    print(f"{'TOTAL':>22} {tot[0]:>5} {pct(tot[1], tot[0]):>8} {pct(tot[2], tot[0]):>7} {pct(tot[4], tot[3]):>8} | {tot[5]:>5} {pct(tot[6], tot[5]):>6}")
    if args.fp_types:
        from collections import Counter
        from analyzers.issue_taxonomy import QUALITY_TYPES
        fp, tp = Counter(), Counter()
        for s_, lvl, issues, _m in res:
            ev = {i["type"] + ":" + i["severity"] for i in issues
                  if i["severity"] in ("CRITICAL", "HIGH", "MEDIUM") and i["type"] not in QUALITY_TYPES}
            if s_["kind"] == "good" and lvl > 0:
                fp.update(ev)
            elif s_["kind"] == "bad" and lvl > 0:
                tp.update(ev)
        print("\nfalse-alarm evidence (good samples):", fp.most_common(14))
        print("hit evidence (bad samples):", tp.most_common(14))
    crashes = sum(1 for r in res if r[1] == -1)
    heur = sum(1 for r in res if r[3] == "heuristic")
    print(f"samples={len(res)}  crashes={crashes}  heuristic-parse={heur}")


if __name__ == "__main__":
    main()
