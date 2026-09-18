"""
IntelliReview — training-set builder
====================================
Builds ml/data/features.csv: one row per labeled source file, with the 36 pipeline features, the label, and what the
RULE ENGINE alone concluded (so models can be compared against it, and against a "size only" baseline).

Sources of samples
  juliet     function-level Juliet _bad / _good functions (ml/juliet_extract.py)   — independent, expert labels
  composite  Juliet functions concatenated into bigger files: 0..N good functions + at most ONE real vulnerable one.
             Label = label of that vulnerable function (or Clean).  Because N is drawn independently of the label,
             program size cannot predict the label — this is what removes the "bigger = riskier" shortcut.
  generated  random programs from ml/program_generator.py (works even without Juliet)

Run from backend/:   python ml/build_features.py [--generated 6000] [--composites 4000] [--workers 6]
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import warnings
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JULIET = os.path.join(DATA_DIR, "juliet_samples.jsonl")
OUT = os.path.join(DATA_DIR, "features.csv")


def rule_label(issues: List[dict]) -> int:
    """What the rule engine alone would answer (same gate the API applies)."""
    from analyzers.issue_taxonomy import risk_level_from_issues
    return risk_level_from_issues(issues)


def _work(sample: dict) -> dict:
    from pipeline import static_analysis, all_feature_names
    names = all_feature_names()
    try:
        r = static_analysis(sample["source"], sample["language"])
    except Exception as e:  # noqa: BLE001 - keep going, record the failure
        return {"_error": f"{type(e).__name__}: {e}", **{k: sample.get(k) for k in ("kind_src", "language", "label", "group")}}
    row = {
        "src": sample["src"], "language": sample["language"], "label": sample["label"], "group": sample["group"],
        "cwe": sample.get("cwe", ""), "kind": sample.get("kind", ""),
        "analysis_mode": r["parse_result"].get("analysis_mode", ""),
        "rule_label": rule_label(r["all_issues"]),
        "issue_types": "|".join(sorted({i["type"] + ":" + i["severity"] for i in r["all_issues"]})),
        "loc": r["parse_result"].get("lines_of_code", 0),
    }
    row.update({n: v for n, v in zip(names, r["feature_vector"])})
    return row


# ------------------------------------------------------------------------------------------------------------
def make_composites(juliet: List[dict], n: int, rng: random.Random) -> List[dict]:
    from ml.juliet_extract import C_STUB_TYPES
    from ml.program_generator import C_SAFE, J_SAFE, _name
    out: List[dict] = []
    for lang in ("C", "JAVA"):
        goods = [s for s in juliet if s["language"] == lang and s["kind"] == "good"]
        highs = [s for s in juliet if s["language"] == lang and s["label"] == 2]
        mods = [s for s in juliet if s["language"] == lang and s["label"] == 1]
        if not goods:
            continue
        for k in range(n // 2):
            r = rng.random()
            bad = None
            if r > 0.55 and highs:
                bad = rng.choice(highs)
            elif r > 0.30 and mods:
                bad = rng.choice(mods)
            n_good = min(45, int(rng.expovariate(1 / 7)))
            chosen = [rng.choice(goods) for _ in range(n_good)]
            parts = [c["body"] for c in chosen]
            n_gen = min(25, int(rng.expovariate(1 / 4)))
            safe_fns = C_SAFE if lang == "C" else J_SAFE
            parts += [rng.choice(safe_fns)(_name(rng, "s"), rng) for _ in range(n_gen)]
            if bad:
                parts.append(bad["body"])
                chosen.append(bad)
            decls = sorted({d for c in chosen for d in c.get("decls", [])})
            rng.shuffle(parts)
            uniq = []
            for i, p in enumerate(parts):
                # main function `fn_<n>` and its helpers `fn_<n>_<k>` become `fn_<i>_<n>[_<k>]`: unique per part
                uniq.append(re.sub(r"\bfn_(\d+)((?:_\d+)?)\b", lambda m, i=i: f"fn_{i}_{m.group(1)}{m.group(2)}", p))
            if lang == "C":
                src = ("#include <stdio.h>\n" + C_STUB_TYPES + "\n" + "\n".join(decls) + "\n\n" + "\n\n".join(uniq) + "\n")
            else:
                src = "public class T {\n" + "\n\n".join(uniq) + "\n}\n"
            out.append({"source": src, "language": lang, "label": bad["label"] if bad else 0, "src": "composite",
                        "group": bad["group"] if bad else f"{lang}-clean-{k % 25}", "cwe": bad["cwe"] if bad else "",
                        "kind": "bad" if bad else "good"})
    return out


def make_generated(n: int, rng: random.Random) -> List[dict]:
    from ml.program_generator import sample_program
    out = []
    for i in range(n):
        p = sample_program(rng)
        out.append({"source": p["source"], "language": p["language"], "label": p["label"], "src": "generated",
                    "group": f"gen-{p['language']}-{i % 25}", "cwe": "", "kind": "+".join(p["defects"]) or "clean"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=int, default=6000)
    ap.add_argument("--composites", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    samples: List[dict] = []
    juliet: List[dict] = []
    if os.path.exists(JULIET):
        with open(JULIET, encoding="utf-8") as f:
            juliet = [json.loads(l) for l in f]
        for s in juliet:
            s["src"] = "juliet"
        samples += juliet
        samples += make_composites(juliet, args.composites, rng)
    else:
        print("note: no Juliet samples found — run ml/juliet_extract.py first for real data; using generated programs only")
    samples += make_generated(args.generated, rng)
    print(f"{len(samples)} samples to analyse with {args.workers} workers ...")

    from pipeline import all_feature_names
    cols = ["src", "language", "label", "group", "cwe", "kind", "analysis_mode", "rule_label", "issue_types", "loc"] + all_feature_names()
    t0 = time.time()
    rows, errors = 0, 0
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f, mp.get_context("spawn").Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(_work, samples, chunksize=64), 1):
            if "_error" in row:
                errors += 1
                continue
            w.writerow(row)
            rows += 1
            if i % 2000 == 0:
                print(f"  {i}/{len(samples)}  ({time.time() - t0:.0f}s)")
    print(f"done: {rows} rows, {errors} failed, {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
