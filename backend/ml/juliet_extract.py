"""
IntelliReview — NIST Juliet test-suite extractor
================================================
Turns the Juliet C and Java test suites into FUNCTION-LEVEL labeled samples:

    <name>_bad(), bad()                       -> vulnerable
    good*(), goodG2B*(), goodB2G*(), ...       -> patched / safe   (label 0 = Clean)

The old collectors labelled by *file name*, but nearly every Juliet file contains BOTH a bad() and its good()
counterparts, so filename labelling is mostly wrong. Function-level labelling is the ground truth Juliet provides.

Only self-contained, single-file flow variants are used (file names ending in _01 .. _99, no a/b/c suffix), because in
the multi-file variants the flaw is split between a source file and a sink file and cannot be seen locally.

Comments are stripped BEFORE analysis: Juliet comments literally say "POTENTIAL FLAW" / "FIX", which would leak the label.
Function names are replaced too (they contain "bad" / "good").

Output: JSON-lines, one sample per line:
    {"source", "language", "cwe", "label", "kind": "bad"|"good", "file", "group"}

USAGE (from backend/):
    python ml/juliet_extract.py \
        --c-root    "C:/.../juliet-test-suite-for-c-cplusplus-v1-3/C/testcases" \
        --java-root "C:/.../juliet-test-suite-for-java-v1-3/Java/src/testcases" \
        --max-files-per-cwe 60
"""
import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.text_utils import sanitize

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "juliet_samples.jsonl")

# CWE -> class of the *bad* variant (2 = High Risk, 1 = Moderate Risk). Only CWEs that the analysis pipeline is designed
# to see are used as positives for TRAINING; everything else is still used for good (Clean) samples and for the coverage report.
C_TRAIN_CWES: Dict[int, int] = {
    121: 2, 122: 2, 124: 2, 126: 2, 127: 2, 134: 2, 242: 2, 415: 2, 416: 2, 476: 2, 562: 2, 590: 2, 676: 2,
    690: 2, 761: 2, 78: 2,
    401: 1, 457: 1, 775: 1, 773: 1,
}
JAVA_TRAIN_CWES: Dict[int, int] = {
    89: 2, 78: 2, 80: 2, 83: 2, 23: 2, 36: 2,
    259: 1, 327: 1, 328: 1, 772: 1, 404: 1, 390: 1,
}

C_STUB_TYPES = (
    "typedef struct _twoIntsStruct { int intOne; int intTwo; } twoIntsStruct; "
    "typedef struct _charVoid { char charFirst[16]; void *voidSecond; void *voidThird; } charVoid; "
    "typedef unsigned int uint; typedef long time_t; typedef int pid_t; typedef int socklen_t; typedef int SOCKET; "
    "typedef unsigned char BYTE; typedef int HANDLE; typedef struct _fs { int a; } fs_t;\n"
    # constants that std_testcase.h defines (flow-variant switches); a real analyzer would resolve the #include
    "static const int GLOBAL_CONST_TRUE = 1; static const int GLOBAL_CONST_FALSE = 0; static const int GLOBAL_CONST_FIVE = 5;\n"
    "static int globalTrue = 1; static int globalFalse = 0; static int globalFive = 5;\n"
)


# ------------------------------------------------------------------------------------------------------------
# tiny conditional-compilation resolver (Juliet wraps everything in #ifndef OMITBAD / #ifdef _WIN32 ...)
# ------------------------------------------------------------------------------------------------------------
def _eval_pp(expr: str) -> bool:
    e = expr.strip()
    if re.search(r"\b_WIN32\b|\bWIN32\b|\b_MSC_VER\b", e) and not re.search(r"!\s*defined", e):
        return False
    if e in ("0",):
        return False
    m = re.fullmatch(r"defined\s*\(?\s*(\w+)\s*\)?", e)
    if m:
        return False
    m = re.fullmatch(r"!\s*defined\s*\(?\s*(\w+)\s*\)?", e)
    if m:
        return True
    return True


# macros std_testcase.h defines (the real analyser would resolve the #include)
_BUILTIN_MACROS = {"ALLOCA": "alloca", "true": "1", "false": "0"}
_DEFINE = re.compile(r"#\s*define\s+(\w+)(?!\()[ \t]+(.+)$")


def _expand(line: str, macros: Dict[str, str]) -> str:
    """Object-like macro expansion (a few rounds, so macros that name other macros resolve)."""
    if not macros:
        return line
    rx = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in macros) + r")\b")
    for _ in range(4):
        new = rx.sub(lambda m: macros[m.group(0)], line)
        if new == line:
            break
        line = new
    return line


def resolve_conditionals(text: str) -> str:
    """Resolve #ifdef/#if (Linux, everything compiled in) and expand object-like #define macros, as a preprocessor would.
    The analyser itself never sees directives; doing this here mimics resolving the headers of real code."""
    out: List[str] = []
    stack: List[List[bool]] = []          # [parent_active, taken, active]
    active = True
    macros: Dict[str, str] = dict(_BUILTIN_MACROS)
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            d = re.match(r"#\s*(\w+)\s*(.*)", s)
            kw, rest = (d.group(1), d.group(2)) if d else ("", "")
            if kw == "define" and active:
                md = _DEFINE.match(s)
                if md:
                    body = re.sub(r"/\*.*?\*/|//.*$", "", md.group(2)).strip()
                    if body and not body.endswith("\\"):
                        macros[md.group(1)] = _expand(body, macros)
            if kw in ("ifdef", "ifndef", "if"):
                if kw == "ifdef":
                    cond = False
                elif kw == "ifndef":
                    cond = True
                else:
                    cond = _eval_pp(rest)
                stack.append([active, cond, active and cond])
                active = stack[-1][2]
                out.append("")
                continue
            if kw in ("else", "elif") and stack:
                parent, taken, _ = stack[-1]
                cond = (not taken) if kw == "else" else (not taken and _eval_pp(rest))
                stack[-1][1] = taken or cond
                stack[-1][2] = parent and cond
                active = stack[-1][2]
                out.append("")
                continue
            if kw == "endif" and stack:
                parent = stack.pop()[0]
                active = parent
                out.append("")
                continue
            out.append("")           # #include / #define / #pragma ...
            continue
        out.append(_expand(line, macros) if active else "")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------
# labelled-function detection
# ------------------------------------------------------------------------------------------------------------
def _is_bad(name: str) -> bool:
    return bool(re.search(r"(?:^|_)bad$", name)) or name == "bad"


def _is_good_variant(name: str) -> bool:
    # good() itself only dispatches to the other good* functions -> skip; keep goodG2B, goodB2G, good1, good2 ...
    if re.search(r"(?:^|_)good$", name) or name == "good":
        return False
    return bool(re.search(r"(?:^|_)good(?:G2B|B2G|\d+|G2B\d+|B2G\d+|B2B\d*|)\w*$", name)) and "Sink" not in name \
        and "Source" not in name


def _cwe_of(path: str) -> Optional[int]:
    m = re.search(r"CWE(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


# functions the Juliet support library (testcasesupport/io.c) provides; included in a sample only when it uses them
C_SUPPORT_FUNCS = ("int globalReturnsTrue() { return 1; }\n int globalReturnsFalse() { return 0; }\n"
                   "int globalReturnsTrueOrFalse() { return (rand() % 2); }\n")

# stubs for the support classes/headers the test cases import (the analyser would resolve the import)
JAVA_STUB = (
    "class IO { static final boolean STATIC_FINAL_TRUE = true; static final boolean STATIC_FINAL_FALSE = false; "
    "static final int STATIC_FINAL_FIVE = 5; static boolean staticTrue = true; static boolean staticFalse = false; "
    "static int staticFive = 5; static boolean staticReturnsTrue() { return true; } "
    "static boolean staticReturnsFalse() { return false; } }\n"
)


def _group_key(fname: str, ext: str) -> Optional[str]:
    """`X_51a.c`, `X_51b.c` -> `X_51` ; `X_07.c` -> `X_07` ; anything else (no variant number) -> None."""
    m = re.search(r"^(.*_\d{2})[a-z]?\." + ext + "$", fname)
    return m.group(1) if m else None


def collect(root: str, language: str, train_cwes: Dict[int, int], max_files_per_cwe: int, seed: int,
            max_good_only_cwes_files: int, multi_file: bool = True) -> List[dict]:
    from ml.juliet_units import split_items, class_body, build_sample
    ext = "c" if language == "C" else "java"
    rng = random.Random(seed)
    samples: List[dict] = []
    stats = {"groups": 0, "bad": 0, "good": 0, "multi": 0}
    for cwe_dir in sorted(os.listdir(root)):
        m = re.match(r"CWE(\d+)_", cwe_dir)
        if not m:
            continue
        cwe = int(m.group(1))
        base = os.path.join(root, cwe_dir)
        groups: Dict[str, List[str]] = {}
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if not fn.endswith("." + ext) or "w32" in fn.lower():
                    continue
                key = _group_key(fn, ext)
                if key is None:
                    continue
                if not multi_file and not re.search(r"_\d{2}\." + ext + "$", fn):
                    continue
                groups.setdefault(os.path.join(dp, key), []).append(os.path.join(dp, fn))
        keys = sorted(groups)
        rng.shuffle(keys)
        trained = cwe in train_cwes
        keys = keys[:max_files_per_cwe if trained else max_good_only_cwes_files]
        for key in keys:
            paths = sorted(groups[key])
            items, ok = [], True
            for path in paths:
                try:
                    raw = open(path, encoding="utf-8", errors="replace").read()
                except OSError:
                    ok = False
                    break
                code = sanitize(resolve_conditionals(raw), keep_strings=True, java_text_blocks=(language != "C"))
                if language == "C":
                    items += split_items(code, "C")
                else:
                    cb = class_body(code)
                    if cb is None:
                        ok = False
                        break
                    items += split_items(cb[1], "JAVA")
            if not ok:
                continue
            if language == "C":
                items += split_items(C_SUPPORT_FUNCS, "C")       # what testcasesupport/io.c defines
            stats["groups"] += 1
            stats["multi"] += len(paths) > 1
            labelled = [it for it in items if it["kind"] == "func" and it["name"]
                        and (_is_bad(it["name"]) or _is_good_variant(it["name"]))
                        and (language != "C" or it["params"] in ("", "void"))]   # C: params => the sink half of a flow
            for idx, target in enumerate(labelled):
                kind = "bad" if _is_bad(target["name"]) else "good"
                if kind == "bad" and not trained:
                    continue                    # positives outside the supported CWE list are not used for training
                built = build_sample(items, target, idx, language)
                if built is None:
                    continue
                body = built["body"]
                if language == "C":
                    src = C_STUB_TYPES + "\n" + "\n".join(built["types"]) + "\n" + body + "\n"
                else:
                    src = "public class T {\n" + body + "\n}\n" + JAVA_STUB
                label = 0 if kind == "good" else train_cwes[cwe]
                samples.append({"source": src, "body": body, "decls": built["types"] if language == "C" else [],
                                "language": "C" if language == "C" else "JAVA", "cwe": cwe,
                                "label": label, "kind": kind, "file": os.path.basename(paths[0]), "group": f"{language}-{cwe}"})
                stats[kind] += 1
    print(f"[{language}] units={stats['groups']} (multi-file {stats['multi']})  bad={stats['bad']}  good={stats['good']}")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-root")
    ap.add_argument("--java-root")
    ap.add_argument("--max-files-per-cwe", type=int, default=60)
    ap.add_argument("--max-files-good-only", type=int, default=15,
                    help="files sampled per CWE that is not a training positive (only their good functions are used)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    samples: List[dict] = []
    if args.c_root:
        samples += collect(args.c_root, "C", C_TRAIN_CWES, args.max_files_per_cwe, args.seed, args.max_files_good_only)
    if args.java_root:
        samples += collect(args.java_root, "JAVA", JAVA_TRAIN_CWES, args.max_files_per_cwe, args.seed, args.max_files_good_only)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} samples -> {args.out}")


if __name__ == "__main__":
    main()
