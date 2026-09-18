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


def resolve_conditionals(text: str) -> str:
    out: List[str] = []
    stack: List[List[bool]] = []          # [parent_active, taken, active]
    active = True
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            d = re.match(r"#\s*(\w+)\s*(.*)", s)
            kw, rest = (d.group(1), d.group(2)) if d else ("", "")
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
        out.append(line if active else "")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------
# function extraction
# ------------------------------------------------------------------------------------------------------------
_C_FUNC = re.compile(r"(?m)^[ \t]*(?:static[ \t]+)?(?:const[ \t]+)?(?:[\w]+[ \t\*]+)+?(\w+)[ \t]*\(([^;{}()]*)\)[ \t]*\n?[ \t]*\{")
_J_FUNC = re.compile(r"(?m)^[ \t]*(?:public|private|protected)[ \t]+(?:static[ \t]+)?(?:final[ \t]+)?[\w<>\[\]]+[ \t]+(\w+)[ \t]*\(([^;{}()]*)\)[ \t]*(?:throws[ \t]+[\w.,\s]+?)?[ \t]*\n?[ \t]*\{")


def _match_brace(code: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(code)):
        c = code[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _is_bad(name: str) -> bool:
    return bool(re.search(r"(?:^|_)bad$", name)) or name == "bad"


def _is_good_variant(name: str) -> bool:
    # good() itself only dispatches to the other good* functions -> skip; keep goodG2B, goodB2G, good1, good2 ...
    if re.search(r"(?:^|_)good$", name) or name == "good":
        return False
    return bool(re.search(r"(?:^|_)good(?:G2B|B2G|\d+|G2B\d+|B2G\d+|B2B\d*|)\w*$", name)) and "Sink" not in name \
        and "Source" not in name


def extract_functions(code: str, language: str) -> List[Tuple[str, str, str]]:
    """Return [(name, params, full_function_text)] for the *bad*/*good* functions in `code` (already comment-free)."""
    rx = _C_FUNC if language == "C" else _J_FUNC
    out = []
    for m in rx.finditer(code):
        name, params = m.group(1), m.group(2).strip()
        if name in ("if", "for", "while", "switch", "return", "sizeof"):
            continue
        if not (_is_bad(name) or _is_good_variant(name)):
            continue
        if params not in ("", "void") and language == "C":
            continue                 # takes input from a caller (sink half of a multi-file flow)
        open_idx = code.index("{", m.end() - 1)
        close = _match_brace(code, open_idx)
        if close == -1:
            continue
        out.append((name, params, code[m.start():close + 1].strip("\n")))
    return out


_FILE_DECL = re.compile(r"^\s*static\s+(?:const\s+)?(?:int|long|size_t|unsigned(?:\s+int)?)\s+\w+\s*=\s*[^;{}]+;\s*$")


def file_level_constants(code: str) -> List[str]:
    """`static int staticTrue = 1;`-style declarations at file scope (brace depth 0): the flow-variant constants."""
    out, depth = [], 0
    for line in code.split("\n"):
        if depth == 0 and _FILE_DECL.match(line):
            out.append(line.strip())
        depth += line.count("{") - line.count("}")
    return out


def _rename(fn_text: str, name: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(name)}\b", new, fn_text)


# ------------------------------------------------------------------------------------------------------------
def _cwe_of(path: str) -> Optional[int]:
    m = re.search(r"CWE(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def _single_file_variant(fname: str) -> bool:
    return bool(re.search(r"_\d{2}\.(?:c|java)$", fname))


def collect(root: str, language: str, train_cwes: Dict[int, int], max_files_per_cwe: int, seed: int,
            max_good_only_cwes_files: int) -> List[dict]:
    ext = ".c" if language == "C" else ".java"
    rng = random.Random(seed)
    samples: List[dict] = []
    stats = {"files": 0, "bad": 0, "good": 0}
    for cwe_dir in sorted(os.listdir(root)):
        m = re.match(r"CWE(\d+)_", cwe_dir)
        if not m:
            continue
        cwe = int(m.group(1))
        base = os.path.join(root, cwe_dir)
        files: List[str] = []
        for dp, _, fns in os.walk(base):
            for fn in fns:
                if fn.endswith(ext) and _single_file_variant(fn) and "w32" not in fn.lower():
                    files.append(os.path.join(dp, fn))
        files.sort()
        rng.shuffle(files)
        trained = cwe in train_cwes
        files = files[:max_files_per_cwe if trained else max_good_only_cwes_files]
        for path in files:
            try:
                raw = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            code = sanitize(resolve_conditionals(raw), keep_strings=True, java_text_blocks=(language != "C"))
            fns = extract_functions(code, language)
            decls = file_level_constants(code) if language == "C" else []
            stats["files"] += 1
            for idx, (name, params, text) in enumerate(fns):
                kind = "bad" if _is_bad(name) else "good"
                if kind == "bad" and not trained:
                    continue                    # positives outside the supported CWE list are not used for training
                body = _rename(text, name, f"fn_{idx}")
                if language == "C":
                    src = C_STUB_TYPES + "\n" + "\n".join(decls) + "\n" + body + "\n"
                else:
                    src = "public class T {\n" + body + "\n}\n"
                label = 0 if kind == "good" else train_cwes[cwe]
                samples.append({"source": src, "body": body, "decls": decls, "language": "C" if language == "C" else "JAVA", "cwe": cwe,
                                "label": label, "kind": kind, "file": os.path.basename(path), "group": f"{language}-{cwe}"})
                stats[kind] += 1
    print(f"[{language}] files={stats['files']}  bad={stats['bad']}  good={stats['good']}")
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
