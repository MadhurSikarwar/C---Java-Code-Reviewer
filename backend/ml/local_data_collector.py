"""
IntelliReview — Local Code Data Collector
==========================================
Reads ALL .c and .java files from a local directory tree,
runs the IntelliReview analysis pipeline on each, auto-labels them
using a scoring function, and saves to ml/real_data.csv.

The user's file at:
  C:\\Users\\Madhu\\OneDrive\\Desktop\\Development

Contains:
  - DSA/                  → mostly Moderate/High Risk (complex algorithms)
  - Programming in C/     → mix of Clean and Moderate
  - JAVA COLLEGE/         → mix of Clean and Moderate

RUN:
  cd c:\\Users\\Madhu\\OneDrive\\Desktop\\Projects\\IntelliReview\\backend
  python ml\\local_data_collector.py

No internet required. No new packages required.
"""
import os
import sys
import csv
import glob
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.synthetic_data import FEATURE_NAMES

OUTPUT_PATH    = os.path.join(os.path.dirname(__file__), "real_data.csv")
DEV_ROOT       = r"C:\Users\Madhu\OneDrive\Desktop\Development"

LABEL_TO_INT = {"Clean": 0, "Moderate Risk": 1, "High Risk": 2}

# ── Directory-level label hints ───────────────────────────────────────
# We assign a HINT based on which folder the file lives in.
# The auto-labeler (rule-based score) can override this.
FOLDER_HINTS = {
    # C folders
    "Lab Programs":           "Clean",
    "Logical Programs":       "Clean",
    "C lab Programs Practice":"Clean",
    "2D Arrays":              "Clean",
    "Arrays Programs":        "Clean",
    "File Handling":          "Moderate Risk",
    "Dynamic Memory Allocation": "Moderate Risk",
    "Pointers":               "Moderate Risk",
    "Strings":                "Clean",
    "stack":                  "Moderate Risk",
    "Queue":                  "Moderate Risk",
    "Linked Lists":           "Moderate Risk",
    "Recursion":              "Moderate Risk",
    "Heaps":                  "High Risk",
    "Graphs":                 "High Risk",
    "Trees":                  "High Risk",
    # Java folders
    "JAVA COLLEGE":           "Clean",  # college intro code = clean
}


def _get_folder_hint(filepath: str) -> str:
    """Return a label hint based on which subfolder the file is in."""
    # Check path parts against known folder hints
    parts = filepath.replace("\\", "/").split("/")
    for part in reversed(parts):
        for key, hint in FOLDER_HINTS.items():
            if key.lower() in part.lower():
                return hint
    return "Moderate Risk"  # default


def _auto_label(features: dict) -> str:
    """Rule-based scoring to assign a risk label from extracted features."""
    score = 0
    score += min(features.get("memory_leak_count", 0) * 3, 9)
    score += min(features.get("unsafe_function_count", 0) * 2, 6)
    score += min(max(0, features.get("cyclomatic_complexity", 1) - 8), 6)
    score += min(features.get("max_nesting_depth", 0) * 1, 4)
    score += min(features.get("recursion_count", 0) * 1, 3)
    score += (1 if features.get("comment_ratio", 0.5) < 0.03 else 0)
    score += min(features.get("dead_code_estimate", 0), 3)
    score += (1 if features.get("magic_number_count", 0) > 10 else 0)
    score += (1 if features.get("halstead_volume", 0) > 1500 else 0)

    if score <= 2:   return "Clean"
    elif score <= 6: return "Moderate Risk"
    else:            return "High Risk"


def _blend_label(hint: str, auto: str) -> str:
    """
    Combine folder hint and auto-label:
     - If they agree → use that label
     - If they differ by one level → use the more risky one (conservative)
     - If far apart → trust auto-label (it's based on actual code metrics)
    """
    order = {"Clean": 0, "Moderate Risk": 1, "High Risk": 2}
    h, a = order[hint], order[auto]
    diff = abs(h - a)
    if diff == 0:
        return hint
    elif diff == 1:
        # One level apart: take the higher risk (conservative)
        return hint if h > a else auto
    else:
        # Far apart: trust the auto-label (code metrics don't lie)
        return auto


def _extract(source: str, language: str) -> Optional[dict]:
    """Run full IntelliReview pipeline and return feature dict."""
    try:
        from parsers.c_parser    import parse_c_code
        from parsers.java_parser import parse_java_code
        from analyzers.memory_leak       import detect_memory_leaks
        from analyzers.unsafe_functions  import detect_unsafe_functions
        from analyzers.complexity        import analyze_complexity
        from analyzers.recursion         import detect_recursion
        from analyzers.feature_extractor import extract_features

        lang = language.upper()
        pr = parse_c_code(source) if lang == "C" else parse_java_code(source)
        mr = detect_memory_leaks(pr)
        ur = detect_unsafe_functions(pr)
        cx = analyze_complexity(pr, source)
        rc = detect_recursion(pr, source)
        fd = extract_features(pr, mr, ur, cx, rc, source=source)
        return fd["features"]
    except Exception as e:
        return None


def collect():
    # ── Discover all .c and .java files recursively ───────────────────
    c_files    = glob.glob(os.path.join(DEV_ROOT, "**", "*.c"),    recursive=True)
    java_files = glob.glob(os.path.join(DEV_ROOT, "**", "*.java"), recursive=True)
    all_files  = [(f, "C") for f in c_files] + [(f, "Java") for f in java_files]

    print(f"🔍 Found {len(c_files)} C files + {len(java_files)} Java files = {len(all_files)} total")
    print(f"📂 Source: {DEV_ROOT}\n")

    fieldnames = FEATURE_NAMES + ["label", "label_name", "language",
                                  "folder_hint", "auto_label", "filepath"]
    file_exists = os.path.exists(OUTPUT_PATH)

    collected = 0
    skipped   = 0
    label_counts = {"Clean": 0, "Moderate Risk": 0, "High Risk": 0}

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for filepath, lang in all_files:
            filename = os.path.basename(filepath)
            relpath  = os.path.relpath(filepath, DEV_ROOT)

            # Read source
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as src:
                    source = src.read().strip()
                if len(source) < 20:   # skip empty/trivial files
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            # Extract features
            features = _extract(source, lang)
            if features is None:
                print(f"   ⚠️  Parse failed: {relpath}")
                skipped += 1
                continue

            # Label
            hint      = _get_folder_hint(filepath)
            auto      = _auto_label(features)
            final_lbl = _blend_label(hint, auto)
            label_int = LABEL_TO_INT[final_lbl]

            # Build row
            row = {k: features.get(k, 0) for k in FEATURE_NAMES}
            row.update({
                "label":       label_int,
                "label_name":  final_lbl,
                "language":    lang,
                "folder_hint": hint,
                "auto_label":  auto,
                "filepath":    relpath,
            })
            writer.writerow(row)
            f.flush()

            label_counts[final_lbl] = label_counts.get(final_lbl, 0) + 1
            collected += 1
            match = "✅" if hint == auto else f"⚡ hint={hint}, auto={auto}"
            print(f"   [{lang:4}] {filename:<40} → {final_lbl}  {match}")

    print(f"\n{'='*60}")
    print(f"✅ Collected {collected} real samples → {OUTPUT_PATH}")
    print(f"   Skipped : {skipped}")
    print(f"   Labels  : {label_counts}")
    print(f"\n👉 Now run: python ml\\train_with_real_data.py")


if __name__ == "__main__":
    collect()
