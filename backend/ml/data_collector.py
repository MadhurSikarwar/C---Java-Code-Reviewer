"""
IntelliReview — Real Code Data Collector
=========================================
Downloads REAL C and Java source files from GitHub (no API key needed)
and auto-labels them using our own static analyzers.

HOW IT WORKS:
  1. Downloads .c / .java files from curated GitHub repo URLs
  2. Runs our full feature extraction pipeline on each file
  3. Auto-labels using a rule-based scoring function:
       score 0–2  → Clean
       score 3–5  → Moderate Risk
       score 6+   → High Risk
  4. Appends to ml/real_data.csv
  5. Combined with synthetic data for training

RUN:
  cd backend
  python ml/data_collector.py

REQUIREMENTS: requests (already installed via fastapi)
"""
import os
import sys
import csv
import json
import time
import hashlib
import urllib.request
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.synthetic_data import FEATURE_NAMES, LABEL_MAP

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "real_data.csv")

# ── Curated GitHub raw file URLs ───────────────────────────────────────
# Selected to give a mix of Clean, Moderate, and High-Risk code.
# These are stable public repos — no API key needed.

C_URLS = [
    # ── CLEAN examples ─────────────────────────────────────
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/searching/binary_search.c",  "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/sorting/bubble_sort.c",      "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/sorting/insertion_sort.c",   "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/math/fibonacci.c",            "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/data_structures/stack/stack_using_array.c", "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/searching/linear_search.c",  "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/sorting/selection_sort.c",   "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/math/prime_numbers.c",        "Clean"),
    # ── MODERATE examples ──────────────────────────────────
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/data_structures/linked_list/singly_linked_list.c", "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/data_structures/binary_tree/binary_search_tree.c", "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/sorting/merge_sort.c",        "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/sorting/quick_sort.c",        "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/data_structures/graphs/breadth_first_search.c", "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/data_structures/queue/queue_using_linked_list.c", "Moderate Risk"),
    # ── HIGH RISK examples (raw OS/system code snippets) ──
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/data_structures/graphs/depth_first_search.c", "High Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/data_structures/binary_tree/avl_tree.c",      "High Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/dynamic_programming/longest_common_subsequence.c", "High Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/C/master/sorting/radix_sort.c",        "High Risk"),
]

JAVA_URLS = [
    # ── CLEAN ──────────────────────────────────────────────
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/searches/BinarySearch.java",     "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/sorts/BubbleSort.java",          "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/sorts/InsertionSort.java",       "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/maths/Fibonacci.java",           "Clean"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/searches/LinearSearch.java",     "Clean"),
    # ── MODERATE ───────────────────────────────────────────
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/sorts/MergeSort.java",           "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/sorts/QuickSort.java",           "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/datastructures/trees/BinarySearchTree.java", "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/datastructures/graphs/BreadthFirstSearch.java", "Moderate Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/sorts/HeapSort.java",            "Moderate Risk"),
    # ── HIGH RISK ──────────────────────────────────────────
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/datastructures/graphs/Dijkstra.java", "High Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/datastructures/trees/AVLTree.java",  "High Risk"),
    ("https://raw.githubusercontent.com/TheAlgorithms/Java/master/src/main/java/com/thealgorithms/dynamicprogramming/LongestCommonSubsequence.java", "High Risk"),
]

LABEL_TO_INT = {"Clean": 0, "Moderate Risk": 1, "High Risk": 2}


# ── Feature extraction pipeline ────────────────────────────────────────
def _extract_features_from_source(source: str, language: str) -> Optional[Dict]:
    """Run the full IntelliReview analysis pipeline on source code."""
    try:
        from parsers.c_parser    import parse_c_code
        from parsers.java_parser import parse_java_code
        from analyzers.memory_leak       import detect_memory_leaks
        from analyzers.unsafe_functions  import detect_unsafe_functions
        from analyzers.complexity        import analyze_complexity
        from analyzers.recursion         import detect_recursion
        from analyzers.feature_extractor import extract_features

        lang = language.upper()
        parse_result     = parse_c_code(source) if lang == "C" else parse_java_code(source)
        memory_result    = detect_memory_leaks(parse_result)
        unsafe_result    = detect_unsafe_functions(parse_result)
        complexity_result= analyze_complexity(parse_result, source)
        recursion_result = detect_recursion(parse_result, source)
        feat_data        = extract_features(
            parse_result, memory_result, unsafe_result,
            complexity_result, recursion_result, source=source,
        )
        return feat_data["features"]
    except Exception as e:
        print(f"     ⚠️  Feature extraction failed: {e}")
        return None


# ── Auto-labeler (rule-based scoring for trust validation) ─────────────
def _auto_label(features: Dict) -> str:
    """
    Assign a risk label from features using a weighted scoring rule.
    This is used to VERIFY the GitHub label, not override it.
    Returns one of: Clean / Moderate Risk / High Risk
    """
    score = 0
    score += min(features.get("memory_leak_count", 0) * 3, 9)
    score += min(features.get("unsafe_function_count", 0) * 2, 6)
    score += min(max(0, features.get("cyclomatic_complexity", 1) - 10), 5)
    score += min(features.get("max_nesting_depth", 0) * 1, 4)
    score += min(features.get("recursion_count", 0) * 1, 3)
    score += (1 if features.get("comment_ratio", 0) < 0.05 else 0)
    score += min(features.get("dead_code_estimate", 0), 3)

    if score <= 2:   return "Clean"
    elif score <= 6: return "Moderate Risk"
    else:            return "High Risk"


# ── Download function ──────────────────────────────────────────────────
def _download(url: str, timeout: int = 10) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IntelliReview/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"     ❌ Download failed: {e}")
        return None


# ── Main collector ─────────────────────────────────────────────────────
def collect(limit: int = None):
    """
    Download all URLs, extract features, and save to real_data.csv.
    Set limit=N to only collect N samples (useful for testing).
    """
    all_samples = (
        [(url, label, "C")    for url, label in C_URLS] +
        [(url, label, "Java") for url, label in JAVA_URLS]
    )
    if limit:
        all_samples = all_samples[:limit]

    # Prepare CSV
    fieldnames = FEATURE_NAMES + ["label", "label_name", "source_url", "language", "auto_label"]
    file_exists = os.path.exists(OUTPUT_PATH)

    collected = 0
    skipped   = 0

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for url, label, lang in all_samples:
            short = url.split("/")[-1]
            print(f"\n📥 [{lang}] {short} → labeled '{label}'")

            source = _download(url)
            if not source:
                skipped += 1
                continue

            features = _extract_features_from_source(source, lang)
            if not features:
                skipped += 1
                continue

            auto_label = _auto_label(features)
            label_int  = LABEL_TO_INT.get(label, 0)

            row = {k: features.get(k, 0) for k in FEATURE_NAMES}
            row.update({
                "label":      label_int,
                "label_name": label,
                "source_url": url,
                "language":   lang,
                "auto_label": auto_label,
            })
            writer.writerow(row)
            f.flush()
            collected += 1

            match = "✅" if auto_label == label else "⚠️ mismatch"
            print(f"   Features extracted | auto_label={auto_label} {match}")
            time.sleep(0.3)   # be polite to GitHub

    print(f"\n{'='*55}")
    print(f"✅ Collected {collected} real samples → {OUTPUT_PATH}")
    print(f"   Skipped {skipped} (download/parse failures)")
    print(f"\n👉 Now run: python ml/train_with_real_data.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of files to download (default: all)")
    args = parser.parse_args()
    collect(limit=args.limit)
