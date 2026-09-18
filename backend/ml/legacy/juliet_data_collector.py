"""
IntelliReview — Juliet Test Suite Data Collector
==================================================
Processes the NIST Juliet Test Suite for C/C++ (v1.3).
This is the GOLD STANDARD dataset for vulnerability detection:
  - 118 CWE categories
  - 50,000+ real C/C++ source files
  - Expert-labeled: _bad = vulnerable, _good* = patched/safe

LABELING STRATEGY:
  1. Filename-based (ground truth, overrides all):
       *_bad*    → High Risk  (confirmed vulnerability present)
       *_good*   → Clean      (patched, safe version)
       other     → determined by CWE severity
  2. CWE category-based severity:
       Critical CWEs (BOF, injection, overflow) → High Risk
       Medium CWEs (leaks, races, unchecked)    → Moderate Risk
       Low CWEs (code quality, dead code)       → Clean

USAGE:
  cd C:\\Users\\Madhu\\OneDrive\\Desktop\\Projects\\IntelliReview\\backend
  python ml\\juliet_data_collector.py [--juliet-root PATH] [--max-per-cwe N]

OUTPUT:
  ml/juliet_data.csv  (separate from real_data.csv)
  Then run: python ml\\train_with_real_data.py
"""
import os
import sys
import csv
import glob
import hashlib
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.synthetic_data import FEATURE_NAMES

# ── Paths ──────────────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "juliet_data.csv")

# Default Juliet root (change if needed)
DEFAULT_JULIET_ROOT = r"C:\Users\Madhu\Downloads\2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3\C\testcases"

LABEL_TO_INT = {"Clean": 0, "Moderate Risk": 1, "High Risk": 2}

# ── CWE → Base Severity Mapping ────────────────────────────────────────────
# Files with _bad/_good suffixes OVERRIDE this. This is used for files
# without a clear suffix (helper/support files, header-only files, etc.)
#
# HIGH_RISK CWEs: memory corruption, injection, integer issues, use-after-free
HIGH_RISK_CWES = {
    "CWE114",   # Process Control
    "CWE121",   # Stack-Based Buffer Overflow
    "CWE122",   # Heap-Based Buffer Overflow
    "CWE123",   # Write-What-Where Condition
    "CWE124",   # Buffer Underwrite
    "CWE126",   # Buffer Overread
    "CWE127",   # Buffer Underread
    "CWE134",   # Uncontrolled Format String
    "CWE190",   # Integer Overflow
    "CWE191",   # Integer Underflow
    "CWE194",   # Unexpected Sign Extension
    "CWE195",   # Signed to Unsigned Conversion Error
    "CWE196",   # Unsigned to Signed Conversion Error
    "CWE197",   # Numeric Truncation Error
    "CWE242",   # Use of Inherently Dangerous Function
    "CWE256",   # Plaintext Storage of Password
    "CWE259",   # Hard-Coded Password
    "CWE319",   # Cleartext Tx of Sensitive Info
    "CWE321",   # Hard-Coded Cryptographic Key
    "CWE327",   # Use Broken Crypto
    "CWE364",   # Signal Handler Race Condition
    "CWE401",   # Memory Leak
    "CWE415",   # Double Free
    "CWE416",   # Use After Free
    "CWE426",   # Untrusted Search Path
    "CWE476",   # NULL Pointer Dereference
    "CWE590",   # Free Memory Not on Heap
    "CWE606",   # Unchecked Loop Condition
    "CWE674",   # Uncontrolled Recursion
    "CWE680",   # Integer Overflow to Buffer Overflow
    "CWE681",   # Incorrect Conversion Between Numeric Types
    "CWE689",   # Permission Race Condition
    "CWE761",   # Free Pointer Not at Start of Buffer
    "CWE762",   # Mismatched Memory Management Routines
    "CWE789",   # Uncontrolled Mem Alloc
    "CWE78",    # OS Command Injection
    "CWE843",   # Type Confusion
    "CWE90",    # LDAP Injection
}

# MODERATE_RISK CWEs: resource handling, logic errors, access control
MODERATE_RISK_CWES = {
    "CWE15",    # External Control of System/Config Setting
    "CWE23",    # Relative Path Traversal
    "CWE36",    # Absolute Path Traversal
    "CWE244",   # Heap Inspection
    "CWE247",   # Reliance on DNS Lookups in Security Decision
    "CWE252",   # Unchecked Return Value
    "CWE253",   # Incorrect Check of Function Return Value
    "CWE272",   # Least Privilege Violation
    "CWE273",   # Improper Check for Dropped Privileges
    "CWE284",   # Improper Access Control
    "CWE325",   # Missing Required Cryptographic Step
    "CWE328",   # Reversible One-Way Hash
    "CWE338",   # Weak PRNG
    "CWE366",   # Race Condition Within Thread
    "CWE367",   # TOC/TOU
    "CWE369",   # Divide by Zero
    "CWE377",   # Insecure Temporary File
    "CWE390",   # Error Without Action
    "CWE391",   # Unchecked Error Condition
    "CWE396",   # Catch Generic Exception
    "CWE397",   # Throw Generic Exception
    "CWE400",   # Resource Exhaustion
    "CWE404",   # Improper Resource Shutdown
    "CWE427",   # Uncontrolled Search Path Element
    "CWE440",   # Expected Behavior Violation
    "CWE457",   # Use of Uninitialized Variable
    "CWE459",   # Incomplete Cleanup
    "CWE464",   # Addition of Data Structure Sentinel
    "CWE467",   # Use of sizeof on Pointer Type
    "CWE468",   # Incorrect Pointer Scaling
    "CWE469",   # Use of Pointer Subtraction to Determine Size
    "CWE475",   # Undefined Behavior for Input to API
    "CWE478",   # Missing Default Case in Switch
    "CWE479",   # Signal Handler Use of Non-Reentrant Function
    "CWE484",   # Omitted Break Statement in Switch
    "CWE506",   # Embedded Malicious Code
    "CWE510",   # Trapdoor
    "CWE511",   # Logic Time Bomb
    "CWE526",   # Info Exposure - Environment Variables
    "CWE534",   # Info Exposure - Debug Log
    "CWE535",   # Info Exposure - Shell Error
    "CWE591",   # Sensitive Data in Improperly Locked Memory
    "CWE605",   # Multiple Binds Same Port
    "CWE615",   # Info Exposure by Comment
    "CWE617",   # Reachable Assertion
    "CWE620",   # Unverified Password Change
    "CWE665",   # Improper Initialization
    "CWE666",   # Operation on Resource in Wrong Phase
    "CWE667",   # Improper Locking
    "CWE672",   # Operation on Resource After Expiration
    "CWE675",   # Duplicate Operations on Resource
    "CWE676",   # Use of Potentially Dangerous Function
    "CWE685",   # Function Call With Incorrect Number of Arguments
    "CWE688",   # Function Call With Incorrect Variable
    "CWE690",   # NULL Deref From Return
    "CWE758",   # Undefined Behavior
    "CWE773",   # Missing Reference to Active File Descriptor
    "CWE775",   # Missing Release of File Descriptor
    "CWE780",   # Use of RSA Without OAEP
    "CWE785",   # Path Manipulation Without Max Buffer
    "CWE832",   # Unlock of Resource Not Locked
    "CWE835",   # Infinite Loop
}

# CLEAN CWEs: code quality, style, informational — not real vulnerabilities
CLEAN_CWES = {
    "CWE176",   # Improper Handling of Unicode
    "CWE188",   # Reliance on Data/Memory Layout
    "CWE222",   # Truncation of Security-Relevant Info
    "CWE223",   # Omission of Security-Relevant Info
    "CWE226",   # Sensitive Info Uncleared Before Release
    "CWE398",   # Poor Code Quality
    "CWE480",   # Use of Incorrect Operator
    "CWE481",   # Assigning Instead of Comparing
    "CWE482",   # Comparing Instead of Assigning
    "CWE483",   # Incorrect Block Delimitation
    "CWE500",   # Public Static Field Not Final
    "CWE546",   # Suspicious Comment
    "CWE561",   # Dead Code
    "CWE562",   # Return of Stack Variable Address
    "CWE563",   # Unused Variable
    "CWE570",   # Expression Always False
    "CWE571",   # Expression Always True
    "CWE587",   # Assignment of Fixed Address to Pointer
    "CWE588",   # Attempt to Access Child of Non-Structure Pointer
}


def _cwe_to_base_label(cwe_folder: str) -> str:
    """Determine base severity from CWE folder name."""
    # Extract CWE ID from folder name (e.g. "CWE121_Stack_Based_Buffer_Overflow" → "CWE121")
    folder_upper = cwe_folder.upper()
    for cwe_id in HIGH_RISK_CWES:
        if folder_upper.startswith(cwe_id.upper()):
            return "High Risk"
    for cwe_id in MODERATE_RISK_CWES:
        if folder_upper.startswith(cwe_id.upper()):
            return "Moderate Risk"
    for cwe_id in CLEAN_CWES:
        if folder_upper.startswith(cwe_id.upper()):
            return "Clean"
    # Default: treat unknown CWEs as Moderate Risk
    return "Moderate Risk"


def _filename_label(filename: str, cwe_base: str) -> str:
    """
    Apply filename-based labeling (ground truth).
    _bad  → High Risk (vulnerable code)
    _good → Clean     (safe/patched code)
    other → use CWE-based severity
    """
    name_lower = filename.lower()
    # Juliet naming: _bad, _goodB2G, _goodG2B, _goodB (all start with _good)
    if "_bad" in name_lower:
        return "High Risk"
    if "_good" in name_lower:
        return "Clean"
    # No clear suffix → use CWE-based severity
    return cwe_base


def _extract(source: str, language: str) -> Optional[dict]:
    """Run the IntelliReview analysis pipeline and return feature dict."""
    try:
        from parsers.c_parser import parse_c_code
        from analyzers.memory_leak import detect_memory_leaks
        from analyzers.unsafe_functions import detect_unsafe_functions
        from analyzers.complexity import analyze_complexity
        from analyzers.recursion import detect_recursion
        from analyzers.feature_extractor import extract_features

        # Treat both .c and .cpp as C for the Juliet suite
        pr = parse_c_code(source)
        mr = detect_memory_leaks(pr)
        ur = detect_unsafe_functions(pr)
        cx = analyze_complexity(pr, source)
        rc = detect_recursion(pr, source)
        fd = extract_features(pr, mr, ur, cx, rc, source=source)
        return fd["features"]
    except Exception:
        return None


def collect(juliet_root: str = DEFAULT_JULIET_ROOT,
            max_per_cwe: int = 0,
            clear_existing: bool = False):
    """
    Walk all CWE folders in the Juliet test suite and collect labeled samples.

    Args:
        juliet_root   : Root directory of Juliet testcases/
        max_per_cwe   : Max files to sample per CWE (0 = unlimited)
        clear_existing: If True, wipe juliet_data.csv before collecting
    """
    if not os.path.isdir(juliet_root):
        print(f"❌ Juliet root not found: {juliet_root}")
        print("   Set --juliet-root to the correct path.")
        return

    if clear_existing and os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print(f"🗑️  Cleared existing {OUTPUT_PATH}")

    # Discover CWE directories
    cwe_dirs = sorted([
        d for d in os.listdir(juliet_root)
        if os.path.isdir(os.path.join(juliet_root, d)) and d.upper().startswith("CWE")
    ])
    print(f"📂 Found {len(cwe_dirs)} CWE directories in: {juliet_root}")
    print(f"📝 Output → {OUTPUT_PATH}\n")

    fieldnames = FEATURE_NAMES + [
        "label", "label_name", "language", "cwe", "cwe_base",
        "filename_label", "filepath"
    ]
    file_exists = os.path.exists(OUTPUT_PATH)

    # Track content hashes for deduplication
    seen_hashes: set = set()
    # Pre-populate from existing file to avoid re-adding
    if file_exists:
        try:
            import csv as _csv
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f_in:
                for row in _csv.DictReader(f_in):
                    fp = row.get("filepath", "")
                    if fp:
                        seen_hashes.add(fp)
            print(f"   ♻️  Resuming — {len(seen_hashes)} files already in CSV\n")
        except Exception:
            pass

    total_collected = 0
    total_skipped   = 0
    total_dup       = 0
    label_counts    = {"Clean": 0, "Moderate Risk": 0, "High Risk": 0}

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for cwe_folder in cwe_dirs:
            cwe_path  = os.path.join(juliet_root, cwe_folder)
            cwe_base  = _cwe_to_base_label(cwe_folder)

            # Collect all .c and .cpp files in this CWE folder (recursive)
            c_files   = glob.glob(os.path.join(cwe_path, "**", "*.c"),   recursive=True)
            cpp_files = glob.glob(os.path.join(cwe_path, "**", "*.cpp"), recursive=True)
            all_files = c_files + cpp_files

            # Optional per-CWE cap (use every other file to preserve diversity)
            if max_per_cwe and len(all_files) > max_per_cwe:
                # Take evenly spaced subset to get diversity across patterns
                step = len(all_files) // max_per_cwe
                all_files = all_files[::max(step, 1)][:max_per_cwe]

            cwe_collected = 0
            cwe_skipped   = 0

            print(f"\n{'─'*60}")
            print(f"📁 {cwe_folder} ({len(all_files)} files | base={cwe_base})")

            for filepath in all_files:
                filename = os.path.basename(filepath)
                relpath  = os.path.relpath(filepath, juliet_root)

                # Skip header files
                if filename.endswith(".h"):
                    continue

                # Skip already-processed files
                if relpath in seen_hashes:
                    total_dup += 1
                    continue

                # Read source
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as src:
                        source = src.read().strip()
                    if len(source) < 50:
                        cwe_skipped += 1
                        continue
                except Exception:
                    cwe_skipped += 1
                    continue

                # Deduplication by content hash
                content_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    total_dup += 1
                    continue
                seen_hashes.add(content_hash)
                seen_hashes.add(relpath)

                # Determine label
                filename_lbl = _filename_label(filename, cwe_base)
                label_int    = LABEL_TO_INT[filename_lbl]
                language     = "CPP" if filename.endswith(".cpp") else "C"

                # Extract features
                features = _extract(source, language)
                if features is None:
                    print(f"   ⚠️ Parse failed: {filename}")
                    cwe_skipped += 1
                    continue

                # Write row
                row = {k: features.get(k, 0) for k in FEATURE_NAMES}
                row.update({
                    "label":          label_int,
                    "label_name":     filename_lbl,
                    "language":       language,
                    "cwe":            cwe_folder,
                    "cwe_base":       cwe_base,
                    "filename_label": filename_lbl,
                    "filepath":       relpath,
                })
                writer.writerow(row)
                f_out.flush()

                label_counts[filename_lbl] = label_counts.get(filename_lbl, 0) + 1
                cwe_collected += 1
                total_collected += 1

                badge = "🔴" if filename_lbl == "High Risk" else ("🟡" if filename_lbl == "Moderate Risk" else "🟢")
                print(f"   {badge} [{language}] {filename:<55} → {filename_lbl}")

            total_skipped += cwe_skipped
            print(f"   ✅ CWE done: {cwe_collected} collected, {cwe_skipped} skipped")

    print(f"\n{'='*65}")
    print(f"🏆 JULIET COLLECTION COMPLETE")
    print(f"   Total collected : {total_collected:,}")
    print(f"   Skipped (errors): {total_skipped:,}")
    print(f"   Duplicates skip : {total_dup:,}")
    print(f"\n   Label distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        pct = cnt / max(total_collected, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"     {lbl:<15} {cnt:>5}  ({pct:.1f}%)  {bar}")
    print(f"\n   📄 Saved → {OUTPUT_PATH}")
    print(f"\n👉 Now run: python ml\\train_with_real_data.py")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect labeled training data from the Juliet C/C++ Test Suite"
    )
    parser.add_argument(
        "--juliet-root",
        default=DEFAULT_JULIET_ROOT,
        help=f"Path to Juliet testcases/ directory (default: {DEFAULT_JULIET_ROOT})",
    )
    parser.add_argument(
        "--max-per-cwe",
        type=int,
        default=0,
        help="Max files to sample per CWE directory (0 = unlimited, recommended: 150)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing juliet_data.csv before collecting (default: append/resume)",
    )
    args = parser.parse_args()

    collect(
        juliet_root=args.juliet_root,
        max_per_cwe=args.max_per_cwe,
        clear_existing=args.clear,
    )
