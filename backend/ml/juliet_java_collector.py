"""
IntelliReview — Juliet Java Test Suite Data Collector
======================================================
Processes the NIST Juliet Test Suite for Java (v1.3).
113 CWE categories, thousands of real .java source files,
all expert-labeled via filename convention:

  *_bad*   → High Risk  (confirmed vulnerability)
  *_good*  → Clean      (patched/safe version)
  other    → determined by CWE severity mapping

USAGE (run from backend/ directory):
  cd C:\\Users\\Madhu\\OneDrive\\Desktop\\Projects\\IntelliReview\\backend

  # Collect up to 100 files per CWE (~11,000 samples):
  python ml\\juliet_java_collector.py --max-per-cwe 100 --clear

  # Collect ALL files (very thorough, takes 30+ min):
  python ml\\juliet_java_collector.py --clear

  # Resume interrupted collection:
  python ml\\juliet_java_collector.py --max-per-cwe 100

OUTPUT:
  ml/juliet_java_data.csv  (separate file, auto-loaded by trainer)
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

# ── Paths ───────────────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "juliet_java_data.csv")

DEFAULT_JULIET_JAVA_ROOT = (
    r"C:\Users\Madhu\Downloads"
    r"\2017-10-01-juliet-test-suite-for-java-v1-3\Java\src\testcases"
)

LABEL_TO_INT = {"Clean": 0, "Moderate Risk": 1, "High Risk": 2}

# ── Java CWE Severity Maps ──────────────────────────────────────────────────

HIGH_RISK_CWES = {
    "CWE78",    # OS Command Injection
    "CWE80",    # XSS
    "CWE81",    # XSS Error Message
    "CWE83",    # XSS Attribute
    "CWE89",    # SQL Injection
    "CWE90",    # LDAP Injection
    "CWE111",   # Unsafe JNI
    "CWE113",   # HTTP Response Splitting
    "CWE114",   # Process Control
    "CWE129",   # Improper Validation of Array Index
    "CWE134",   # Uncontrolled Format String
    "CWE190",   # Integer Overflow
    "CWE191",   # Integer Underflow
    "CWE193",   # Off-by-One Error
    "CWE256",   # Plaintext Storage of Password
    "CWE259",   # Hard-Coded Password
    "CWE315",   # Plaintext Storage in Cookie
    "CWE319",   # Cleartext Tx of Sensitive Info
    "CWE321",   # Hard-Coded Cryptographic Key
    "CWE327",   # Use Broken Crypto
    "CWE329",   # Not Using Random IV with CBC
    "CWE336",   # Same Seed in PRNG
    "CWE338",   # Weak PRNG
    "CWE366",   # Race Condition Within Thread
    "CWE470",   # Unsafe Reflection
    "CWE476",   # NULL Pointer Dereference
    "CWE491",   # Object Hijack
    "CWE499",   # Sensitive Data Serializable
    "CWE506",   # Embedded Malicious Code
    "CWE510",   # Trapdoor
    "CWE511",   # Logic Time Bomb
    "CWE523",   # Unprotected Credential Transport
    "CWE539",   # Info Exposure Through Persistent Cookie
    "CWE566",   # Authorization Bypass Through SQL
    "CWE601",   # Open Redirect
    "CWE609",   # Double-Checked Locking
    "CWE643",   # XPath Injection
    "CWE674",   # Uncontrolled Recursion
    "CWE681",   # Incorrect Conversion Between Numeric Types
    "CWE689",   # Permission Race Condition
    "CWE759",   # Unsalted One-Way Hash
    "CWE760",   # Predictable Salt One-Way Hash
    "CWE789",   # Uncontrolled Mem Alloc
    "CWE833",   # Deadlock
    "CWE835",   # Infinite Loop
}

MODERATE_RISK_CWES = {
    "CWE15",    # External Control of System/Config
    "CWE23",    # Relative Path Traversal
    "CWE36",    # Absolute Path Traversal
    "CWE197",   # Numeric Truncation Error
    "CWE209",   # Information Leak via Error
    "CWE248",   # Uncaught Exception
    "CWE252",   # Unchecked Return Value
    "CWE253",   # Incorrect Check of Function Return Value
    "CWE325",   # Missing Required Cryptographic Step
    "CWE328",   # Reversible One-Way Hash
    "CWE369",   # Divide by Zero
    "CWE378",   # Temp File with Insecure Perms
    "CWE379",   # Temp File in Insecure Dir
    "CWE382",   # Use of System.exit()
    "CWE383",   # Direct Use of Threads
    "CWE390",   # Error Without Action
    "CWE395",   # Catch NullPointerException
    "CWE396",   # Catch Generic Exception
    "CWE397",   # Throw Generic
    "CWE400",   # Resource Exhaustion
    "CWE404",   # Improper Resource Shutdown
    "CWE459",   # Incomplete Cleanup
    "CWE477",   # Obsolete Functions
    "CWE478",   # Missing Default Case in Switch
    "CWE484",   # Omitted Break Statement in Switch
    "CWE486",   # Compare Classes by Name
    "CWE526",   # Info Exposure - Env Variables
    "CWE533",   # Info Exposure Server Log
    "CWE534",   # Info Exposure Debug Log
    "CWE535",   # Info Exposure Shell Error
    "CWE549",   # Missing Password Masking
    "CWE568",   # Finalize Without Super
    "CWE572",   # Call to Thread.run() Instead of start()
    "CWE579",   # Non-Serializable in Session
    "CWE580",   # Clone Without Super
    "CWE581",   # Object Model Violation
    "CWE584",   # Return in Finally Block
    "CWE585",   # Empty Sync Block
    "CWE586",   # Explicit Call to Finalize
    "CWE598",   # Info Exposure QueryString
    "CWE600",   # Uncaught Exception in Servlet
    "CWE605",   # Multiple Binds to Same Port
    "CWE606",   # Unchecked Loop Condition
    "CWE607",   # Public Static Final Mutable
    "CWE613",   # Insufficient Session Expiration
    "CWE614",   # Sensitive Cookie Without Secure Flag
    "CWE615",   # Info Exposure by Comment
    "CWE617",   # Reachable Assertion
    "CWE667",   # Improper Locking
    "CWE690",   # NULL Deref From Return
    "CWE698",   # Redirect Without Exit
    "CWE764",   # Multiple Locks
    "CWE765",   # Multiple Unlocks
    "CWE772",   # Missing Release of Resource
    "CWE775",   # Missing Release of File Descriptor
    "CWE832",   # Unlock of Resource Not Locked
}

CLEAN_CWES = {
    "CWE226",   # Sensitive Info Uncleared Before Release
    "CWE398",   # Poor Code Quality
    "CWE481",   # Assigning Instead of Comparing
    "CWE482",   # Comparing Instead of Assigning
    "CWE483",   # Incorrect Block Delimitation
    "CWE500",   # Public Static Field Not Final
    "CWE546",   # Suspicious Comment
    "CWE561",   # Dead Code
    "CWE563",   # Unused Variable
    "CWE570",   # Expression Always False
    "CWE571",   # Expression Always True
    "CWE582",   # Array Public Final Static
    "CWE597",   # Wrong Operator String Comparison
}


def _cwe_to_base_label(cwe_folder: str) -> str:
    """Return base severity from Java CWE folder name."""
    f = cwe_folder.upper()
    for cwe in HIGH_RISK_CWES:
        if f.startswith(cwe.upper()):
            return "High Risk"
    for cwe in MODERATE_RISK_CWES:
        if f.startswith(cwe.upper()):
            return "Moderate Risk"
    for cwe in CLEAN_CWES:
        if f.startswith(cwe.upper()):
            return "Clean"
    return "Moderate Risk"


def _filename_label(filename: str, cwe_base: str) -> str:
    """
    Ground-truth label from Juliet filename convention:
      _bad  → High Risk   (vulnerable code)
      _good → Clean       (patched/safe)
      other → CWE-based severity
    """
    name = filename.lower()
    if "_bad" in name:
        return "High Risk"
    if "_good" in name:
        return "Clean"
    return cwe_base


def _extract(source: str) -> Optional[dict]:
    """Run IntelliReview Java pipeline and return feature dict."""
    try:
        from parsers.java_parser import parse_java_code
        from analyzers.memory_leak import detect_memory_leaks
        from analyzers.unsafe_functions import detect_unsafe_functions
        from analyzers.complexity import analyze_complexity
        from analyzers.recursion import detect_recursion
        from analyzers.feature_extractor import extract_features

        pr = parse_java_code(source)
        mr = detect_memory_leaks(pr)
        ur = detect_unsafe_functions(pr)
        cx = analyze_complexity(pr, source)
        rc = detect_recursion(pr, source)
        fd = extract_features(pr, mr, ur, cx, rc, source=source)
        return fd["features"]
    except Exception:
        return None


def collect(juliet_root: str = DEFAULT_JULIET_JAVA_ROOT,
            max_per_cwe: int = 0,
            clear_existing: bool = False):
    """
    Walk all Java CWE folders and collect labeled training samples.

    Args:
        juliet_root   : Path to Java testcases/ directory
        max_per_cwe   : Max files per CWE (0 = unlimited)
        clear_existing: Wipe output before collecting
    """
    if not os.path.isdir(juliet_root):
        print(f"❌ Juliet Java root not found: {juliet_root}")
        return

    if clear_existing and os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        print(f"🗑️  Cleared {OUTPUT_PATH}")

    cwe_dirs = sorted([
        d for d in os.listdir(juliet_root)
        if os.path.isdir(os.path.join(juliet_root, d))
        and d.upper().startswith("CWE")
    ])

    print(f"☕ Found {len(cwe_dirs)} Java CWE directories")
    print(f"📂 Root : {juliet_root}")
    print(f"📝 Output → {OUTPUT_PATH}\n")

    fieldnames = FEATURE_NAMES + [
        "label", "label_name", "language", "cwe",
        "cwe_base", "filename_label", "filepath"
    ]
    file_exists = os.path.exists(OUTPUT_PATH)

    # Pre-load seen paths for resume support
    seen_hashes: set = set()
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
            cwe_path = os.path.join(juliet_root, cwe_folder)
            cwe_base = _cwe_to_base_label(cwe_folder)

            java_files = glob.glob(
                os.path.join(cwe_path, "**", "*.java"), recursive=True
            )

            # Optional per-CWE cap with even spacing for diversity
            if max_per_cwe and len(java_files) > max_per_cwe:
                step = len(java_files) // max_per_cwe
                java_files = java_files[::max(step, 1)][:max_per_cwe]

            cwe_collected = 0
            cwe_skipped   = 0

            print(f"\n{'─'*65}")
            print(f"☕ {cwe_folder}  ({len(java_files)} files | base={cwe_base})")

            for filepath in java_files:
                filename = os.path.basename(filepath)
                relpath  = os.path.relpath(filepath, juliet_root)

                # Skip if already in CSV (resume mode)
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

                # Dedup by content hash
                content_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    total_dup += 1
                    continue
                seen_hashes.add(content_hash)
                seen_hashes.add(relpath)

                # Label
                lbl      = _filename_label(filename, cwe_base)
                lbl_int  = LABEL_TO_INT[lbl]

                # Extract features
                features = _extract(source)
                if features is None:
                    print(f"   ⚠️  Parse failed: {filename}")
                    cwe_skipped += 1
                    continue

                # Write row
                row = {k: features.get(k, 0) for k in FEATURE_NAMES}
                row.update({
                    "label":          lbl_int,
                    "label_name":     lbl,
                    "language":       "Java",
                    "cwe":            cwe_folder,
                    "cwe_base":       cwe_base,
                    "filename_label": lbl,
                    "filepath":       relpath,
                })
                writer.writerow(row)
                f_out.flush()

                label_counts[lbl] += 1
                cwe_collected     += 1
                total_collected   += 1

                badge = ("🔴" if lbl == "High Risk"
                         else "🟡" if lbl == "Moderate Risk"
                         else "🟢")
                print(f"   {badge} {filename:<55} → {lbl}")

            total_skipped += cwe_skipped
            print(f"   ✅ Done: {cwe_collected} collected, {cwe_skipped} skipped")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"🏆 JAVA JULIET COLLECTION COMPLETE")
    print(f"   Total collected  : {total_collected:,}")
    print(f"   Skipped (errors) : {total_skipped:,}")
    print(f"   Duplicates skipped: {total_dup:,}")
    print(f"\n   Label distribution:")
    for lbl, cnt in sorted(label_counts.items()):
        pct = cnt / max(total_collected, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"     {lbl:<15} {cnt:>5}  ({pct:.1f}%)  {bar}")
    print(f"\n   📄 Saved → {OUTPUT_PATH}")
    print(f"\n👉 Next: python ml\\train_with_real_data.py")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect labeled training data from Juliet Java Test Suite"
    )
    parser.add_argument(
        "--juliet-root",
        default=DEFAULT_JULIET_JAVA_ROOT,
        help="Path to Java testcases/ directory",
    )
    parser.add_argument(
        "--max-per-cwe",
        type=int, default=0,
        help="Max .java files per CWE (0 = unlimited, recommended: 100)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing juliet_java_data.csv before collecting",
    )
    args = parser.parse_args()
    collect(
        juliet_root=args.juliet_root,
        max_per_cwe=args.max_per_cwe,
        clear_existing=args.clear,
    )
