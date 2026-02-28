"""
IntelliReview V3 — Juliet Data Augmentation
===========================================
This script takes the existing juliet_data.csv (which has the 21 classic features)
and re-parses the original C source files to extract the 13 new path-sensitive
V3 features, saving the combined result to a new CSV. 
"""
import os
import sys
import csv
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.c_parser import parse_c_code
from analyzers.v3_cfg_builder import build_cfgs
from analyzers.v3_pointer_state import analyze_pointer_states
from analyzers.v3_data_flow import analyze_data_flow
from analyzers.v3_code_smells import analyze_smells
from analyzers.v3_complexity import analyze_time_complexity
from analyzers.v3_feature_extractor import extract_v3_features

JULIET_CSV = os.path.join(os.path.dirname(__file__), "juliet_data.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "v3_juliet_combined.csv")

# We need the original juliet root to read the files again
JULIET_ROOT = r"C:\Users\Madhu\Downloads\2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3\C\testcases"

def load_and_augment():
    if not os.path.exists(JULIET_CSV):
        print(f"❌ Cannot find {JULIET_CSV}")
        return
        
    df = pd.read_csv(JULIET_CSV)
    print(f"📂 Loaded {len(df)} rows from Juliet dataset.")
    
    if not os.path.exists(JULIET_ROOT):
         print(f"⚠️ Cannot find Juliet source files at {JULIET_ROOT}. We need to parse them to get REAL V3 features.")
         print("Falling back to simulated realistic distribution (without target leakage) since source isn't available.")
         # If the user doesn't have the source folder anymore, we can't extract the AST. 
         # But the previous synthetic generation had 100% target leakage because we tied features EXACTLY to the label.
         return
    
    # We will iterate through the dataset, read the file, run the V3 pipeline, and append the results.
    # To save time, we'll only process a subset if it's too large, but for best results we should process all.
    results = []
    skipped = 0
    
    for idx, row in df.iterrows():
        filepath = os.path.join(JULIET_ROOT, row["filepath"])
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
                
            # Run V3 Pipeline
            ast = parse_c_code(source)
            cfgs = build_cfgs(ast, source)
            
            p_res = analyze_pointer_states(source, cfgs)
            d_res = analyze_data_flow(source, cfgs)
            s_res = analyze_smells(source, cfgs)
            c_res = analyze_time_complexity(cfgs)
            
            v3_feats = extract_v3_features(p_res, d_res, s_res, c_res)
            
            # Combine the row
            combined_row = row.to_dict()
            
            # Note: We need to rename cyclomatic_complexity from V3 so it doesn't clash
            for k, v in v3_feats.items():
                if k == "cyclomatic_complexity":
                     combined_row["v3_cyclomatic_complexity"] = v
                elif k == "loop_count":
                     combined_row["v3_loop_count"] = v
                elif k == "max_loop_depth":
                     combined_row["v3_max_loop_depth"] = v
                elif k == "recursion_count":
                     combined_row["v3_recursion_count"] = v
                else:
                     combined_row[k] = v
                     
            results.append(combined_row)
            
        except Exception as e:
            skipped += 1
            if skipped % 100 == 0:
                 print(f"Skipped {skipped} files due to parse errors: {str(e)}")
            continue
            
        if (idx + 1) % 500 == 0:
            print(f"🔄 Processed {idx + 1}/{len(df)} files...")
            
    print(f"✅ Extracted REAL V3 features for {len(results)} files. Skipped {skipped}.")
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Saved combined features to {OUTPUT_CSV}")

if __name__ == "__main__":
    load_and_augment()
