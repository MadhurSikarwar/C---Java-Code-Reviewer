"""
IntelliReview — retrain all four models on real, size-decoupled data and evaluate them honestly
==============================================================================================
Data:    ml/data/features.csv   (build with ml/juliet_extract.py + ml/build_features.py)
Models:  ensemble   LightGBM + XGBoost + ExtraTrees soft-voting on the 21 classic features   -> model.pkl
         dl         Keras MLP on the 21 classic features                                     -> model_dl_v1_19feat.keras
         v3         Random Forest on 34 features (21 classic + 13 path-sensitive)            -> model_v3_path_sensitive.pkl
         v4_dl      Keras MLP on all 36 features                                             -> model_v4_dedup_massive.keras

Evaluation (why the old "100 % accuracy" meant nothing): metrics come from GROUP-held-out folds — every Juliet CWE (and every
generated-program bucket) is entirely absent from the training folds of the model that is tested on it. Reported next to:
  * a SIZE-ONLY baseline   (LOC, #functions, ... only)   -> should be near chance if size no longer leaks the label
  * the RULE ENGINE alone  (verdict from the analyzers' findings, no ML)
  * each model AS DEPLOYED (model prediction + the evidence gate that predict.py applies)

Run from backend/:   python ml/retrain_all.py [--no-dl] [--folds 5]
"""
import argparse
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from analyzers.issue_taxonomy import NON_SECURITY_TYPES, QUALITY_TYPES
from ml.scalers import LogStandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "features.csv")
REPORT = os.path.join(HERE, "model_report.json")
LABELS = ["Clean", "Moderate", "High"]
CHOSEN_GATE = ["high"]      # provisional; replaced by the best variant after the comparison below
SIZE_ONLY = ["lines_of_code", "num_functions", "num_loops", "num_conditionals", "cyclomatic_complexity",
             "function_call_count", "halstead_volume"]


# --------------------------------------------------------------------------------------------------
def apply_gate(pred: int, issue_types: str, mode: str = "full") -> int:
    """Evidence gate (ml/predict.py applies the variant chosen in model_report.json).
       mode 'none'  : raw model output
       mode 'high'  : upgrade to High when a critical/high SECURITY finding exists; cap High->Moderate without one
       mode 'full'  : 'high' + never call code with a medium-severity finding Clean"""
    if mode == "none":
        return pred
    types = [t.rsplit(":", 1) for t in str(issue_types).split("|") if t and t != "nan"]
    high = any(s in ("CRITICAL", "HIGH") and t not in NON_SECURITY_TYPES for t, s in types)
    anyev = any(s in ("CRITICAL", "HIGH", "MEDIUM") and t not in QUALITY_TYPES for t, s in types)
    if high and pred < 2:
        pred = 2
    elif not high and pred == 2:
        pred = 1
    if mode == "full" and anyev and pred == 0:
        pred = 1
    if mode == "calm" and not anyev and pred == 1:
        pred = 0            # Moderate with no supporting finding at all: not something the reader can act on
    return pred


def make_lgbm(**kw):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                              class_weight="balanced", random_state=42, verbose=-1, n_jobs=4, **kw)


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         random_state=42, n_jobs=4, eval_metric="mlogloss", verbosity=0)


def make_ensemble():
    return VotingClassifier([
        ("lgbm", make_lgbm()),
        ("xgb", make_xgb()),
        ("et", ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=4)),
    ], voting="soft")


def make_rf():
    return RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, class_weight="balanced",
                                  random_state=42, n_jobs=4)


def build_mlp(dim):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input(shape=(dim,)),
        layers.Dense(128, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.25),
        layers.Dense(32, activation="relu"), layers.Dropout(0.15),
        layers.Dense(3, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def fit_mlp(X, y, groups, epochs=80):
    """Returns (model, scaler). Early stopping on a GROUP-held-out validation split."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=0).split(X, y, groups))
    scaler = LogStandardScaler().fit(X[tr])
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y[tr])
    model = build_mlp(X.shape[1])
    model.fit(scaler.transform(X[tr]), y[tr], validation_data=(scaler.transform(X[va]), y[va]), epochs=epochs,
              batch_size=256, verbose=0, class_weight=dict(enumerate(cw)),
              callbacks=[EarlyStopping(patience=8, restore_best_weights=True), ReduceLROnPlateau(patience=4, factor=0.5)])
    return model, scaler


def predict_mlp(model, scaler, X):
    return np.argmax(model.predict(scaler.transform(X), verbose=0), axis=1)


# --------------------------------------------------------------------------------------------------
def score_block(y, pred, gated, mask, name):
    """Metrics on a subset: accuracy, macro-F1, false-alarm rate on Clean, High recall."""
    if mask.sum() == 0:
        return None
    yy, pp, gg = y[mask], pred[mask], gated[mask]
    out = {"n": int(mask.sum()),
           "acc": round(accuracy_score(yy, pp), 4), "macro_f1": round(f1_score(yy, pp, average="macro"), 4),
           "acc_gated": round(accuracy_score(yy, gg), 4), "macro_f1_gated": round(f1_score(yy, gg, average="macro"), 4)}
    clean, high = yy == 0, yy == 2
    out["false_alarm_clean"] = round(float((gg[clean] > 0).mean()), 4) if clean.any() else None
    out["false_high_clean"] = round(float((gg[clean] == 2).mean()), 4) if clean.any() else None
    out["high_recall"] = round(float((gg[high] == 2).mean()), 4) if high.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dl", action="store_true", help="skip the two neural networks")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    feat_cols = [c for c in df.columns if c not in
                 ("src", "language", "label", "group", "cwe", "kind", "analysis_mode", "rule_label", "issue_types", "loc")]
    assert len(feat_cols) == 36, len(feat_cols)
    classic, v3cols = feat_cols[:21], feat_cols[:34]
    y = df["label"].values.astype(int)
    groups = df["group"].astype(str).values
    issue_types = df["issue_types"].fillna("").values
    print(f"{len(df)} samples | class balance {np.bincount(y).tolist()} | sources {df.src.value_counts().to_dict()}")
    print(f"groups: {len(set(groups))}\n")

    specs = {
        "size_only": (SIZE_ONLY, make_lgbm, "tree"),
        "ensemble": (classic, make_ensemble, "tree"),
        "v3": (v3cols, make_rf, "tree"),
    }
    if not args.no_dl:
        specs["dl"] = (classic, None, "dl")
        specs["v4_dl"] = (feat_cols, None, "dl")

    # ---------------- cross-validation, groups held out ----------------
    gkf = GroupKFold(n_splits=args.folds)
    oof = {name: np.zeros(len(df), dtype=int) for name in specs}
    t0 = time.time()
    for fold, (tr, te) in enumerate(gkf.split(df, y, groups)):
        for name, (cols, factory, kind) in specs.items():
            if kind == "dl" and fold > 0:
                continue                                   # one held-out fold is enough (and much faster) for the nets
            Xtr, Xte = df.iloc[tr][cols], df.iloc[te][cols]
            if kind == "tree":
                m = factory().fit(Xtr, y[tr])
                oof[name][te] = m.predict(Xte)
            else:
                model, sc = fit_mlp(df.iloc[tr][cols].values, y[tr], groups[tr])
                oof[name][te] = predict_mlp(model, sc, Xte.values)
        print(f"  fold {fold + 1}/{args.folds} done ({time.time() - t0:.0f}s)")

    # DL nets were only evaluated on fold 0: restrict their metrics to that fold's samples
    fold0_mask = np.zeros(len(df), dtype=bool)
    fold0_mask[next(iter(gkf.split(df, y, groups)))[1]] = True

    rule = df["rule_label"].values.astype(int)
    results = {}
    subsets = {"ALL": np.ones(len(df), bool)}
    for s in ("juliet", "composite", "generated"):
        subsets[s] = (df.src == s).values
    big_clean = (df.src != "juliet").values & (y == 0) & (df["loc"].values > 200)
    small_clean = (df.src != "juliet").values & (y == 0) & (df["loc"].values <= 60)

    GATES = ("none", "high", "full", "calm")
    gate_table = {}

    def evaluate(name, pred, base_mask):
        for g in GATES:
            gp = np.array([apply_gate(p, it, g) for p, it in zip(pred, issue_types)])
            gate_table[(name, g)] = {sn: score_block(y, pred, gp, subsets[sn] & base_mask, sn)
                                     for sn in ("juliet", "composite", "generated")}
        gated = np.array([apply_gate(p, it, CHOSEN_GATE[0]) for p, it in zip(pred, issue_types)])
        res = {}
        for sname, m in subsets.items():
            r = score_block(y, pred, gated, m & base_mask, sname)
            if r:
                res[sname] = r
        # size leakage probe: how often is CLEAN code flagged, small vs big programs?
        res["size_probe"] = {
            "false_alarm_small_clean(<=60 LOC)": round(float((gated[small_clean & base_mask] > 0).mean()), 4)
            if (small_clean & base_mask).any() else None,
            "false_alarm_big_clean(>200 LOC)": round(float((gated[big_clean & base_mask] > 0).mean()), 4)
            if (big_clean & base_mask).any() else None,
        }
        res["confusion_gated"] = confusion_matrix(y[base_mask], gated[base_mask], labels=[0, 1, 2]).tolist()
        results[name] = res

    evaluate("rule_engine_only", rule, np.ones(len(df), bool))
    for name, (_, _, kind) in specs.items():
        evaluate(name, oof[name], fold0_mask if kind == "dl" else np.ones(len(df), bool))

    # ---- choose the gate that gives the best macro-F1 on the REAL-labelled data (juliet + composite), averaged over
    #      the four deployed models
    print("\nGate comparison (macro-F1 on held-out juliet | composite | generated, mean over deployed models):")
    best, best_score = None, -1
    for g in GATES:
        row = []
        for sn in ("juliet", "composite", "generated"):
            vals = [gate_table[(m, g)][sn]["macro_f1_gated"] for m in ("ensemble", "v3", "dl", "v4_dl")
                    if (m, g) in gate_table and gate_table[(m, g)].get(sn)]
            row.append(float(np.mean(vals)) if vals else float("nan"))
        real = float(np.nanmean(row[:2])) if not np.isnan(row[:2]).all() else row[2]   # no Juliet data: use generated programs
        print(f"  gate={g:5}  juliet {row[0]:.3f}  composite {row[1]:.3f}  generated {row[2]:.3f}   real-data mean {real:.3f}")
        # never allow a variant that lets a model call code High Risk without evidence
        if g != "none" and real > best_score:
            best, best_score = g, real
    CHOSEN_GATE[0] = best
    print(f"  -> using gate '{best}'\n")
    results = {}
    evaluate("rule_engine_only", rule, np.ones(len(df), bool))
    for name, (_, _, kind) in specs.items():
        evaluate(name, oof[name], fold0_mask if kind == "dl" else np.ones(len(df), bool))

    print("\n================ GROUP-HELD-OUT RESULTS ================")
    hdr = f"{'model':18} {'subset':10} {'n':>6} {'acc':>6} {'F1':>6} | gated: {'acc':>6} {'F1':>6} {'FA(clean)':>9} {'FalseHigh':>9} {'HighRecall':>10}"
    print(hdr)
    for name, res in results.items():
        for sname in ("juliet", "composite", "generated"):
            r = res.get(sname)
            if r:
                print(f"{name:18} {sname:10} {r['n']:>6} {r['acc']:>6.3f} {r['macro_f1']:>6.3f} |        "
                      f"{r['acc_gated']:>6.3f} {r['macro_f1_gated']:>6.3f} {str(r['false_alarm_clean']):>9} "
                      f"{str(r['false_high_clean']):>9} {str(r['high_recall']):>10}")
        print(f"{'':18} size probe: {res['size_probe']}")
    print()

    # ---------------- final models on ALL data ----------------
    print("Training final models on all data ...")
    joblib.dump(make_ensemble().fit(df[classic], y), os.path.join(HERE, "model.pkl"))
    joblib.dump(make_rf().fit(df[v3cols], y), os.path.join(HERE, "model_v3_path_sensitive.pkl"))
    # label encoder file is read (but not required) by config.py consumers
    from sklearn.preprocessing import LabelEncoder
    joblib.dump(LabelEncoder().fit([0, 1, 2]), os.path.join(HERE, "label_encoder.pkl"))
    if not args.no_dl:
        for cols, mpath, spath in ((classic, "model_dl_v1_19feat.keras", "dl_scaler_v1_19feat.pkl"),
                                   (feat_cols, "model_v4_dedup_massive.keras", "v4_scaler.pkl")):
            model, sc = fit_mlp(df[cols].values, y, groups)
            model.save(os.path.join(HERE, mpath))
            joblib.dump(sc, os.path.join(HERE, spath))
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"n_samples": len(df), "class_balance": np.bincount(y).tolist(), "folds": args.folds,
                   "protocol": "GroupKFold by CWE / program bucket; neural nets evaluated on fold 0 only",
                   "gate": CHOSEN_GATE[0],
                   "results": results}, f, indent=2)
    print(f"saved models + {REPORT}")


if __name__ == "__main__":
    main()
