"""
IntelliReview — first-run bootstrap

Trained models are not stored in git. When `predict.load_model()` finds none (a fresh clone), this builds them with the SAME
honest pipeline used everywhere else (ml/build_features.py -> ml/retrain_all.py), never the old size-only synthetic trainer.

  * if ml/data/juliet_samples.jsonl exists (see ml/juliet_extract.py) it is used together with generated programs;
  * otherwise generated programs only — fewer real-world signals, but the size shortcut is still gone.

Run manually:   python ml/bootstrap.py [--fast]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)


def models_present() -> bool:
    return all(os.path.exists(os.path.join(HERE, f)) for f in ("model.pkl", "model_v3_path_sensitive.pkl"))


def bootstrap(fast: bool = False) -> None:
    """Build features + train. `fast` skips the two neural networks and shrinks the generated set (~1 minute)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    juliet = os.path.exists(os.path.join(HERE, "data", "juliet_samples.jsonl"))
    print(f"[bootstrap] training models ({'with Juliet data' if juliet else 'generated programs only'}"
          f"{', fast' if fast else ''}) — this happens once", flush=True)
    build = [sys.executable, os.path.join(HERE, "build_features.py"),
             "--generated", "2500" if fast else "6000", "--composites", "1500" if juliet and fast else "4000" if juliet else "0"]
    train = [sys.executable, os.path.join(HERE, "retrain_all.py")] + (["--no-dl"] if fast else [])
    for cmd in (build, train):
        subprocess.run(cmd, check=True, cwd=BACKEND, env=env)


if __name__ == "__main__":
    bootstrap(fast="--fast" in sys.argv)
