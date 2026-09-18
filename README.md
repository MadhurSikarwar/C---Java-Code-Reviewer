# IntelliReview

A static analyzer for **C, C++ and Java**. Paste or open a file and it comes back marked up: the exact lines that worry a
reviewer, what is wrong with each, what to change, and a risk verdict.

## Run it

```bat
run.bat
```

then open <http://127.0.0.1:8000>. (`run.bat` uses the virtual environment in `.venv`; the backend also serves the web page.)

First-time setup, if you are on a fresh checkout:

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r backend\requirements.txt
.venv\Scripts\python backend\ml\retrain_all.py      REM models are not stored in git; see "Retraining"
```

API docs: <http://127.0.0.1:8000/docs>. Main endpoint: `POST /api/analyze` with `{"code", "language": "C"|"C++"|"Java", "model_type": "v3"|"dl"|"ensemble"|"v4_dl"|"all"}`.
Bad input gets a 4xx with a readable message (empty, binary, not code, unbalanced brackets, >200 KB / 6000 lines).

## How a verdict is reached

1. **Parse** (`backend/parsers`): pycparser for C (unknown typedefs are learned and the parse retried), javalang for Java,
   pattern-based fallback for C++.
2. **Analyse** (`backend/analyzers`, `backend/parsers/c_bounds.py`, `c_taint.py`): control-flow graph, path-sensitive pointer and
   resource states (use-after-free, double/invalid free, NULL dereference, leaks, unchecked allocation), definite assignment,
   buffer-bounds intervals, taint tracking, constant-condition folding. Java: SQL/command injection, XSS, path traversal,
   deserialization, weak crypto, hard-coded credentials, resource leaks (`java_security.py`).
3. **Grade** each finding; only Critical/High *security* findings can produce "High risk" (`analyzers/issue_taxonomy.py`).
4. **Weigh** with four models (`backend/ml`). Their answer is bounded by the evidence (`ml/predict.py`): no High risk without a
   serious finding, a serious finding always means High, and no Moderate without something concrete to point at. The site's "How it decides" page explains this with live, measured numbers.

`backend/pipeline.py` is the single place where code becomes findings + the 36-number feature vector, and it is shared by the API and
the training scripts, so the models train on exactly what they see in production.

## Beyond the browser

```bat
cd backend
..\.venv\Scripts\python cli.py src\ --fail-on high            REM CI gate: exit code 1 on High/Critical findings
..\.venv\Scripts\python cli.py src\ --format sarif -o r.sarif   REM GitHub code scanning / VS Code SARIF viewer
```

* **Suppress a finding you have accepted:** `// intellireview: ignore` (same line, or the line above), `// intellireview: ignore[USE_AFTER_FREE]`,
  or `/* intellireview: ignore-file */`. Suppressed findings are counted and reported, never hidden silently.
* **Suggested edits:** for findings with a mechanical, safe repair (`printf(x)` -> `printf("%s", x)`, `gets` / `strcpy` / `sprintf` on a declared array,
  MD5 -> SHA-256, empty `catch`, hard-coded credential), the UI shows the before/after line and an *Apply* button. If a fix is only correct under an
  assumption we can't verify (e.g. `sizeof(ptr)`), no edit is offered.
* `.github/workflows/ci.yml` runs the tests on every push.

## Retraining

The models are **not** stored in git. Training data is the NIST Juliet test suites (function-level `_bad`/`_good` labels), glued into
larger files so file size cannot predict the label, plus generated programs.

```bat
cd backend
..\.venv\Scripts\python ml\juliet_extract.py --c-root "<...>\juliet-test-suite-for-c-cplusplus-v1-3\C\testcases" --java-root "<...>\juliet-test-suite-for-java-v1-3\Java\src\testcases"
..\.venv\Scripts\python ml\build_features.py
..\.venv\Scripts\python ml\retrain_all.py
```

Without Juliet, `build_features.py` still works using generated programs only. `retrain_all.py` evaluates with whole vulnerability classes held
out, compares against a size-only baseline and the rules alone, chooses the evidence-gate variant by measurement, and writes
`backend/ml/model_report.json`, which the web page reads. `ml/evaluate_rules.py` prints per-CWE recall / false-alarm rates of the rule engine.
(The older `train*.py` scripts train on synthetic size-only data and are kept only for reference; do not use them.)

## Tests

```bat
cd backend
..\.venv\Scripts\python -m pytest tests -q                 REM 77 rule/regression tests, no server needed
..\.venv\Scripts\python tests\adversarial_probe.py         REM hostile inputs against a running server
```

## Known limits

It does not find logic errors, integer overflow, races, ownership that crosses files, or (in Java) taint that crosses methods; C++ gets
a lighter, heuristic read. Long files collect small findings and are more often called Moderate. Details and the measured numbers are on
the "How it decides" page.
