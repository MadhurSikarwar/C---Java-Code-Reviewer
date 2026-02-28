# IntelliReview — AI-Based Static Code Analyzer

## Quick Start

### 1. Install Dependencies (run once)
```
cd c:\Users\Madhu\OneDrive\Desktop\Projects\IntelliReview\backend
pip install -r requirements.txt
```

### 2. Train the ML Model (run once)
```
cd c:\Users\Madhu\OneDrive\Desktop\Projects\IntelliReview\backend
python ml/train.py
```

### 3. Start the Backend Server
```
cd c:\Users\Madhu\OneDrive\Desktop\Projects\IntelliReview\backend
python main.py
```
Server runs at: **http://localhost:8000**
API docs at: **http://localhost:8000/docs**

### 4. Open the Frontend
Double-click: `frontend/index.html`  
Or open in browser: `file:///c:/Users/Madhu/OneDrive/Desktop/Projects/IntelliReview/frontend/index.html`

---

## Project Structure

```
IntelliReview/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration & constants
│   ├── requirements.txt     # Dependencies
│   ├── parsers/
│   │   ├── c_parser.py      # C AST parser (pycparser + regex fallback)
│   │   └── java_parser.py   # Java AST parser (javalang + regex fallback)
│   ├── analyzers/
│   │   ├── memory_leak.py   # malloc/free analysis
│   │   ├── unsafe_functions.py  # gets(), strcpy(), etc.
│   │   ├── complexity.py    # Cyclomatic & time complexity
│   │   ├── recursion.py     # Recursion detection
│   │   └── feature_extractor.py # ML feature vector builder
│   ├── ml/
│   │   ├── synthetic_data.py # Training data generator
│   │   ├── train.py          # Model training (RF, XGBoost, LR)
│   │   ├── predict.py        # Inference & risk scoring
│   │   └── model.pkl         # Saved model (auto-generated)
│   ├── suggestions/
│   │   └── generator.py     # Rule-based suggestion engine
│   └── routes/
│       ├── analyze.py       # POST /api/analyze
│       └── health.py        # GET /health
└── frontend/
    ├── index.html           # Main app page
    ├── style.css            # Premium dark design system
    └── app.js               # Frontend logic
```

## API Reference

### `POST /api/analyze`
```json
{
  "code": "int main() { ... }",
  "language": "C"
}
```
Response: full JSON report with risk score, issues, suggestions, complexity metrics.

### `GET /health`
Returns server status.

## Tech Stack
- **Backend**: Python, FastAPI, pycparser, javalang
- **ML**: scikit-learn (Random Forest), XGBoost, Logistic Regression
- **Frontend**: HTML5, CSS3, Vanilla JS, Chart.js, Mermaid.js
