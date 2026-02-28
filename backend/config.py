"""
IntelliReview — App Configuration
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "ml", "label_encoder.pkl")

# API settings
API_TITLE = "IntelliReview API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "AI-Based Static Code Analyzer for C and Java"

# Allowed origins for CORS (frontend)
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "null",  # Allows file:// origin
]

# Risk score thresholds
RISK_CLEAN_MAX = 35
RISK_MODERATE_MAX = 65
# Above 65 → High Risk

# Feature names (order must match ML training)
FEATURE_NAMES = [
    "num_functions",
    "num_loops",
    "max_nesting_depth",
    "num_conditionals",
    "num_pointer_uses",
    "num_mallocs",
    "num_frees",
    "memory_leak_count",
    "unsafe_function_count",
    "cyclomatic_complexity",
    "recursion_count",
    "lines_of_code",
    "function_call_count",
]
