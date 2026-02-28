"""
IntelliReview — Tests: ML Pipeline
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.synthetic_data import generate_synthetic_data, FEATURE_NAMES, LABEL_MAP


def test_synthetic_data_shape():
    df = generate_synthetic_data(n=300)
    assert len(df) == 300
    for feat in FEATURE_NAMES:
        assert feat in df.columns, f"Missing column: {feat}"
    assert "label" in df.columns
    assert "label_name" in df.columns
    print("✅ test_synthetic_data_shape passed")


def test_synthetic_data_labels():
    df = generate_synthetic_data(n=600)
    counts = df["label_name"].value_counts()
    assert "Clean" in counts.index
    assert "Moderate Risk" in counts.index
    assert "High Risk" in counts.index
    print("✅ test_synthetic_data_labels passed")


def test_synthetic_feature_ranges():
    df = generate_synthetic_data(n=500)
    # All features should be non-negative integers
    for feat in FEATURE_NAMES:
        assert (df[feat] >= 0).all(), f"Negative values in {feat}"
    print("✅ test_synthetic_feature_ranges passed")


def test_train_and_predict():
    """Train a quick model and verify prediction output."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = generate_synthetic_data(n=600)
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    assert acc > 0.70, f"Accuracy too low: {acc:.2f}"
    print(f"✅ test_train_and_predict passed (accuracy={acc:.3f})")


def test_feature_vector_order():
    """Feature vector must follow FEATURE_NAMES order for model compatibility."""
    from analyzers.feature_extractor import extract_features
    dummy = {
        "num_functions": 3, "num_loops": 2, "max_nesting_depth": 1,
        "num_conditionals": 4, "num_pointer_uses": 5, "num_mallocs": 2,
        "num_frees": 2, "lines_of_code": 80, "function_call_count": 10,
        "language": "C",
    }
    dummy_mem = {"memory_leak_count": 0}
    dummy_unsafe = {"unsafe_function_count": 0}
    dummy_cx = {"cyclomatic_complexity": 5}
    dummy_rec = {"recursion_count": 0}

    result = extract_features(dummy, dummy_mem, dummy_unsafe, dummy_cx, dummy_rec)
    assert result["feature_names"] == FEATURE_NAMES
    assert len(result["feature_vector"]) == len(FEATURE_NAMES)
    print("✅ test_feature_vector_order passed")


if __name__ == "__main__":
    test_synthetic_data_shape()
    test_synthetic_data_labels()
    test_synthetic_feature_ranges()
    test_train_and_predict()
    test_feature_vector_order()
    print("\n🎉 All ML tests passed!")
