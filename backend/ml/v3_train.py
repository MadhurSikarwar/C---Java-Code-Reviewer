"""
IntelliReview V3 — Path-Sensitive ML Model Training
===================================================
Trains the Random Forest model exclusively on the new V3 CFG/DFA features.
Outputs `model_v3_path_sensitive.pkl`.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
try:
    from ml.v3_synthetic_data import V3_FEATURE_NAMES
except ImportError:
    from v3_synthetic_data import V3_FEATURE_NAMES

def train():
    data_path = os.path.join(os.path.dirname(__file__), "v3_training_data.csv")
    if not os.path.exists(data_path):
        print("Required v3_training_data.csv not found. Did you run v3_synthetic_data.py?")
        return
        
    df = pd.read_csv(data_path)
    X = df[V3_FEATURE_NAMES]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🧠 Training V3 Path-Sensitive Random Forest (100 estimators)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Training Complete. Validation Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=["Clean", "Moderate Risk", "High Risk"]))
    
    # Save the isolated V3 model
    out_path = os.path.join(os.path.dirname(__file__), "model_v3_path_sensitive.pkl")
    joblib.dump(clf, out_path)
    print(f"💾 V3 Model saved to: {out_path}")

if __name__ == "__main__":
    train()
