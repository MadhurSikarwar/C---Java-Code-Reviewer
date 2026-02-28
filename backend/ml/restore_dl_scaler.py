import sys
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_dl_model import load_all_data, SCALER_PATH

def restore_scaler():
    print("Loading datasets...")
    X, y, class_weights = load_all_data()

    print("Splitting identically to original training run (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting Scaler...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Successfully restored {SCALER_PATH}!")
    
if __name__ == "__main__":
    restore_scaler()
