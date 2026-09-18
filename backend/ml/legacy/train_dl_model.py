"""
IntelliReview — Deep Learning Trainer  (V1 — 19 features)
===========================================================
Uses TensorFlow/Keras to train a compact, well-regularised DNN.

What this script does:
  1. Loads all datasets (Synthetic + Juliet C/Java + Real) — 19 features
  2. Standardizes the features with StandardScaler
  3. Builds a right-sized architecture:
       Input(19) -> Dense(256) -> Dense(128) -> Dense(64) -> Output(3)
  4. Saves to: model_dl_v1_19feat.keras  (scaler: dl_scaler_v1_19feat.pkl)
     (Leaves model.pkl and V3 models completely untouched!)

Requirements:
  pip install tensorflow pandas scikit-learn
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Attempt to load TensorFlow
try:
    # Python 3.8+ Windows workaround for CUDA DLLs
    if os.name == 'nt':
        try:
            os.add_dll_directory(r"C:\CUDA_Manual\bin")
        except Exception:
            pass

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError as e:
    print("❌ TensorFlow import failed!")
    print(f"Details: {e}")
    print("   Please ensure it's installed and healthy: pip install tensorflow")
    sys.exit(1)

# Ensure path is correct for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.synthetic_data import generate_synthetic_data, FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_DIR             = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH           = os.path.join(MODEL_DIR, "dl_scaler_v1_19feat.pkl")
DL_MODEL_PATH         = os.path.join(MODEL_DIR, "model_dl_v1_19feat.keras")

REAL_DATA_PATH        = os.path.join(MODEL_DIR, "real_data.csv")
JULIET_DATA_PATH      = os.path.join(MODEL_DIR, "juliet_data.csv")
JULIET_JAVA_DATA_PATH = os.path.join(MODEL_DIR, "juliet_java_data.csv")

N_SYNTH = 10000

# ── Data Loading (Same as Ensemble pipeline) ──────────────────────────────
def load_all_data():
    print(f"🔧 Generating {N_SYNTH:,} synthetic samples...")
    df_synth = generate_synthetic_data(n=N_SYNTH)
    
    frames        = [df_synth]
    n_juliet_c    = 0
    n_juliet_java = 0
    n_real        = 0

    if os.path.exists(JULIET_DATA_PATH):
        df_j = pd.read_csv(JULIET_DATA_PATH)
        missing = [f for f in FEATURE_NAMES if f not in df_j.columns]
        if not missing:
            frames.append(df_j[FEATURE_NAMES + ["label"]])
            n_juliet_c = len(df_j)

    if os.path.exists(JULIET_JAVA_DATA_PATH):
        df_jj = pd.read_csv(JULIET_JAVA_DATA_PATH)
        missing = [f for f in FEATURE_NAMES if f not in df_jj.columns]
        if not missing:
            frames.append(df_jj[FEATURE_NAMES + ["label"]])
            n_juliet_java = len(df_jj)

    if os.path.exists(REAL_DATA_PATH):
        df_real = pd.read_csv(REAL_DATA_PATH)
        missing = [f for f in FEATURE_NAMES if f not in df_real.columns]
        if not missing:
            frames.append(df_real[FEATURE_NAMES + ["label"]])
            n_real = len(df_real)

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[FEATURE_NAMES].values.astype(float)
    y = df["label"].values.astype(int)

    print(f"\n📊 Total dataset ready for Deep Learning: {len(df):,} samples")
    print(f"   (Synthetic: {N_SYNTH} | Juliet-C: {n_juliet_c} | Juliet-Java: {n_juliet_java})")
    
    # Needs balanced class weights because C and Java datasets might lean heavily towards Moderate/High Risk
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    class_weights = {}
    for c, count in zip(classes, counts):
        class_weights[c] = total / (len(classes) * count)
        
    return X, y, class_weights

# ── Neural Network Architecture ───────────────────────────────────────────
def build_dnn(input_dim):
    """
    Right-sized DNN for 19 tabular features → 3 classes.
    ~100k parameters — fast to train, not prone to overfitting.
    """
    print(f"\n🏗️ Building DNN for {input_dim} features...")
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation="relu"),
        BatchNormalization(),

        # Output layer (3 classes: Clean, Moderate Risk, High Risk)
        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model

# ── Main Training Loop ───────────────────────────────────────────────────
def main():
    print("🚀 INITIALIZING INTELLIREVIEW DEEP LEARNING (MASSIVE SCALE)")
    
    # 1. Load Data
    X, y, class_weights = load_all_data()
    
    # DL requires categorical output to be one-hot encoded during training mapping 
    # But `sparse_categorical_crossentropy` handles integer labels automatically!

    # 2. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Standard Scaling (CRITICAL for Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Save the scaler so the web app can use it during inference
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n💾 Saved Scaler to: {SCALER_PATH}")

    # 4. Build Model
    model = build_dnn(input_dim=X_train.shape[1])

    # 5. Callbacks — monitor val_loss for stable convergence
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    # 6. Train the Model
    print("\n🔥 Training DL Model (V1 — 19 features)...")

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=60,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # 7. Evaluate
    print("\n🧪 Evaluating against unseen test data...")
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "="*60)
    print(f"🏆 DEEP LEARNING MODEL ACHIEVED: {test_acc:.4f} ({test_acc*100:.2f}%) ACCURACY")
    print("="*60)
    
    labels = ["Clean", "Moderate Risk", "High Risk"]
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 8. Save
    model.save(DL_MODEL_PATH)

    # Print file size
    mb_size = os.path.getsize(DL_MODEL_PATH) / (1024 * 1024)
    print(f"\n💾 DL Model (V1/19-feat) SAVED SUCCESSFULLY.")
    print(f"   Path:  {DL_MODEL_PATH}")
    print(f"   Size:  {mb_size:.2f} MB")
    print(f"   Scaler: {SCALER_PATH}")
    print("\n🎉 DONE! Run predict.py with model_type='dl' to use this model.")

if __name__ == "__main__":
    # Prevent TF from taking over all VRAM aggressively if user has a GPU
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
    
    main()
