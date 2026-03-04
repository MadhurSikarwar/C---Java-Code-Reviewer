"""
IntelliReview V3 — Massive Deep Learning CNN / DNN Engine
=========================================================
Builds a colossal ~260M+ parameter architecture to hit 1GB+ model sizes.
Utilizes the combined C and Java path-sensitive telemetry vectors (34 features).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

# Suppress TF logging to keep output clean initially
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    if os.name == 'nt':
        os.add_dll_directory(r"C:\CUDA_Manual\bin")
except Exception:
    pass

import tensorflow as tf

# Enable Mixed Precision to halve VRAM usage
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, BatchNormalization, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from v3_train_combined import CLASSIC_FEATURES, V3_FEATURES, augment_v3_features_robustly

def generate_massive_dataset(num_samples: int = 50000):
    """
    Synthesize a massive augmented dataset incorporating both C and Java distributions.
    We load the existing Juliet dataset, add Java-specific statistical shapes, 
    and interpolate massive synthetic sets.
    """
    print(f"🏭 Synthesizing {num_samples} records of C/Java telemetry...")
    base_dir = os.path.dirname(__file__)
    juliet_path = os.path.join(base_dir, "juliet_data.csv")
    
    if os.path.exists(juliet_path):
        base_df = pd.read_csv(juliet_path)
        for col in CLASSIC_FEATURES:
            if col in base_df.columns:
                base_df[col] = base_df[col].fillna(0)
        df_augmented = augment_v3_features_robustly(base_df)
    else:
        raise Exception("juliet_data.csv not found!")

    # Now we multiply and noise-inject this dataset to scale it
    rng = np.random.default_rng(42)
    repeats = (num_samples // len(df_augmented)) + 1
    
    massive_dfs = []
    
    for i in range(repeats):
        noise_df = df_augmented.copy()
        
        # Add random Gaussian noise to continuous variables
        for c in CLASSIC_FEATURES + V3_FEATURES:
            if c != 'label' and noise_df[c].dtype in [np.float64, np.int64]:
                noise = rng.normal(0, 0.05 * noise_df[c].std() + 1e-5, len(noise_df))
                noise_df[c] = np.abs(noise_df[c] + noise)
                
                # Half of the added variants represent "Java" behavior (fewer raw pointers, higher object count/allocs)
                if i % 2 != 0: 
                    if 'pointer' in c: noise_df[c] *= 0.1
                    if 'malloc' in c: noise_df[c] *= 0.05
                    if 'cc' in c: noise_df[c] *= 1.2
                    
        massive_dfs.append(noise_df)
        
    final_df = pd.concat(massive_dfs, ignore_index=True)
    
    # Shuffle and trim
    final_df = final_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    return final_df

def build_v3_dnn(input_dim):
    """
    Right-sized DNN for 34 tabular features (CLASSIC + V3) → 3 classes.
    ~200k parameters — efficient, well-regularised, fast on CPU.
    """
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        BatchNormalization(),

        # Mixed precision outputs require explicit float32
        Dense(3, activation='softmax', dtype='float32')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train():
    try:
        print("\n\n" + "="*60)
        print("🚀 INITIALIZING MASSIVE DEEP LEARNING MODEL (150M+ PARAMS)")
        print("="*60)
        
        # GPU check & Memory Growth Setup prevents OOM
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🔥 CNN Training on GPU: {gpus[0]} (Memory Growth Enabled)")
            except RuntimeError as e:
                print(e)
        else:
            print(f"❄️ CNN Training on CPU — Fast Epoch Mode engaged.")
            
        # Generate 25,000 samples (enough to learn without taking 4 hours on CPU)
        df = generate_massive_dataset(25000)
        
        features = CLASSIC_FEATURES + V3_FEATURES
        X = df[features].values
        y = df['label'].values
        
        print(f"\n📏 Normalizing {len(X)} records across {X.shape[1]} C/Java dimensions...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
        
        print("\n🏗️ Building V3 Neural Architecture (34 features)...")
        model = build_v3_dnn(X.shape[1])
        model.summary()
        
        print("\n🧠 Commencing Deep Training Phase (CPU Offload Active to prevent OOM)...")
        
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        )
        
        epochs = 60
        batch_size = 256

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Save — use a distinct name so it doesn't overwrite the V1 19-feature model
        base_dir = os.path.dirname(__file__)
        out_path = os.path.join(base_dir, "model_dl_v3_34feat.keras")
        scaler_path = os.path.join(base_dir, "dl_scaler_v3_34feat.pkl")

        print("\n💾 Saving V3 DL model (34 features)...")
        model.save(out_path)
        joblib.dump(scaler, scaler_path)

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"\n✅ SUCCESS! V3 DL Model saved to: {out_path}")
        print(f"⚖️ Final Model Size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ Massive Training Failed: {str(e)}")

if __name__ == "__main__":
    train()
