import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ================= CONFIGURATION =================
INPUT_TRAIN_CSV = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"
MODEL_OUTPUT_DIR = r"C:\Users\User\Desktop\CP2\baseline_models"
RANDOM_STATE = 42
# =================================================

# --- Ensure output directory exists ---
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def run_baseline_training():
    print("🚀 LOADING TRAINING DATASET...")
    df = pd.read_csv(INPUT_TRAIN_CSV)

    if 'PHQ8_Binary' not in df.columns or 'participant_id' not in df.columns:
        raise ValueError("❌ Missing 'PHQ8_Binary' or 'participant_id' columns in dataset!")

    # Prepare data
    X_train = df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_train = df['PHQ8_Binary']

    print(f"✅ Loaded {len(X_train)} training samples.")
    print(f"   Class balance: {y_train.value_counts().to_dict()}")

    # --- CALCULATE IMBALANCE FOR XGBOOST ---
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"   XGBoost Scale Weight: {scale_weight:.2f}")

    # --- DEFINE BASELINE MODELS ---
    models_config = {
        'SVM': SVC(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            probability=True
        ),

        'RandomForest': RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ),

        'LogisticRegression': LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        ),

        'KNN': KNeighborsClassifier(n_neighbors=5),

        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=scale_weight,
            tree_method='hist',
            eval_metric='logloss',
            random_state=RANDOM_STATE
        )
    }

    print("\n⚔️ STARTING BASELINE MODEL TRAINING...")

    for name, model in models_config.items():
        print(f"\n🧩 Training {name}...")

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        try:
            pipeline.fit(X_train, y_train)

            # Save model
            model_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}_baseline.pkl")
            joblib.dump(pipeline, model_path)

            print(f"✅ {name} model saved to: {model_path}")

        except Exception as e:
            print(f"❌ Failed to train {name}: {e}")

    print("\n🎉 All models trained and saved successfully!")


if __name__ == "__main__":
    run_baseline_training()
