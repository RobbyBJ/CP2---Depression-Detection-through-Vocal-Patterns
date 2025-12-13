import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ================= CONFIGURATION =================
TRAIN_CSV = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"
MODEL_OUTPUT_DIR = r"C:\Users\User\Desktop\CP2\baseline_models"
RANDOM_STATE = 42

# RFE Settings
N_FEATURES_TO_KEEP = 30  # Keep top 30 features (Removes noise)
RFE_STEP = 5             # Remove 5 features at a time (Speeds up training)
# =================================================

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def run_training():
    print("🚀 LOADING TRAIN DATA (Segment-Level)...")
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ Error: {TRAIN_CSV} not found.")
        return

    df_train = pd.read_csv(TRAIN_CSV)
    
    # Drop metadata for training
    X_train = df_train.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    y_train = df_train['PHQ8_Binary']

    print(f"✅ Training on {len(X_train)} segments.")
    
    # Calculate Class Imbalance for XGBoost
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"   Class Balance: {y_train.value_counts().to_dict()}")
    print(f"   XGBoost Scale Weight: {scale_pos_weight:.2f}")

    # ================= DEFINE RFE SELECTOR =================
    # Use a simple Decision Tree to judge feature importance for the RFE step.
    # Fast and effective at finding non-linear relationships.
    rfe_selector = RFE(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_features_to_select=N_FEATURES_TO_KEEP,
        step=RFE_STEP
    )

    # ================= DEFINE MODELS =================
    # Added class_weight='balanced' to boost Sensitivity
    models_config = {
        'SVM': SVC(
            random_state=RANDOM_STATE, 
            probability=True, 
            class_weight='balanced'  # <--- BOOST SENSITIVITY
        ),
        
        'RandomForest': RandomForestClassifier(
            random_state=RANDOM_STATE, 
            n_estimators=200, 
            max_depth=10, 
            n_jobs=-1,
            class_weight='balanced'  # <--- BOOST SENSITIVITY
        ),
        
        'LogisticRegression': LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=2000,
            class_weight='balanced'  # <--- BOOST SENSITIVITY
        ),
        
        'KNN': KNeighborsClassifier(n_neighbors=9), # KNN doesn't support class_weights natively
        
        'XGBoost': XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=4, 
            tree_method='hist', 
            eval_metric='logloss', 
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight  # <--- BOOST SENSITIVITY
        )
    }

    print(f"\n⚔️ STARTING TRAINING WITH RFE & CLASS WEIGHTS...")

    for name, model in models_config.items():
        print(f"\n🧩 Training {name}...")
        
        # Create Pipeline
        # 1. Impute missing values
        # 2. Scale features (StandardScaler)
        # 3. RFE (Select Best Features)
        # 4. Train Classifier
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('feature_selection', rfe_selector),
            ('classifier', model)
        ])

        try:
            pipeline.fit(X_train, y_train)
            
            # Save Model
            save_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}_v8.pkl")
            joblib.dump(pipeline, save_path)
            print(f"✅ Saved: {save_path}")

        except Exception as e:
            print(f"❌ Failed {name}: {e}")

    print("\n🎉 Training Complete. The models are now 'Recall-Optimized'.")

if __name__ == "__main__":
    run_training()