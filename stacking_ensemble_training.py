import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ================= CONFIGURATION =================
# 1. Path to train dataset
TRAIN_CSV = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"

# 2. Path to tuned models
TUNED_MODEL_DIR = r"C:\Users\User\Desktop\CP2\tuned_models"

# 3. Output for the Final Ensemble model
OUTPUT_DIR = r"C:\Users\User\Desktop\CP2\ensemble_model"
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tuned_models():
    print("🔍 Searching for tuned models...")
    estimators = []
    
    # Look for the .pkl files created by the Tuning Script
    if not os.path.exists(TUNED_MODEL_DIR):
        print("❌ Tuned model directory not found.")
        return []

    for f in os.listdir(TUNED_MODEL_DIR):
        if f.endswith(".pkl"):
            name = f.replace(".pkl", "")
            path = os.path.join(TUNED_MODEL_DIR, f)
            
            try:
                # Load the full pipeline (RFE + Classifier)
                model = joblib.load(path)
                estimators.append((name, model))
                print(f"   ✅ Loaded: {name}")
            except:
                print(f"   ❌ Failed to load: {f}")
                
    return estimators

def train_stacking():
    print("🚀 STARTING STACKING ENSEMBLE TRAINING...")
    
    # 1. Load Data
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ Error: {TRAIN_CSV} not found.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    X_train = df.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    y_train = df['PHQ8_Binary']
    
    # 2. Load Base Learners
    estimators = load_tuned_models()
    if not estimators:
        print("❌ No models found! Run 'train_tuned_models_with_rfe.py' first.")
        return

    print(f"\n🧠 Building Meta-Learner with {len(estimators)} base models...")

    # 3. Define Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            class_weight='balanced',  # <--- THIS IS THE KEY FIX
            random_state=42,
            max_iter=2000
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        passthrough=False 
    )
    
    print("\n⏳ Training Meta-Learner (This takes time as it cross-validates)...")
    try:
        stacking_clf.fit(X_train, y_train)
        
        # 4. Save
        save_path = os.path.join(OUTPUT_DIR, "Final_Stacking_Ensemble_v9.pkl")
        joblib.dump(stacking_clf, save_path)
        print(f"\n✅ ENSEMBLE TRAINED & SAVED: {save_path}")
        print("👉 Now test this using 'test_v9_soft_voting.py'!")
        
    except Exception as e:
        print(f"❌ Stacking Failed: {e}")

if __name__ == "__main__":
    train_stacking()