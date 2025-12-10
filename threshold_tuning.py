import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, recall_score, confusion_matrix

# ================= CONFIGURATION =================
# Path to your Stacking Model
MODEL_PATH = r"C:\Users\User\Desktop\CP2\ensemble_models\stacking_ridge.pkl"
# Path to Baseline Handcrafted Dev Data
TEST_DATASET = r"C:\Users\User\Desktop\CP2\depression_test_dataset.csv"
# =================================================

def main():
    print("🚀 TUNING ENSEMBLE THRESHOLD...")
    
    # 1. Load Data & Model
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found!")
        return
    
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_DATASET)
    
    # Align features (Drop non-features)
    X_test = df.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    y_test = df['PHQ8_Binary']
    
    print(f"✅ Loaded Model & {len(X_test)} Test Samples.")

    # 2. Get Probabilities (Confidence Scores)
    # StackingClassifier usually has predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1] # Probability of Class 1 (Depression)
    else:
        # If RidgeClassifier was used as final_estimator, it might use decision_function
        print("⚠️ Model doesn't support probability. Using decision function...")
        probs = model.decision_function(X_test)
        # Normalize to 0-1 range for easier thresholding
        probs = (probs - probs.min()) / (probs.max() - probs.min())

    # 3. Test Thresholds from 0.1 to 0.9
    print("\n📊 TESTING THRESHOLDS:")
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Specificity':<12} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 65)
    
    best_thresh = 0.5
    best_f1 = 0
    
    thresholds = np.arange(0.1, 0.6, 0.05) # Test 0.10, 0.15, ... 0.55
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        rec = recall_score(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        spec = tn / (tn + fp)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        print(f"{t:.2f}       | {rec:.4f}     | {spec:.4f}       | {acc:.4f}     | {f1:.4f}")
        
        # Logic: We want Recall > 0.80 but Maximize F1
        if rec > 0.80 and f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print("-" * 65)
    print(f"\n🏆 RECOMMENDATION:")
    print(f"To fix the low recall, use Threshold = {best_thresh:.2f}")
    
    # Show final stats for recommended threshold
    final_preds = (probs >= best_thresh).astype(int)
    final_rec = recall_score(y_test, final_preds)
    final_acc = accuracy_score(y_test, final_preds)
    
    print(f"   -> New Sensitivity: {final_rec:.2%}")
    print(f"   -> New Accuracy:    {final_acc:.2%}")

if __name__ == "__main__":
    main()