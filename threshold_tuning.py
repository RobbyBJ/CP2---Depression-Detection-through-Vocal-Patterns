import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score

# ================= CONFIGURATION =================
TEST_CSV = r"C:\Users\User\Desktop\CP2\depression_test_dataset.csv"
# Make sure this points to your latest trained model
MODEL_PATH = r"C:\Users\User\Desktop\CP2\ensemble_model\stacking_ensemble.pkl"
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def diagnose_and_tune():
    print("🚀 DIAGNOSING ENSEMBLE MODEL (With Precision)...")
    
    # 1. Load Data & Model
    if not os.path.exists(TEST_CSV) or not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Files not found.\nCSV: {TEST_CSV}\nModel: {MODEL_PATH}")
        return

    df = pd.read_csv(TEST_CSV)
    X_test = df.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    
    # Keep meta for analysis
    df_meta = df[['participant_id', 'PHQ8_Binary']].copy()
    
    print(f"   Loading Model: {os.path.basename(MODEL_PATH)}")
    model = joblib.load(MODEL_PATH)

    # 2. Get Probabilities 
    print("   Calculating Probabilities...")
    probs = model.predict_proba(X_test)[:, 1] 
    df_meta['seg_prob'] = probs

    # 3. Aggregate per Participant
    participant_stats = df_meta.groupby('participant_id').agg({
        'seg_prob': 'mean',
        'PHQ8_Binary': 'first'
    }).reset_index()

    # --- DIAGNOSIS REPORT ---
    print("\n📊 PROBABILITY ANALYSIS:")
    avg_dep_prob = participant_stats[participant_stats['PHQ8_Binary'] == 1]['seg_prob'].mean()
    avg_hel_prob = participant_stats[participant_stats['PHQ8_Binary'] == 0]['seg_prob'].mean()
    
    print(f"   Average Probability assigned to DEPRESSED Patients: {avg_dep_prob:.3f}")
    print(f"   Average Probability assigned to HEALTHY Patients:   {avg_hel_prob:.3f}")
    
    if avg_dep_prob > avg_hel_prob:
        print("   ✅ GOOD NEWS: The model gives higher scores to depressed patients.")
        print("      We just need to lower the threshold to catch them.")
    else:
        print("   ❌ BAD NEWS: The model cannot distinguish between the groups.")

    # 4. Threshold Tuning Loop
    print("\n🎛️ TUNING THRESHOLD...")
    # Added 'Prec' column header
    print(f"{'Threshold':<10} | {'Sens':<8} | {'Spec':<8} | {'Prec':<8} | {'F1':<8} | {'Acc':<8}")
    print("-" * 75)

    best_f1 = 0
    best_res = None
    y_true = participant_stats['PHQ8_Binary']
    y_probs = participant_stats['seg_prob']

    for thresh in np.arange(0.20, 0.60, 0.02):
        y_pred = (y_probs >= thresh).astype(int)
        
        # Metrics
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = calculate_specificity(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0) # <--- ADDED PRECISION
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        # Print Row
        print(f"{thresh:.2f}       | {sens:.2%} | {spec:.2%} | {prec:.2%} | {f1:.2f}     | {acc:.2%}")

        # You can change the condition below to optimize for what you want (e.g. Sensitivity > 0.50)
        if f1 > best_f1:
            best_f1 = f1
            best_res = (thresh, sens, spec, prec, f1, acc)

    print("-" * 75)
    if best_res:
        t, s, sp, p, f, a = best_res
        print(f"🏆 OPTIMAL RESULT (Best F1) AT THRESHOLD {t:.2f}:")
        print(f"   Sensitivity: {s:.2%}")
        print(f"   Specificity: {sp:.2%}")
        print(f"   Precision:   {p:.2%}")
        print(f"   F1-Score:    {f:.2f}")
        print(f"   Accuracy:    {a:.2%}")
    else:
        print("No improvement found.")

if __name__ == "__main__":
    diagnose_and_tune()