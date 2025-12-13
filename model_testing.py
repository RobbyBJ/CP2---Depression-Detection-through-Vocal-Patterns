import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
)

# ================= CONFIGURATION =================
TEST_DATASET = r"C:\Users\User\Desktop\CP2\depression_test_dataset.csv"
MODEL_DIR = r"C:\Users\User\Desktop\CP2\ensemble_model"
OUTPUT_RESULTS = r"C:\Users\User\Desktop\CP2\ensemble__results.csv"

#  MANUAL THRESHOLD SETTIN
MANUAL_THRESHOLD = 0.50
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def align_features(model, X):
    if hasattr(model, "feature_names_in_"):
        trained_features = model.feature_names_in_
        X_aligned = X.reindex(columns=trained_features, fill_value=0)
        return X_aligned
    return X

def get_participant_scores(model, X_test, df_meta):
    """
    Returns the average probability score (Risk Score) for each participant.
    """
    # 1. Segment Probabilities
    if hasattr(model, "predict_proba"):
        seg_probs = model.predict_proba(X_test)[:, 1]
    else:
        seg_probs = model.predict(X_test)

    # 2. Combine with Metadata
    results = df_meta.copy()
    results['seg_prob'] = seg_probs
    
    # 3. Aggregation (Mean Risk Score per Patient)
    participant_scores = results.groupby('participant_id')['seg_prob'].mean()
    true_labels = results.groupby('participant_id')['PHQ8_Binary'].first()
    
    return participant_scores, true_labels

def main():
    print(f"🚀 STARTING EVALUATION WITH MANUAL THRESHOLD: {MANUAL_THRESHOLD}")
    print(f"📘 Data: {TEST_DATASET}")
    
    if not os.path.exists(TEST_DATASET) or not os.path.exists(MODEL_DIR):
        print("❌ Error: Dataset or Model Directory not found.")
        return

    # Load Data
    df = pd.read_csv(TEST_DATASET)
    X_test = df.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    df_meta = df[['participant_id', 'PHQ8_Binary']]
    
    print(f"✅ Loaded {len(X_test)} segments from {df_meta['participant_id'].nunique()} participants.")

    # Find Models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    
    final_results = []

    print("\n⚔️ FINAL MODEL REPORT")
    print(f"{'Model':<25} | {'Acc':<8} | {'Sens':<8} | {'Spec':<8} | {'F1':<8}")
    print("-" * 75)

    for f in model_files:
        model_path = os.path.join(MODEL_DIR, f)
        model_name = f.replace(".pkl", "")
        
        try:
            model = joblib.load(model_path)
            X_test_aligned = align_features(model, X_test)
            
            # 1. Get Scores
            scores, labels = get_participant_scores(model, X_test_aligned, df_meta)
            
            # 2. Apply Manual Threshold
            y_pred = (scores >= MANUAL_THRESHOLD).astype(int)
            
            # 3. Calculate Metrics
            acc = accuracy_score(labels, y_pred)
            sens = recall_score(labels, y_pred)
            spec = calculate_specificity(labels, y_pred)
            f1 = f1_score(labels, y_pred)
            prec = precision_score(labels, y_pred, zero_division=0)
            
            # Print Row
            print(f"{model_name:<25} | {acc:.2%} | {sens:.2%} | {spec:.2%} | {f1:.2f}")
            
            # Add to Log
            entry = {
                "Model": model_name,
                "Threshold": MANUAL_THRESHOLD,
                "Accuracy": acc,
                "Sensitivity": sens,
                "Specificity": spec,
                "F1-Score": f1,
                "Precision": prec
            }
            final_results.append(entry)
            
        except Exception as e:
            print(f"❌ Error {model_name}: {e}")

    # Save
    if final_results:
        res_df = pd.DataFrame(final_results).sort_values(by="F1-Score", ascending=False)
        res_df.to_csv(OUTPUT_RESULTS, index=False)
        print(f"\n✅ Results saved to: {OUTPUT_RESULTS}")
        
        # Print detailed report for the best model
        best_model_name = res_df.iloc[0]['Model']
        print(f"\n📝 Detailed Report for Best Model: {best_model_name}")

if __name__ == "__main__":
    main()