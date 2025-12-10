import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ================= CONFIGURATION =================
TEST_DATASET = r"C:\Users\User\Desktop\CP2\depression_test_dataset.csv"
MODEL_DIR = r"C:\Users\User\Desktop\CP2\ensemble_models"
OUTPUT_RESULTS = r"C:\Users\User\Desktop\CP2\ensemble_models_test_results.csv"
# =================================================

# --- Helper Function ---
def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def align_features(model, X):
    """Ensure X_test matches model's training features exactly."""
    # Some models (especially pipelines) store feature names
    if hasattr(model, "feature_names_in_"):
        trained_features = model.feature_names_in_
        X_aligned = X.reindex(columns=trained_features, fill_value=0)
        return X_aligned
    else:
        # Fall back if model doesn't have feature_names_in_
        return X

def evaluate_model(model_path, X_test, y_test):
    """Load model and evaluate metrics."""
    model_name = os.path.basename(model_path).replace(".pkl", "")
    print(f"\n🧠 Evaluating {model_name}...")

    model = joblib.load(model_path)

    # Align test features to training features
    X_test = align_features(model, X_test)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    spec = calculate_specificity(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    recall = report.get('1', {}).get('recall', 0.0)
    precision = report.get('1', {}).get('precision', 0.0)

    print(f"   → Accuracy: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | Spec: {spec:.4f}")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "F1-Score": f1,
        "Recall (Sensitivity)": recall,
        "Precision": precision,
        "Specificity": spec
    }

def main():
    print("🚀 STARTING TUNED MODEL EVALUATION")
    print(f"📘 Loading test dataset from: {TEST_DATASET}")

    # Load dataset
    df = pd.read_csv(TEST_DATASET)
    if 'PHQ8_Binary' not in df.columns:
        raise ValueError("❌ Missing PHQ8_Binary in test dataset!")

    X_test = df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_test = df['PHQ8_Binary']

    print(f"✅ Loaded {len(X_test)} test samples.")
    print(f"   Class distribution: {y_test.value_counts().to_dict()}")

    # Scan for models
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"❌ Model directory not found: {MODEL_DIR}")

    model_files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not model_files:
        print("⚠️ No models found in directory!")
        return

    print(f"🧩 Found {len(model_files)} models:")
    for f in model_files:
        print(f"   - {os.path.basename(f)}")

    results = []

    # Evaluate models
    for model_path in model_files:
        try:
            metrics = evaluate_model(model_path, X_test, y_test)
            results.append(metrics)
        except Exception as e:
            print(f"   ❌ Failed evaluating {model_path}: {e}")

    # Save results
    if results:
        results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
        results_df.to_csv(OUTPUT_RESULTS, index=False)
        print("\n🏁 Evaluation complete! Results saved to:")
        print(f"   📄 {OUTPUT_RESULTS}\n")

        print("🏆 TEST LEADERBOARD 🏆")
        print(results_df[['Model', 'Accuracy', 'Precision', 'F1-Score', 'Recall (Sensitivity)', 'Specificity']].to_string(index=False))
    else:
        print("⚠️ No valid results generated.")

if __name__ == "__main__":
    main()
