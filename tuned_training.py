import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA 
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, make_scorer

# --- MODELS ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ================= CONFIGURATION =================
TRAIN_DATASET = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"
MODEL_SAVE_DIR = r"C:\Users\User\Desktop\CP2\tuned_models"
RANDOM_STATE = 42
N_JOBS = -1
# =================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("🚀 STARTING HYPERPARAMETER TUNING PIPELINE (Wav2Vec + PCA Edition)...")
    ensure_dir(MODEL_SAVE_DIR)

    if not os.path.exists(TRAIN_DATASET):
        print(f"❌ Error: Dataset not found at {TRAIN_DATASET}")
        return

    # 1️⃣ Load dataset
    df = pd.read_csv(TRAIN_DATASET)
    
    # CRITICAL: Extract Groups before dropping columns
    if "participant_id" in df.columns:
        groups = df["participant_id"]
    else:
        print("⚠️ Warning: 'participant_id' missing. GroupKFold cannot be used (Leakage Risk!).")
        groups = None

    # Drop non-feature columns
    X = df.drop(columns=["PHQ8_Binary", "participant_id", "filename"], errors="ignore")
    y = df["PHQ8_Binary"]

    print(f"✅ Loaded training dataset: {len(X)} samples")
    print(f"   Features: {X.shape[1]} (Should be ~768 for Wav2Vec)")
    print(f"   Class distribution: {y.value_counts().to_dict()}")

    # 2️⃣ Define models and their grids
    param_grids = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1],
            "classifier__solver": ["lbfgs"]
        },
        "SVM": {
            # PCA makes SVM fast enough to try both kernels again
            "classifier__C": [1, 10], 
            "classifier__kernel": ["rbf"] 
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [4, 6],
        },
        "KNN": {
            "classifier__n_neighbors": [5, 9],
            "classifier__weights": ["uniform"]
        }
    }

    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            tree_method="hist",
            eval_metric="logloss",
        )
    }

    results = []

    # 3️⃣ Perform tuning for each model
    for name, model in base_models.items():
        print(f"\n🔍 Tuning {name}...")

        # --- BUILD PIPELINE DYNAMICALLY ---
        # 1. Base steps (Imputer -> Scaler -> SMOTE)
        steps = [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
        ]
        
        # 2. OPTIMIZATION: Add PCA only for slow models (SVM & KNN)
        if name in ["SVM", "KNN"]:
            print(f"   👉 Adding PCA (0.95 variance) to speed up {name}...")
            steps.append(("pca", PCA(n_components=0.90))) 

        # 3. Add the classifier
        steps.append(("classifier", model))

        pipeline = ImbPipeline(steps)

        # 4. Stratified Group K-Fold (Prevents Leakage)
        cv_strategy = StratifiedGroupKFold(n_splits=5)

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids.get(name, {}),
            scoring=make_scorer(f1_score),
            n_jobs=N_JOBS,
            cv=cv_strategy, 
            verbose=2
        )

        # 5. Fit with Groups
        if groups is not None:
            grid.fit(X, y, groups=groups)
        else:
            grid.fit(X, y)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_

        print(f"✅ Best F1-score: {best_score:.4f}")
        print(f"🏆 Best Parameters: {best_params}")

        # Save tuned model
        save_path = os.path.join(MODEL_SAVE_DIR, f"tuned_{name.replace(' ', '_')}.pkl")
        joblib.dump(best_model, save_path)
        print(f"💾 Saved model to: {save_path}")

        results.append({
            "Model": name,
            "Best F1-Score": best_score,
            "Best Params": best_params
        })

    # 4️⃣ Save all results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Best F1-Score", ascending=False)
    results_df.to_csv(os.path.join(MODEL_SAVE_DIR, "tuned_model_summary.csv"), index=False)

    print("\n🏁 All models tuned and saved successfully!")
    print(results_df)

if __name__ == "__main__":
    main()