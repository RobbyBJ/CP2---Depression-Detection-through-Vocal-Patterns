import os
import joblib
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ================= CONFIGURATION =================
# Update path if necessary for your environment
TRAIN_DATASET = r"C:\Users\User\Desktop\depression_train_handcrafted_v2.csv"
MODEL_SAVE_DIR = r"C:\Users\User\Desktop\CP2\tuned_models_v3"
RANDOM_STATE = 42
# =================================================

def get_best_params():
    """
    Returns the best hyperparameters extracted from 'tuned_model_summary.csv'.
    """
    # 1. XGBoost (Best F1: 0.734)
    # Params: learning_rate=0.05, max_depth=4, n_estimators=100
    xgb_params = {
        'learning_rate': 0.05,
        'max_depth': 4,
        'n_estimators': 100,
        'subsample': 1.0,           # Kept from your config
        'eval_metric': 'logloss',
        'tree_method': 'hist', 
    }
    
    # 2. Random Forest (Best F1: 0.756)
    # Params: max_depth=10, n_estimators=200
    rf_params = {
        'max_depth': 10,
        'min_samples_split': 2,     # Kept default/config
        'n_estimators': 200,        # Increased from 100
        'class_weight': 'balanced'
    }
    
    # 3. SVM (Best F1: 0.707)
    # Params: C=1, kernel='rbf'
    svm_params = {
        'C': 1,                     # Changed from 0.1
        'gamma': 'scale',
        'kernel': 'rbf',            # Changed from 'linear'
        'probability': True,
        'class_weight': 'balanced'
    }

    # 4. KNN (Best F1: 0.586)
    # Params: n_neighbors=9, weights='uniform'
    knn_params = {
        'n_neighbors': 9,           # Changed from 7
        'weights': 'uniform'
    }
    
    return xgb_params, rf_params, svm_params, knn_params

def main():
    print("🚀 Building Stacking Ensemble with Best Tuned Parameters...")
    
    # 1. Load Data
    if not os.path.exists(TRAIN_DATASET):
        print(f"❌ Error: Dataset not found at {TRAIN_DATASET}")
        return

    df = pd.read_csv(TRAIN_DATASET)
    
    # Ensure participant_id exists for safe splitting
    if "participant_id" not in df.columns:
        print("⚠️ Warning: 'participant_id' column missing! Leakage is possible.")
        groups = None
        # Drop only target if ID is missing (and filename if present)
        X = df.drop(columns=["PHQ8_Binary", "filename"], errors="ignore")
    else:
        groups = df["participant_id"]
        # Drop target, ID, and filename
        X = df.drop(columns=["PHQ8_Binary", "participant_id", "filename"], errors="ignore")
        
    y = df["PHQ8_Binary"]

    print(f"✅ Loaded Data: {len(X)} rows | {X.shape[1]} features")

    # 2. Get the Best Params
    xgb_p, rf_p, svm_p, knn_p = get_best_params()

    # 3. Define Base Learners (Level 0)
    # Note: We pass the unpacked dictionary (**xgb_p) into the classifiers
    estimators = [
        ('xgb', ImbPipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', XGBClassifier(**xgb_p, random_state=RANDOM_STATE))
        ])),
        
        ('rf', ImbPipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', RandomForestClassifier(**rf_p, random_state=RANDOM_STATE))
        ])),
        
        ('svm', ImbPipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', SVC(**svm_p, random_state=RANDOM_STATE))
        ])),

        ('knn', ImbPipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', KNeighborsClassifier(**knn_p))
        ]))
    ]

    # 4. Define Meta-Learner (Level 1)
    # Using Logistic Regression with balanced weights as the combiner
    final_estimator = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)

    # ---------------------------------------------------------
    # 🔧 PRE-CALCULATE CV SPLITS
    # ---------------------------------------------------------
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=5)
        # We generate the indices list directly so StackingClassifier doesn't need 'groups' later
        cv_splits = list(sgkf.split(X, y, groups=groups))
    else:
        cv_splits = 5  # Fallback to default if no groups

    # 5. Build the Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv_splits,   # 👈 Pass the pre-calculated list of splits here
        n_jobs=-1,
        passthrough=False 
    )

    # 6. Fit the Stack
    print("⏳ Training Stacking Ensemble (this may take a few minutes)...")
    stacking_clf.fit(X, y)

    # 7. Save
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    save_path = os.path.join(MODEL_SAVE_DIR, "final_stacking_ensemble_v2.pkl")
    joblib.dump(stacking_clf, save_path)
    
    print(f"\n✅ Stacking Ensemble Saved to: {save_path}")
    print("👉 You can now load this .pkl file in your demo app just like any other model.")

if __name__ == "__main__":
    main()