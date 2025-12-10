import os
import joblib
import pandas as pd
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ================= CONFIGURATION =================
TRAIN_DATASET = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"
MODEL_SAVE_DIR = r"C:\Users\User\Desktop\CP2\ensemble_models"
RANDOM_STATE = 42
# =================================================

def get_best_params():
    # Use your BEST params from the summary CSV
    xgb_params = {
        'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 100,
        'subsample': 1.0, 'eval_metric': 'logloss', 'tree_method': 'hist'
    }
    rf_params = {
        'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200,
        'class_weight': 'balanced'
    }
    svm_params = {
        'C': 1, 'gamma': 'scale', 'kernel': 'rbf', 'probability': True,
        'class_weight': 'balanced'
    }
    knn_params = {'n_neighbors': 9, 'weights': 'uniform'}
    
    return xgb_params, rf_params, svm_params, knn_params

def main():
    print("🚀 Building Advanced Ensembles (Voting vs. Ridge Stacking)...")
    
    # 1. Load Data
    if not os.path.exists(TRAIN_DATASET):
        print(f"❌ Error: Dataset not found at {TRAIN_DATASET}")
        return

    df = pd.read_csv(TRAIN_DATASET)
    
    if "participant_id" in df.columns:
        groups = df["participant_id"]
        X = df.drop(columns=["PHQ8_Binary", "participant_id", "filename"], errors="ignore")
    else:
        groups = None
        X = df.drop(columns=["PHQ8_Binary", "filename"], errors="ignore")
        
    y = df["PHQ8_Binary"]
    
    xgb_p, rf_p, svm_p, knn_p = get_best_params()

    # 2. Define Base Learners
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

    # --- STRATEGY 1: SOFT VOTING (Averaging) ---
    # This is often more stable than Stacking for small datasets
    print("\n📊 Training Voting Classifier (Soft Vote)...")
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    voting_clf.fit(X, y)
    
    # --- STRATEGY 2: STACKING WITH RIDGE (Better for correlated models) ---
    print("🏗️  Training Stacking Classifier (Ridge Meta-Learner)...")
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=5)
        cv_splits = list(sgkf.split(X, y, groups=groups))
    else:
        cv_splits = 5

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=RidgeClassifier(random_state=RANDOM_STATE), # <--- CHANGED HERE
        cv=cv_splits,
        n_jobs=-1,
        passthrough=False
    )
    stacking_clf.fit(X, y)

    # Save Both
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump(voting_clf, os.path.join(MODEL_SAVE_DIR, "final_voting_soft.pkl"))
    joblib.dump(stacking_clf, os.path.join(MODEL_SAVE_DIR, "final_stacking_ridge.pkl"))
    
    print("\n✅ Saved models:")
    print(f"   1. final_voting_soft.pkl (Averages probabilities)")
    print(f"   2. final_stacking_ridge.pkl (Uses Ridge Regression to judge)")

if __name__ == "__main__":
    main()