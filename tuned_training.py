import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ================= CONFIGURATION =================
TRAIN_CSV = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"
MODEL_OUTPUT_DIR = r"C:\Users\User\Desktop\CP2\tuned_models" 
RANDOM_STATE = 42
SENSITIVE_WEIGHTS = {0: 1.0, 1: 2.0}
# =================================================

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def run_tuning():
    print("🚀 LOADING TRAIN DATA (With Participant Groups)...")
    if not os.path.exists(TRAIN_CSV):
        print(f"❌ Error: {TRAIN_CSV} not found.")
        return

    df_train = pd.read_csv(TRAIN_CSV)
    
    # 1. EXTRACT GROUPS (Patient IDs) BEFORE DROPPING THEM
    # This is critical for StratifiedGroupKFold to prevent leakage
    groups = df_train['participant_id'] 
    
    X_train = df_train.drop(columns=['PHQ8_Binary', 'participant_id', 'filename'], errors='ignore')
    y_train = df_train['PHQ8_Binary']

    print(f"✅ Tuning on {len(X_train)} segments.")
    print(f"   Unique Subjects (Groups): {df_train['participant_id'].nunique()}")

    # Calculate Class Balance for XGBoost
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"   Class Balance: {y_train.value_counts().to_dict()}")
    print(f"   XGBoost Scale Weight: {scale_pos_weight:.2f}")

    # ================= DEFINE GRID SEARCH SPACE =================
    # We tune both the Model Hyperparameters AND the number of RFE features
    
    MODEL_PARAMS = {
        'SVM': {
            'model': SVC(probability=True, random_state=RANDOM_STATE, class_weight=SENSITIVE_WEIGHTS),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['rbf'],
                'selector__n_features_to_select': [20, 30, 40]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight=SENSITIVE_WEIGHTS),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_leaf': [2, 4], # Higher leaf count prevents overfitting
                'selector__n_features_to_select': [20, 30, 40]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, class_weight=SENSITIVE_WEIGHTS),
            'params': {
                'classifier__C': [0.01, 0.1, 1, 10],
                'selector__n_features_to_select': [20, 30, 40]
            }
        },
        'KNN': {
        'model': ImbPipeline([
            ('sampler', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ('classifier', KNeighborsClassifier())
        ]),
     
            'params': {
                'classifier__classifier__n_neighbors': [5, 7, 9, 11], 
                'classifier__classifier__weights': ['uniform', 'distance'],
                'selector__n_features_to_select': [20, 30, 40] 
            }
        },
        'XGBoost': {
            'model': XGBClassifier(
                tree_method='hist', eval_metric='logloss', random_state=RANDOM_STATE,
                scale_pos_weight=3.0
            ),
            'params': {
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5],
                'classifier__n_estimators': [100, 200],
                'selector__n_features_to_select': [20, 30, 40]
            }
        }
    }

    print("\n⚔️ STARTING GROUP-AWARE TUNING...")
    print("   (Using StratifiedGroupKFold to prevent leakage)\n")

    # 2. USE STRATIFIED GROUP K-FOLD
    # Splits by PATIENT, not by SEGMENT.
    cv = StratifiedGroupKFold(n_splits=3)

    for name, config in MODEL_PARAMS.items():
        print(f"🧩 Tuning {name}...")
        
        # RFE Step (Selecting features)
        rfe_step = RFE(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), step=5)

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('selector', rfe_step),
            ('classifier', config['model'])
        ])

        # Grid Search
        # scoring='recall' optimizes for SENSITIVITY
        grid = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=cv, 
            scoring='recall', 
            n_jobs=-1, 
            verbose=1
        )

        try:
            # 3. PASS GROUPS TO FIT
            grid.fit(X_train, y_train, groups=groups)
            
            print(f"   🏆 Best Params: {grid.best_params_}")
            print(f"   🏆 Best Recall (Group-CV): {grid.best_score_:.2%}")
            
            # Save Best Model
            save_path = os.path.join(MODEL_OUTPUT_DIR, f"{name}_tuned_v2.pkl")
            joblib.dump(grid.best_estimator_, save_path)
            print(f"   ✅ Saved: {save_path}\n")

        except Exception as e:
            print(f"   ❌ Failed {name}: {e}\n")

    print(f"🎉 Tuning Complete. Models saved to: {MODEL_OUTPUT_DIR}")
    print("👉 Update your 'test_v9_soft_voting.py' to point to this new folder!")

if __name__ == "__main__":
    run_tuning()