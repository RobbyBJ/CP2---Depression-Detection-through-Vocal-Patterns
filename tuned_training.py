import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

# --- CRITICAL IMPORTS FOR SMOTE ---
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # <--- RENAME TO AVOID CONFUSION

# --- IMPORT MODELS ---
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ================= CONFIGURATION =================
INPUT_CSV = r"C:\Users\User\Desktop\CP2\depression_dataset.csv"
TRAIN_SPLIT_CSV = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_SPLIT_CSV   = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"

RANDOM_STATE = 42
N_ITER_SEARCH = 15 
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def tune_model_with_smote(name, model, param_grid, X, y):
    print(f"\n🔎 TUNING {name.upper()} (WITH SMOTE)...")
    start = time.time()
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # --- PIPELINE WITH SMOTE ---
    # 1. Impute missing values
    # 2. Scale features
    # 3. SMOTE (Generates fake depressed samples to balance classes)
    # 4. Classifier
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)), 
        ('clf', model)
    ])
    
    # Prefix params with 'clf__' so the tuner knows they belong to the model step
    grid_prefixed = {f'clf__{k}': v for k, v in param_grid.items()}
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=grid_prefixed,
        n_iter=N_ITER_SEARCH,
        scoring='f1', # Still optimizing for F1
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    search.fit(X, y)
    elapsed = time.time() - start
    
    print(f"   ✅ Best {name} Params: {search.best_params_}")
    print(f"   📊 Best CV F1 Score: {search.best_score_:.4f}")
    print(f"   ⏱️ Tuning took: {elapsed:.1f}s")
    
    return search.best_estimator_

def run_tuning_smote():
    print("🚀 LOADING DATASET (OFFICIAL SPLITS)...")
    df = pd.read_csv(INPUT_CSV)
    
    try:
        train_ids = pd.read_csv(TRAIN_SPLIT_CSV)['Participant_ID'].values
        dev_ids = pd.read_csv(DEV_SPLIT_CSV)['Participant_ID'].values
    except Exception as e:
        print(f"❌ Error loading splits: {e}")
        return

    train_df = df[df['participant_id'].isin(train_ids)]
    test_df = df[df['participant_id'].isin(dev_ids)]
    
    X_train = train_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_train = train_df['PHQ8_Binary']
    
    X_test = test_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_test = test_df['PHQ8_Binary']

    print(f"   Train Segments: {len(X_train)}")
    print(f"   Test (Dev) Segments: {len(X_test)}")

    # Calculate Scale Weight (Just in case we want to try it alongside SMOTE)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # ==========================================
    # DEFINING GRIDS
    # Note: We relax class_weights because SMOTE balances the data already.
    # ==========================================
    
    svm_params = {
        'C': [0.1, 1, 10, 50, 100],
        'gamma': ['scale', 0.1, 0.01],
        'kernel': ['rbf'],
        # SMOTE handles balance, so we usually set class_weight to None, 
        # but we can try both.
        'class_weight': [None, 'balanced'] 
    }

    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }

    lr_params = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }

    knn_params = {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance']
    }

    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        # Since SMOTE balances data, scale_pos_weight should ideally be 1.
        # But we let the tuner decide if it wants "Super Aggressive" (SMOTE + Weight)
        'scale_pos_weight': [1, scale_weight] 
    }

    # ==========================================
    # RUN TUNING
    # ==========================================
    
    models_to_tune = [
        ("SVM", SVC(probability=True, random_state=RANDOM_STATE), svm_params),
        ("Random Forest", RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE), rf_params),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), lr_params),
        ("KNN", KNeighborsClassifier(n_jobs=-1), knn_params),
        ("XGBoost", XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE), xgb_params)
    ]

    results = []
    
    print("\n⚔️ STARTING HYPERPARAMETER TUNING (WITH SMOTE)...")
    
    for name, model, grid in models_to_tune:
        try:
            # 1. Tune
            best_estimator = tune_model_with_smote(name, model, grid, X_train, y_train)
            
            # 2. Evaluate
            y_pred = best_estimator.predict(X_test)
            
            # 3. Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            spec = calculate_specificity(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            recall = report['1']['recall'] if '1' in report else 0.0
            
            results.append({
                'Model': name,
                'Accuracy': acc,
                'F1-Score': f1,
                'Recall': recall,
                'Specificity': spec,
                'Best Params': str(best_estimator.named_steps['clf'].get_params())
            })
            
            # Save the Tuned Model
            joblib.dump(best_estimator, f"best_tuned_smote_{name.replace(' ', '_')}.pkl")
            
        except Exception as e:
            print(f"❌ Failed to tune {name}: {e}")

    # ==========================================
    # FINAL LEADERBOARD
    # ==========================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1-Score', ascending=False)
    
    print("\n🏆 TUNED BASELINE LEADERBOARD (WITH SMOTE) 🏆")
    cols = ['Model', 'Accuracy', 'F1-Score', 'Recall', 'Specificity']
    print(results_df[cols].to_string(index=False))
    
    results_df.to_csv("tuned_baseline_results_smote.csv", index=False)
    print("\n✅ Results saved to 'tuned_baseline_results_smote.csv'")

if __name__ == "__main__":
    run_tuning_smote()