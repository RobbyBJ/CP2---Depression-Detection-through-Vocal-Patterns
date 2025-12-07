import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# --- IMPORT MODELS ---
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  # <--- NEW IMPORT

# ================= CONFIGURATION =================
INPUT_CSV = r"C:\Users\User\Desktop\CP2\depression_dataset.csv"

# PATHS TO OFFICIAL SPLITS
TRAIN_SPLIT_CSV = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_SPLIT_CSV   = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"

RANDOM_STATE = 42
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def run_baseline_training():
    print("🚀 LOADING SEGMENT DATASET (OFFICIAL SPLITS)...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. LOAD OFFICIAL SPLITS
    try:
        train_split_df = pd.read_csv(TRAIN_SPLIT_CSV)
        dev_split_df = pd.read_csv(DEV_SPLIT_CSV)
        
        train_ids = train_split_df['Participant_ID'].values
        dev_ids = dev_split_df['Participant_ID'].values
        
    except Exception as e:
        print(f"❌ Error loading splits: {e}")
        return

    # Filter Data
    train_df = df[df['participant_id'].isin(train_ids)]
    test_df = df[df['participant_id'].isin(dev_ids)]
    
    # Prepare X and y
    X_train = train_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_train = train_df['PHQ8_Binary']
    
    X_test = test_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
    y_test = test_df['PHQ8_Binary']

    print(f"   Train Segments: {len(X_train)}")
    print(f"   Test (Dev) Segments: {len(X_test)}")

    # --- CALCULATE IMBALANCE FOR XGBOOST ---
    # XGBoost needs 'scale_pos_weight' to handle the imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"   XGBoost Scale Weight: {scale_weight:.2f}")

    # 2. DEFINE BASELINE MODELS
    models_config = {
        'SVM': SVC(random_state=RANDOM_STATE, class_weight='balanced'),
        
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE, 
            class_weight='balanced', 
            n_jobs=-1
        ),
        
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, 
            class_weight='balanced', 
            max_iter=1000
        ),
        
        'KNN': KNeighborsClassifier(n_neighbors=5), 
        
        # --- SWAPPED OUT NAIVE BAYES FOR XGBOOST ---
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=scale_weight, # Handles imbalance
            tree_method='hist',            # Optimized mode
            device='cuda',                 # Uses your GTX 1080 Ti
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE
        )
    }

    results = []
    
    print("\n⚔️ STARTING BASELINE TRAINING...")
    
    for name, model in models_config.items():
        print(f"\n... Training {name} ...")
        
        # Pipeline: Impute -> Scale -> Train
        # Note: XGBoost doesn't strictly need scaling, but it helps when comparing 
        # in a pipeline with other models like SVM/KNN that DO need it.
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler()), 
            ('classifier', model)
        ])
        
        try:
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            spec = calculate_specificity(y_test, y_pred)
            
            # Extract Recall/Precision safely
            report = classification_report(y_test, y_pred, output_dict=True)
            # Handle cases where model predicts only one class
            if '1' in report:
                recall = report['1']['recall'] 
                precision = report['1']['precision']
            else:
                recall = 0.0
                precision = 0.0

            results.append({
                'Model': name,
                'Accuracy': acc,
                'F1-Score': f1,
                'Recall (Sensitivity)': recall,
                'Precision': precision,
                'Specificity': spec
            })
            
            print(f"   -> F1-Score: {f1:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # 3. SAVE RESULTS
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1-Score', ascending=False)
    
    print("\n🏆 BASELINE LEADERBOARD 🏆")
    # Reorder columns for cleaner output
    cols = ['Model', 'Accuracy', 'F1-Score', 'Recall (Sensitivity)', 'Specificity']
    print(results_df[cols].to_string(index=False))
    
    results_df.to_csv("baseline_model_results_official.csv", index=False)
    print("\n✅ Results saved to 'baseline_model_results_official.csv'")

if __name__ == "__main__":
    run_baseline_training()