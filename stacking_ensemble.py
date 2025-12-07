import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, recall_score, confusion_matrix
)
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
INPUT_CSV = r"C:\Users\User\Desktop\CP2\depression_dataset.csv"
TRAIN_SPLIT_CSV = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_SPLIT_CSV   = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"
RANDOM_STATE = 42
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

print("🚀 Loading dataset and splits...")
df = pd.read_csv(INPUT_CSV)
train_ids = pd.read_csv(TRAIN_SPLIT_CSV)['Participant_ID'].values
dev_ids = pd.read_csv(DEV_SPLIT_CSV)['Participant_ID'].values

train_df = df[df['participant_id'].isin(train_ids)]
test_df = df[df['participant_id'].isin(dev_ids)]

X_train = train_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
y_train = train_df['PHQ8_Binary']
X_test = test_df.drop(columns=['PHQ8_Binary', 'participant_id'], errors='ignore')
y_test = test_df['PHQ8_Binary']

print(f"   Train Segments: {len(X_train)}")
print(f"   Test (Dev) Segments: {len(X_test)}")

# ==============================================
# LOAD TUNED MODELS (.pkl)
# ==============================================
print("\n🤝 Loading tuned base models...")

model_files = {
    "SVM": "best_tuned_smote_SVM.pkl",
    "Random Forest": "best_tuned_smote_Random_Forest.pkl",
    "Logistic Regression": "best_tuned_smote_Logistic_Regression.pkl",
    "KNN": "best_tuned_smote_KNN.pkl",
    "XGBoost": "best_tuned_smote_XGBoost.pkl"
}

loaded_models = {}
for name, path in model_files.items():
    try:
        model = joblib.load(path)
        loaded_models[name] = model
        print(f"   ✅ Loaded {name}")
    except Exception as e:
        print(f"   ⚠️ Could not load {name}: {e}")

if len(loaded_models) < 2:
    raise RuntimeError("❌ Not enough models loaded for stacking ensemble.")

# ==============================================
# BUILD STACKING ENSEMBLE (LIGHTGBM META-LEARNER)
# ==============================================
print("\n🧠 Building Stacking Ensemble with LightGBM meta-learner...")

estimators = [(name, model.named_steps['clf']) for name, model in loaded_models.items()]

meta_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    stack_method='predict_proba',
    n_jobs=-1
)

# ==============================================
# PREPROCESS DATA
# ==============================================
print("\n⚙️ Preprocessing datasets (Imputation + Scaling)...")

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_prepared = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_prepared = scaler.transform(imputer.transform(X_test))

# ==============================================
# APPLY SMOTE TO META-TRAINING DATA
# ==============================================
print("\n🧩 Applying SMOTE to balance classes for meta-learner...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train_prepared, y_train)
print(f"   Balanced training samples: {len(X_train_bal)} (0={sum(y_train_bal==0)}, 1={sum(y_train_bal==1)})")

# ==============================================
# TRAIN STACKING ENSEMBLE
# ==============================================
print("\n🏋️‍♂️ Training stacking ensemble...")
stacking_clf.fit(X_train_bal, y_train_bal)

# ==============================================
# EVALUATE MODEL (DEFAULT THRESHOLD 0.5)
# ==============================================
print("\n🔎 Evaluating ensemble on test set (threshold = 0.5)...")

probs = stacking_clf.predict_proba(X_test_prepared)[:, 1]
y_pred = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
spec = calculate_specificity(y_test, y_pred)

print("\n🏆 STACKING ENSEMBLE PERFORMANCE (LGBM Meta-Learner) 🏆")
print(f"   Accuracy:    {acc:.4f}")
print(f"   F1-Score:    {f1:.4f}")
print(f"   Recall:      {recall:.4f}")
print(f"   Specificity: {spec:.4f}")

# ==============================================
# OPTIONAL: FIND BEST THRESHOLD
# ==============================================
print("\n📊 Searching for better threshold (0.3–0.6)...")
for t in [0.3, 0.4, 0.5, 0.6]:
    preds_t = (probs >= t).astype(int)
    acc_t = accuracy_score(y_test, preds_t)
    f1_t = f1_score(y_test, preds_t)
    rec_t = recall_score(y_test, preds_t)
    spec_t = calculate_specificity(y_test, preds_t)
    print(f"   Threshold={t:.1f} | Acc={acc_t:.4f} | F1={f1_t:.4f} | Recall={rec_t:.4f} | Spec={spec_t:.4f}")

# ==============================================
# SAVE FINAL STACKING MODEL
# ==============================================
joblib.dump(stacking_clf, "final_stacking_ensemble.pkl")
print("\n✅ Final Stacking Ensemble saved as 'final_stacking_ensemble_lgbm.pkl'")
