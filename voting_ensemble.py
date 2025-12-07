import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

# ================= CONFIGURATION =================
INPUT_CSV = r"C:\Users\User\Desktop\CP2\depression_dataset.csv"
TRAIN_SPLIT_CSV = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_SPLIT_CSV   = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"
# =================================================

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

print("🚀 Loading datasets...")
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
# LOAD SAVED MODELS
# ==============================================
print("\n🤝 Loading tuned models...")

model_names = ["SVM", "Random_Forest", "Logistic_Regression", "KNN", "XGBoost"]
loaded_models = {}

for name in model_names:
    try:
        model = joblib.load(f"best_tuned_smote_{name}.pkl")
        loaded_models[name] = model
        print(f"   ✅ Loaded {name}")
    except Exception as e:
        print(f"   ⚠️ Could not load {name}: {e}")

if len(loaded_models) < 2:
    raise RuntimeError("❌ Not enough models loaded for ensemble.")

# ==============================================
# BUILD ENSEMBLE
# ==============================================
print("\n🧠 Building Voting Ensemble...")

# Extract classifier step from each pipeline
estimators = [(name, model.named_steps['clf']) for name, model in loaded_models.items()]

ensemble = VotingClassifier(
    estimators=estimators,
    voting='soft'  # use probabilities for smoother combination
)

# Prepare data (impute + scale, no SMOTE)
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_prepared = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_prepared = scaler.transform(imputer.transform(X_test))

# Fit ensemble
ensemble.fit(X_train_prepared, y_train)

# Predict
y_pred = ensemble.predict(X_test_prepared)

# ==============================================
# EVALUATION
# ==============================================
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
spec = calculate_specificity(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
recall = report['1']['recall'] if '1' in report else 0.0

print("\n🏆 ENSEMBLE PERFORMANCE (Voting) 🏆")
print(f"   Accuracy:    {acc:.4f}")
print(f"   F1-Score:    {f1:.4f}")
print(f"   Recall:      {recall:.4f}")
print(f"   Specificity: {spec:.4f}")

joblib.dump(ensemble, "final_ensemble_model.pkl")
print("\n✅ Final Ensemble Model saved as 'final_ensemble_model.pkl'")
