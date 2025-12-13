import os
import pandas as pd
import numpy as np
import opensmile
from tqdm import tqdm

# ================= CONFIGURATION =================
PROCESSED_AUDIO_DIR = r"C:\Users\User\Desktop\processed_audio_files" 

# Labels Paths
TRAIN_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"

# Output Files
OUTPUT_TRAIN = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"
OUTPUT_TEST = r"C:\Users\User\Desktop\CP2\depression_test_dataset.csv"

# Logic Settings
MIN_SEGMENTS = 5  # To skip empty/broken folders
# =================================================

def extract_opensmile_features():
    print("🚀 Initializing OpenSMILE (eGeMAPS)...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return smile

def process_split(smile, label_file, output_csv, split_name):
    print(f"\n📘 Processing {split_name} split...")
    
    if not os.path.exists(label_file):
        print(f"❌ Label file not found: {label_file}")
        return

    # Load Labels
    labels = pd.read_csv(label_file)
    if 'Participant_ID' in labels.columns:
        labels.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    
    labels['participant_id'] = labels['participant_id'].astype(int)
    label_map = pd.Series(labels.PHQ8_Binary.values, index=labels.participant_id).to_dict()

    all_features = []

    # Walk through the entire processed folder (Train and Test subfolders)
    for root, _, files in os.walk(PROCESSED_AUDIO_DIR):
        folder_name = os.path.basename(root)
        
        # Check if folder is a Participant ID (e.g., "300")
        if not folder_name.isdigit(): 
            continue

        pid = int(folder_name)
        
        # CRITICAL FILTER: Only process PIDs that belong to the current CSV (Train or Test)
        if pid not in label_map: 
            continue

        target_label = label_map[pid]
        all_wavs = [f for f in files if f.endswith('.wav')]
        
        # ==========================================
        # 🧠 SELECTION LOGIC
        # ==========================================
        
        if split_name == "TRAIN":
            # For Training: Take EVERYTHING (Base + Augmentations)
            final_files = all_wavs
            
        else: # TEST / DEV
            # For Testing: Take EVERYTHING that isn't augmented
            final_files = [f for f in all_wavs if "_aug" not in f]
        
        # Skip if folder is empty
        if len(final_files) == 0:
            continue

        print(f"➡️ Participant {pid}: Extracting {len(final_files)} segments...")

        # Extract features for each segment
        for wav in tqdm(final_files, leave=False):
            full_path = os.path.join(root, wav)
            try:
                # OpenSMILE extraction
                df_feat = smile.process_file(full_path)
                df_feat = df_feat.reset_index(drop=True)
                
                # Attach Meta
                df_feat['participant_id'] = pid
                df_feat['filename'] = wav
                df_feat['PHQ8_Binary'] = target_label
                
                all_features.append(df_feat)
                
            except Exception:
                pass # Skip broken files

    # Save to CSV
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"✅ {split_name} Saved: {len(final_df)} total segments.")
    else:
        print(f"⚠️ No data extracted for {split_name}. Check paths/labels.")

def main():
    smile = extract_opensmile_features()
    
    # 1. Process Train 
    # Will find files in the processed\training
    process_split(smile, TRAIN_LABELS, OUTPUT_TRAIN, "TRAIN")
    
    # 2. Process Test/Dev
    # Will find files in processed\testing
    process_split(smile, DEV_LABELS, OUTPUT_TEST, "TEST")
    
    print("\n🎉 DONE! Features extracted.")

if __name__ == "__main__":
    main()