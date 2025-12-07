import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# Path to the folder containing your segmented .wav files
PROCESSED_AUDIO_DIR = r"C:\Users\User\Desktop\processed"

# Paths to the label files from DIAC-WOZ folder
TRAIN_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"

# Output file
OUTPUT_CSV = r"C:\Users\User\Desktop\depression_dataset.csv"

def calculate_jitter(f0):
    """Calculates local Jitter (pitch instability)."""
    if len(f0) < 2:
        return 0.0
    return np.mean(np.abs(np.diff(f0))) / np.mean(f0)

def calculate_shimmer(y, sr):
    """Calculates local Shimmer (loudness instability)."""
    rmse = librosa.feature.rms(y=y)[0]
    if len(rmse) < 2 or np.mean(rmse) == 0:
        return 0.0
    return np.mean(np.abs(np.diff(rmse))) / np.mean(rmse)

def extract_segment_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # Skips empty or too short segments
        if len(y) < sr: return None

        # --- 1. PROSODIC FEATURES (Pitch & Voice Quality) ---
        # Using pYIN for pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 0:
            pitch_mean = np.mean(f0_clean)
            pitch_std = np.std(f0_clean) # Micro-prosody (variation within 3 seconds)
            jitter = calculate_jitter(f0_clean)
        else:
            pitch_mean = 0
            pitch_std = 0
            jitter = 0

        shimmer = calculate_shimmer(y, sr)

        # --- 2. SPECTRAL FEATURES ---
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_flux = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # --- 3. ACOUSTIC FEATURES (MFCCs) ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Build Dictionary
        features = {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,     
            "jitter": jitter,          
            "shimmer": shimmer,         
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff,
            "spectral_flux": spectral_flux,
            "zcr": zcr,
        }

        for i in range(13):
            features[f"mfcc_{i+1}"] = mfcc_mean[i]

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    print("🚀 STARTING SEGMENT-LEVEL FEATURE EXTRACTION...")
    
    # 1. LOAD LABELS FIRST
    print("📘 Loading Labels...")
    train_labels = pd.read_csv(TRAIN_LABELS)
    dev_labels = pd.read_csv(DEV_LABELS)
    all_labels = pd.concat([train_labels, dev_labels])
    all_labels['Participant_ID'] = all_labels['Participant_ID'].astype(int)
    
    # Create a quick lookup dictionary for speed: ID -> Binary Class
    # We only care about the Binary Label (0 or 1) for now
    label_map = pd.Series(all_labels.PHQ8_Binary.values, index=all_labels.Participant_ID).to_dict()

    # 2. SCAN & EXTRACT
    data = []
    
    # Walk through the processed folder
    for root, dirs, files in os.walk(PROCESSED_AUDIO_DIR):
        folder_name = os.path.basename(root)
        
        # Skip if not a participant folder
        if not folder_name.isdigit():
            continue
            
        participant_id = int(folder_name)
        
        # Skip participants if we don't have a label for them (e.g. from Test set without labels)
        if participant_id not in label_map:
            continue
            
        target_label = label_map[participant_id]
        wav_files = [f for f in files if f.endswith('.wav')]
        
        print(f"Processing Participant {participant_id} ({len(wav_files)} segments)...")
        
        for wav in tqdm(wav_files, leave=False):
            file_path = os.path.join(root, wav)
            feats = extract_segment_features(file_path)
            
            if feats:
                # Add Metadata directly to the row
                feats['participant_id'] = participant_id
                feats['PHQ8_Binary'] = target_label 
                data.append(feats)

    # 3. SAVE
    df = pd.DataFrame(data)
    print(f"\n Extracted features from {len(df)} segments.")
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f" SUCCESS! Big Dataset saved to: {OUTPUT_CSV}")
    print(f"   Total Samples: {len(df)}")
    print(f"   Class Balance: {df['PHQ8_Binary'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()