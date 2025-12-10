import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from scipy.stats import skew, kurtosis

# ================== CONFIGURATION ==================
PROCESSED_AUDIO_DIR = r"C:\Users\User\Desktop\processed_augmented" 
TRAIN_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"

OUTPUT_TRAIN = r"C:\Users\User\Desktop\CP2\depression_train_dataset.csv"
OUTPUT_DEV = r"C:\Users\User\Desktop\CP2\depression_test_dataset.csv"

MIN_SEGMENTS = 5
MAX_SEGMENTS = 150
# ===================================================

def extract_segment_features(file_path):
    """
    Extracts enriched features: Prosodic + Spectral (Contrast/Flatness) + MFCCs + Higher Order Stats.
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) < sr: return None

        y_pre = librosa.effects.preemphasis(y)

        # --- 1. PROSODIC FEATURES ---
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            pitch_mean = np.mean(f0_clean)
            pitch_std = np.std(f0_clean)
            pitch_skew = skew(f0_clean) # NEW
            pitch_kurt = kurtosis(f0_clean) # NEW
            
            rms = librosa.feature.rms(y=y)[0]
            loudness_mean = np.mean(rms)
            loudness_std = np.std(rms)
            
            jitter = np.mean(np.abs(np.diff(f0_clean))) / pitch_mean
            shimmer = np.mean(np.abs(np.diff(rms))) / loudness_mean
        else:
            pitch_mean = pitch_std = pitch_skew = pitch_kurt = 0.0
            jitter = shimmer = loudness_mean = loudness_std = 0.0

        # --- 2. SPECTRAL FEATURES ---
        # Masking: Only analyze voiced regions
        rms_frames = librosa.feature.rms(y=y)[0]
        voiced_mask = rms_frames > 0.005
        
        def get_stats(feature_vector, prefix):
            # Helper to calculate Mean, Std, Skew, Kurtosis for any vector
            if len(feature_vector) == 0: return {}
            
            # Apply mask if dimensions match
            if feature_vector.shape[-1] == len(voiced_mask):
                if feature_vector.ndim == 1:
                    data = feature_vector[voiced_mask]
                else:
                    data = feature_vector[:, voiced_mask]
            else:
                data = feature_vector # Fallback if sizes differ (rare)

            # If empty after masking, return zeros
            if data.size == 0: 
                return {f"{prefix}_mean": 0, f"{prefix}_std": 0, f"{prefix}_skew": 0, f"{prefix}_kurt": 0}

            # Calculate stats across time axis (axis=-1)
            return {
                f"{prefix}_mean": np.mean(data),
                f"{prefix}_std": np.std(data),
                f"{prefix}_skew": np.mean(skew(data, axis=-1, bias=False)), # Mean skew across bands if 2D
                f"{prefix}_kurt": np.mean(kurtosis(data, axis=-1, bias=False))
            }

        # Basic Spectral
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr       = librosa.feature.zero_crossing_rate(y)[0]
        
        # NEW: Spectral Contrast (Peaks vs Valleys) & Flatness (Noise-like vs Tone-like)
        spec_cont = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        spec_flat = librosa.feature.spectral_flatness(y=y)[0]

        # --- 3. MFCC + DELTAS ---
        mfcc = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # --- 4. BUILD DICTIONARY ---
        features = {
            "pitch_mean": pitch_mean, "pitch_std": pitch_std, 
            "pitch_skew": pitch_skew, "pitch_kurt": pitch_kurt,
            "jitter": jitter, "shimmer": shimmer,
            "loudness_mean": loudness_mean, "loudness_std": loudness_std,
            **get_stats(spec_cent, "spec_cent"),
            **get_stats(spec_roll, "spec_roll"),
            **get_stats(spec_bw, "spec_bw"),
            **get_stats(zcr, "zcr"),
            **get_stats(spec_cont, "spec_contrast"), # NEW
            **get_stats(spec_flat, "spec_flatness"), # NEW
        }

        # Detailed MFCC Stats (The Big Upgrade)
        # Instead of just mean, we get Mean, Std, Skew, Kurt for EACH MFCC coefficient
        if np.any(voiced_mask):
             mfcc_masked = mfcc[:, voiced_mask]
             d1_masked = mfcc_delta[:, voiced_mask]
             d2_masked = mfcc_delta2[:, voiced_mask]
        else:
             mfcc_masked = mfcc
             d1_masked = mfcc_delta
             d2_masked = mfcc_delta2

        for i in range(13):
            # MFCC Static
            features[f"mfcc_{i+1}_mean"] = np.mean(mfcc_masked[i])
            features[f"mfcc_{i+1}_std"]  = np.std(mfcc_masked[i])
            features[f"mfcc_{i+1}_skew"] = skew(mfcc_masked[i])
            features[f"mfcc_{i+1}_kurt"] = kurtosis(mfcc_masked[i])
            
            # MFCC Delta
            features[f"mfcc_d_{i+1}_mean"] = np.mean(d1_masked[i])
            features[f"mfcc_d_{i+1}_std"]  = np.std(d1_masked[i])
            
            # MFCC Delta-Delta
            features[f"mfcc_d2_{i+1}_mean"] = np.mean(d2_masked[i])
            features[f"mfcc_d2_{i+1}_std"]  = np.std(d2_masked[i])

        return features

    except Exception as e:
        return None

def process_split(label_file, output_csv, split_name):
    print(f"\n📘 Processing {split_name} split...")
    
    if not os.path.exists(label_file):
        print(f"❌ Label file not found: {label_file}")
        return

    labels = pd.read_csv(label_file)
    if 'Participant_ID' in labels.columns:
        labels.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
    
    labels['participant_id'] = labels['participant_id'].astype(int)
    label_map = pd.Series(labels.PHQ8_Binary.values, index=labels.participant_id).to_dict()

    data = []
    processed_count = 0

    for root, _, files in os.walk(PROCESSED_AUDIO_DIR):
        folder_name = os.path.basename(root)
        if not folder_name.isdigit(): continue

        pid = int(folder_name)
        if pid not in label_map: continue

        target_label = label_map[pid]
        wav_files = [f for f in files if f.endswith('.wav')]
        n_segments = len(wav_files)

        if n_segments < MIN_SEGMENTS: continue
        if n_segments > MAX_SEGMENTS:
            wav_files = np.random.choice(wav_files, MAX_SEGMENTS, replace=False)

        processed_count += 1
        print(f"➡️ Participant {pid} ({len(wav_files)} segments)")

        for wav in tqdm(wav_files, leave=False):
            feats = extract_segment_features(os.path.join(root, wav))
            if feats:
                feats['participant_id'] = pid
                feats['PHQ8_Binary'] = target_label
                data.append(feats)

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"✅ {split_name} Saved: {len(df)} segments | {df.shape[1]} features")
        return df
    return None

def main():
    print("🚀 STARTING ADVANCED FEATURE EXTRACTION (V2)...")
    process_split(TRAIN_LABELS, OUTPUT_TRAIN, "TRAIN")
    process_split(DEV_LABELS, OUTPUT_DEV, "DEV")
    print("\n🎉 DONE! Feature set V2 generated.")

if __name__ == "__main__":
    main()