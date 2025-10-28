import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===============================================================
# Extract acoustic & prosodic features from preprocessed segments
# ===============================================================

def extract_features(file_path, sr=16000):
    """Extracts relevant features from a single audio file."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = y / np.max(np.abs(y))  # normalize amplitude

        # --- Time-domain features ---
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        energy = np.mean(y ** 2)

        # --- Spectral features ---
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # --- MFCC features ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # --- Approximate jitter and shimmer ---
        # Simple method based on amplitude and pitch variation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[pitches > 0]
        jitter = np.std(np.diff(pitches)) / np.mean(pitches) if len(pitches) > 1 else 0
        shimmer = np.std(magnitudes[magnitudes > 0]) / np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0

        # --- Combine features ---
        features = {
            "zcr": zcr,
            "energy": energy,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_contrast": spectral_contrast,
            "spectral_rolloff": spectral_rolloff,
            "jitter": jitter,
            "shimmer": shimmer,
            "duration_sec": librosa.get_duration(y=y, sr=sr)
        }

        # Add MFCC mean and std
        for i in range(13):
            features[f"mfcc{i+1}_mean"] = mfcc_mean[i]
            features[f"mfcc{i+1}_std"] = mfcc_std[i]

        return features

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None


def process_dataset(audio_dir, output_csv="features.csv"):
    """Iterate through processed audio folders and extract features for all segments."""
    all_features = []
    print(f"🔍 Scanning base directory: {audio_dir}\n")

    for root, _, files in os.walk(audio_dir):
        # Check if the current folder contains wav files
        wav_files = [f for f in files if f.lower().endswith(".wav")]
        if not wav_files:
            continue

        participant_id = os.path.basename(root)
        print(f"🎧 Processing participant {participant_id} ({len(wav_files)} segments)...")

        for file in tqdm(wav_files, desc=f"   Extracting segments", leave=False):
            pid = file.split("_")[0]
            segment_id = file.split("_")[1].replace(".wav", "")
            path = os.path.join(root, file)

            feats = extract_features(path)
            if feats:
                feats["participant_id"] = pid
                feats["segment_id"] = segment_id
                all_features.append(feats)

    # Save results
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Feature extraction completed. Saved to {output_csv}")
    print(f"Total segments processed: {len(df)}")
    return df


if __name__ == "__main__":
    data_dir = r"C:\\Users\\User\\Desktop\\DAIC-WOZ\\processed"
    output_file = r"C:\Users\User\Desktop\DepressionDetection\features\features.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_dataset(data_dir, output_file)
