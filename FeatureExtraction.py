import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import opensmile

# ============================================================
# 1. ACOUSTIC FEATURES (MFCC, Jitter, Shimmer, ZCR)
# ============================================================

def extract_acoustic_features(y, sr):
    """
    Extract core acoustic features from audio signal.
    Includes MFCCs, Jitter, Shimmer, and Zero-Crossing Rate.
    """
    features = {}

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfccs.shape[0]):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

    # --- Zero-Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # --- Approximate Jitter & Shimmer ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    nonzero_pitches = pitches[pitches > 0]
    if len(nonzero_pitches) > 1:
        pitch_diff = np.diff(nonzero_pitches)
        features['jitter'] = np.mean(np.abs(pitch_diff)) / np.mean(nonzero_pitches)
    else:
        features['jitter'] = 0.0

    frame_energies = np.array([
        np.sum(np.abs(y[i:i+1024])) for i in range(0, len(y), 512)
    ])
    if len(frame_energies) > 1:
        amp_diff = np.diff(frame_energies)
        features['shimmer'] = np.mean(np.abs(amp_diff)) / np.mean(frame_energies)
    else:
        features['shimmer'] = 0.0

    return features


# ============================================================
# 2. PROSODIC FEATURES (Speech Rate, Pause Duration, Intensity)
# ============================================================

def extract_prosodic_features(y, sr):
    """
    Estimate prosodic features such as speech rate, pauses, and intensity.
    """
    features = {}

    # Intensity (RMS energy)
    rms = librosa.feature.rms(y=y)
    features['intensity_mean'] = np.mean(rms)
    features['intensity_std'] = np.std(rms)

    # Speech Rate (approximation using onset detection)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    total_time = librosa.get_duration(y=y, sr=sr)
    features['speech_rate'] = len(onset_frames) / total_time if total_time > 0 else 0

    # Pause Duration (ratio of silence)
    intervals = librosa.effects.split(y, top_db=30)
    voiced_durations = np.sum([(e - s) / sr for s, e in intervals])
    features['pause_ratio'] = 1 - (voiced_durations / total_time)

    return features


# ============================================================
# 3. SPECTRAL FEATURES (Centroid, Rolloff, Flux)
# ============================================================

def extract_spectral_features(y, sr):
    """
    Extract frequency-domain features that describe tone and brightness.
    """
    features = {}

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flux = np.sqrt(np.mean(np.diff(y) ** 2))

    features['spectral_centroid_mean'] = np.mean(centroid)
    features['spectral_rolloff_mean'] = np.mean(rolloff)
    features['spectral_flux'] = flux

    return features


# ============================================================
# 4. VOICE QUALITY FEATURES (OpenSMILE)
# ============================================================

def extract_voice_quality_features(audio_path):
    """
    Use OpenSMILE to extract eGeMAPS low-level descriptors (jitter, shimmer, HNR, etc.)
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    features = smile.process_file(audio_path)
    return features.iloc[0].to_dict()


# ============================================================
# 5. MAIN EXTRACTION PIPELINE
# ============================================================

def extract_features_from_folder(input_folder, output_file):
    """
    Recursively loops through all subfolders, extracts features from every .wav file,
    and saves the result as a single CSV file at the specified output path.
    Includes progress checkpoints per participant folder.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_features = []

    print(f"🎧 Scanning participant folders in: {input_folder}")
    print(f"📁 Output will be saved to: {output_file}\n")

    participant_folders = [os.path.join(input_folder, d) for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

    for participant_folder in participant_folders:
        participant_id = os.path.basename(participant_folder)
        wav_files = [f for f in os.listdir(participant_folder) if f.endswith('.wav')]

        if not wav_files:
            print(f"⚠️ No .wav files found in {participant_id}, skipping...")
            continue

        print(f"\n🎤 Processing participant {participant_id} ({len(wav_files)} files)")

        for file in wav_files:
            file_path = os.path.join(participant_folder, file)
            try:
                y, sr = librosa.load(file_path, sr=None)

                acoustic = extract_acoustic_features(y, sr)
                prosodic = extract_prosodic_features(y, sr)
                spectral = extract_spectral_features(y, sr)

                combined = {
                    'participant_id': participant_id,
                    'filename': file
                }
                combined.update(acoustic)
                combined.update(prosodic)
                combined.update(spectral)

                all_features.append(combined)

            except Exception as e:
                print(f"⚠️ Error processing {file} in {participant_id}: {e}")

        print(f"✅ Finished participant {participant_id}")

    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Feature extraction completed successfully for {len(participant_folders)} participants.")
        print(f"📄 CSV saved at: {output_file}")
    else:
        print("\n⚠️ No .wav files were processed. Please check your directory paths.")


# ============================================================
# 6. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    data_dir = r"C:\\Users\\User\\Desktop\\DAIC-WOZ\\processed"
    output_file = r"C:\Users\User\Desktop\DepressionDetection\features\features.csv"

    extract_features_from_folder(data_dir, output_file)
