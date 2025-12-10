import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter
from tqdm import tqdm

# ================= CONFIGURATION =================
# Update these paths to match your folder structure
DATASET_DIR = r"C:\Users\User\Desktop\DAIC-WOZ"
OUT_DIR = r"C:\Users\User\Desktop\processed_augmented"

# Path to labels (Needed to know who to Augment)
TRAIN_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"

# Augmentation Settings
AUGMENT_DEPRESSED_ONLY = True  # Set False to augment everyone (not recommended for balance)
# =================================================

# --- HELPER: Load Labels ---
def load_labels():
    """Loads Train and Dev labels into a single dictionary mapping ID -> Is_Depressed"""
    labels_map = {}
    for path in [TRAIN_LABELS, DEV_LABELS]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure columns match standard format
            if 'Participant_ID' in df.columns:
                df.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
            
            # 1 = Depressed, 0 = Healthy
            for _, row in df.iterrows():
                labels_map[str(int(row['participant_id']))] = row['PHQ8_Binary']
    return labels_map

# --- STEP 1: Bandpass Filter ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300, highcut=3400, fs=16000, order=5):
    """Keeps frequencies between 300Hz and 3400Hz (Human Voice Range)."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# --- STEP 2: Noise Reduction ---
def reduce_noise(y, sr):
    """Reduces background static (fan, AC, hum)."""
    try:
        # stationary=True assumes noise is constant (like a fan)
        # prop_decrease=0.75 is conservative to avoid 'robotic' artifacts
        return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.75)
    except Exception:
        return y

# --- STEP 3: Isolate Participant ---
def isolate_participant_audio(y, sr, transcript_path):
    """Uses transcript to keep only Participant speech."""
    if not os.path.exists(transcript_path):
        return y 

    try:
        df = pd.read_csv(transcript_path, sep='\t')
        
        # Check column names
        if 'speaker' not in df.columns:
            return y
            
        mask = np.zeros_like(y, dtype=bool)
        
        for _, row in df.iterrows():
            if str(row['speaker']).strip() == 'Participant':
                start = int(row['start_time'] * sr)
                stop = int(row['stop_time'] * sr)
                
                # Safety bounds
                start = max(0, start)
                stop = min(len(y), stop)
                
                if start < stop:
                    mask[start:stop] = True
                    
        y_participant = y[mask]
        return y_participant
        
    except Exception as e:
        print(f"❌ Transcript error: {e}")
        return y

# --- STEP 4: Silence Removal ---
def remove_silence(y, sr, top_db=25):
    """Removes dead silence between sentences."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) > 0:
        return np.concatenate([y[start:end] for start, end in intervals])
    return y

# --- STEP 5: Normalization ---
def normalize_audio(y):
    """Peak normalization to -1.0 to 1.0 range."""
    return librosa.util.normalize(y)

# --- STEP 6: Segmentation ---
def segment_audio(y, sr, segment_length=3.0, overlap=0.5):
    """Cuts audio into fixed 3-second chunks with 50% overlap."""
    seg_samples = int(segment_length * sr)
    step = int(seg_samples * (1 - overlap))
    segments = []
    
    if len(y) < seg_samples:
        return []

    for start in range(0, len(y) - seg_samples + 1, step):
        end = start + seg_samples
        segments.append(y[start:end])
        
    return segments

# --- STEP 7: Data Augmentation ---
def augment_audio(y, sr):
    """Generates 2 augmented versions: Pitch Shift & Noise Injection."""
    aug_segments = []
    
    # A. Pitch Shift (Simulate different vocal tract)
    # Conservative shift (+/- 1.5 semitones) to keep emotion intact
    try:
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5)
        aug_segments.append(y_pitch)
    except:
        pass
    
    # B. Add Noise (Simulate poorer recording condition)
    try:
        noise_amp = 0.005 * np.max(np.abs(y))
        y_noise = y + noise_amp * np.random.normal(size=len(y))
        aug_segments.append(y_noise)
    except:
        pass
    
    return aug_segments

# --- MAIN PROCESSING LOOP ---
def preprocess_participant(audio_path, transcript_path, out_dir, pid, is_depressed):
    print(f"Processing Participant {pid} (Depressed: {is_depressed})...")

    try:
        # 1. Load Audio
        y, sr = librosa.load(audio_path, sr=16000)

        # 2. Noise Reduction (Full File)
        y = reduce_noise(y, sr)
        
        # 3. Bandpass Filter (Voice Range)
        y = bandpass_filter(y, fs=sr)

        # 4. Isolate Participant
        y = isolate_participant_audio(y, sr, transcript_path)
        
        if len(y) == 0:
            print(f"⚠️ {pid}: Empty after isolation.")
            return

        # 5. Silence Removal
        y = remove_silence(y, sr)

        # 6. Normalization
        y = normalize_audio(y)

        # 7. Segmentation
        segments = segment_audio(y, sr)

        # 8. Save & Augment
        if segments:
            participant_dir = os.path.join(out_dir, pid)
            os.makedirs(participant_dir, exist_ok=True)
            
            saved_count = 0
            
            for i, seg in enumerate(segments):
                # Save Original
                out_path = os.path.join(participant_dir, f"{pid}_seg{i}.wav")
                sf.write(out_path, seg, sr)
                saved_count += 1
                
                # --- AUGMENTATION LOGIC ---
                # Only augment if participant is Depressed (to fix imbalance)
                if AUGMENT_DEPRESSED_ONLY and is_depressed == 1:
                    augmented_versions = augment_audio(seg, sr)
                    for idx, aug_seg in enumerate(augmented_versions):
                        out_path_aug = os.path.join(participant_dir, f"{pid}_seg{i}_aug{idx}.wav")
                        sf.write(out_path_aug, aug_seg, sr)
                        saved_count += 1
                        
            print(f"✅ Saved {saved_count} segments for {pid}")
        else:
            print(f"⚠️ {pid}: Resulted in 0 segments.")

    except Exception as e:
        print(f"❌ Error processing {pid}: {e}")

# --- BATCH PROCESS ---
def process_dataset():
    print(f"🚀 STARTING ENHANCED PREPROCESSING...")
    print(f"📂 Dataset: {DATASET_DIR}")
    print(f"📂 Output: {OUT_DIR}")
    
    # 1. Load Labels
    labels_map = load_labels()
    print(f"📋 Loaded labels for {len(labels_map)} participants.")

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # 2. Walk Directories
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith("_AUDIO.wav"):
                pid = file.split("_")[0]
                
                # Skip if we don't have a label for this person
                if pid not in labels_map:
                    continue
                    
                is_depressed = labels_map[pid]
                
                audio_path = os.path.join(root, file)
                transcript_file = f"{pid}_TRANSCRIPT.csv"
                transcript_path = os.path.join(root, transcript_file)
                
                preprocess_participant(audio_path, transcript_path, OUT_DIR, pid, is_depressed)
                
    print("\n🎉 Preprocessing & Augmentation Complete!")

if __name__ == "__main__":
    process_dataset()