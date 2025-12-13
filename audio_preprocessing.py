import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import noisereduce as nr
import random 
from scipy.signal import butter, lfilter
from tqdm import tqdm

# ================= CONFIGURATION =================
DATASET_DIR = r"C:\Users\User\Desktop\DAIC-WOZ"
OUT_DIR = r"C:\Users\User\Desktop\processed_audio_files"

# Path to the label files
TRAIN_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\train_split_Depression_AVEC2017.csv"
DEV_LABELS = r"C:\Users\User\Desktop\DAIC-WOZ\dev_split_Depression_AVEC2017.csv"

# --- BALANCING SETTINGS (Applies to TRAIN only) ---
MAX_SEGMENTS_PER_PARTICIPANT = 75 # Cap training data to prevent bias
AUGMENT_DEPRESSED_ONLY = True      # Augment minority class in training
# =================================================

# --- HELPER: Load Labels & Split Info ---
def load_labels_and_splits():
    labels_map = {}
    split_map = {} 
    
    # 1. Load Train
    if os.path.exists(TRAIN_LABELS):
        df = pd.read_csv(TRAIN_LABELS)
        if 'Participant_ID' in df.columns: df.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
        for _, row in df.iterrows():
            pid = str(int(row['participant_id']))
            labels_map[pid] = row['PHQ8_Binary']
            split_map[pid] = 'train' 

    # 2. Load Dev
    if os.path.exists(DEV_LABELS):
        df = pd.read_csv(DEV_LABELS)
        if 'Participant_ID' in df.columns: df.rename(columns={'Participant_ID': 'participant_id'}, inplace=True)
        for _, row in df.iterrows():
            pid = str(int(row['participant_id']))
            labels_map[pid] = row['PHQ8_Binary']
            split_map[pid] = 'dev'   

    return labels_map, split_map

# --- STEP 1: Bandpass Filter ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300, highcut=3400, fs=16000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# --- STEP 2: Noise Reduction ---
def reduce_noise(y, sr):
    try:
        return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.75)
    except Exception:
        return y

# --- STEP 3: Isolate Participant ---
def isolate_participant_audio(y, sr, transcript_path):
    if not os.path.exists(transcript_path):
        return y 

    try:
        df = pd.read_csv(transcript_path, sep='\t')
        if 'speaker' not in df.columns:
            return y
            
        mask = np.zeros_like(y, dtype=bool)
        for _, row in df.iterrows():
            if str(row['speaker']).strip() == 'Participant':
                start = int(row['start_time'] * sr)
                stop = int(row['stop_time'] * sr)
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
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) > 0:
        return np.concatenate([y[start:end] for start, end in intervals])
    return y

# --- STEP 5: Normalization ---
def normalize_audio(y):
    return librosa.util.normalize(y)

# --- STEP 6: Segmentation ---
def segment_audio(y, sr, segment_length=5.0, overlap=0.5):
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
    aug_segments = []
    
    # A. Pitch Shift
    try:
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5)
        aug_segments.append(y_pitch)
    except:
        pass
    
    # B. Noise Injection
    try:
        noise_amp = 0.005 * np.max(np.abs(y))
        y_noise = y + noise_amp * np.random.normal(size=len(y))
        aug_segments.append(y_noise)
    except:
        pass
    
    return aug_segments

# --- MAIN PROCESSING LOOP ---
def preprocess_participant(audio_path, transcript_path, out_dir, pid, is_depressed, split):
    split_folder = "train" if split == "train" else "test"
    participant_dir = os.path.join(out_dir, split_folder, pid)
    

    print(f"Processing {pid} ({split.upper()} | Depressed: {is_depressed})...")

    try:
        y, sr = librosa.load(audio_path, sr=16000)
        y = reduce_noise(y, sr)
        y = bandpass_filter(y, fs=sr)
        y = isolate_participant_audio(y, sr, transcript_path)
        
        if len(y) == 0:
            print(f"⚠️ {pid}: Empty audio.")
            return

        y = remove_silence(y, sr)
        y = normalize_audio(y)
        segments = segment_audio(y, sr)

        if segments:
            # ==========================================
            # (TRAIN ONLY) SPLIT LOGIC
            # ==========================================
            
            # 1. TRAINING SET: Apply Cap & Augmentation
            if split == 'train':
                # A. Downsample (Cap) Healthy/Depressed to avoid one speaker dominating
                if len(segments) > MAX_SEGMENTS_PER_PARTICIPANT:
                    selected_indices = sorted(random.sample(range(len(segments)), MAX_SEGMENTS_PER_PARTICIPANT))
                    segments = [segments[i] for i in selected_indices]
                    print(f"   ✂️ Downsampled (Train) to {MAX_SEGMENTS_PER_PARTICIPANT} segments.")
                
                # B. Augment Depressed (ONLY in Train)
                should_augment = (AUGMENT_DEPRESSED_ONLY and is_depressed == 1)

            # 2. Keep all segments for dev/test set
            else:
                # Ensures no capping or augmentation in dev/test
                print(f"   🛡️ Keeping ALL {len(segments)} segments (Test Integrity).")
                should_augment = False

            # ==========================================
            # SAVE SEGMENTS
            # ==========================================
            os.makedirs(participant_dir, exist_ok=True)
            saved_count = 0
            
            for i, seg in enumerate(segments):
                # Save Base Segment
                out_path = os.path.join(participant_dir, f"{pid}_seg{i}.wav")
                sf.write(out_path, seg, sr)
                saved_count += 1
                
                # Apply Augmentation (Only if flagged above)
                if should_augment:
                    augmented_versions = augment_audio(seg, sr)
                    for idx, aug_seg in enumerate(augmented_versions):
                        out_path_aug = os.path.join(participant_dir, f"{pid}_seg{i}_aug{idx}.wav")
                        sf.write(out_path_aug, aug_seg, sr)
                        saved_count += 1
                        
            print(f"   ✅ Saved {saved_count} files for {pid}.")
            
        else:
            print(f"   ⚠️ {pid}: No valid segments after silence removal.")

    except Exception as e:
        print(f"   ❌ Error processing {pid}: {e}")

# --- BATCH PROCESS ---
def process_dataset():
    print(f"🚀 STARTING V2 PREPROCESSING (Split-Aware)...")
    labels_map, split_map = load_labels_and_splits()
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith("_AUDIO.wav"):
                pid = file.split("_")[0]
                
                if pid not in labels_map:
                    continue
                    
                is_depressed = labels_map[pid]
                split = split_map.get(pid, 'unknown') 
                
                audio_path = os.path.join(root, file)
                transcript_file = f"{pid}_TRANSCRIPT.csv"
                transcript_path = os.path.join(root, transcript_file)
                
                preprocess_participant(audio_path, transcript_path, OUT_DIR, pid, is_depressed, split)
                
    print("\n🎉 Preprocessing Complete!")

if __name__ == "__main__":
    process_dataset()