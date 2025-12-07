import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import noisereduce as nr

# --- Step 0: Isolate Participant ---
def isolate_participant_audio(y, sr, transcript_path):
    """
    Uses the transcript CSV to isolate Participant audio.
    Replaces 'Ellie' (Interviewer) audio with silence or removes it.
    """
    if not os.path.exists(transcript_path):
        print(f"Warning: Transcript not found at {transcript_path}")
        return y # Return original if no transcript (risky)

    df = pd.read_csv(transcript_path, sep='\t') # DAIC-WOZ transcripts are usually tab-separated
    
    # Create a mask of zeros (silence)
    mask = np.zeros_like(y, dtype=bool)
    
    # Check column names (adjust if your CSV headers differ)
    # Usually: 'start_time', 'stop_time', 'speaker'
    for _, row in df.iterrows():
        if row['speaker'] == 'Participant':
            start_sample = int(row['start_time'] * sr)
            stop_sample = int(row['stop_time'] * sr)
            
            # Ensure we don't go out of bounds
            if start_sample < len(y):
                stop_sample = min(stop_sample, len(y))
                mask[start_sample:stop_sample] = True
                
    # Keep only the marked samples (Concatenate them to remove gaps, 
    # OR just zero them out if you want to keep timing - usually concatenation is better for features)
    y_participant = y[mask] 
    
    return y_participant

# --- Step 1: Silence Removal ---
def remove_silence(y, sr, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)
    non_silent = np.concatenate([y[start:end] for start, end in intervals]) if len(intervals) > 0 else y
    return non_silent

# --- Step 2: Noise Reduction ---
def reduce_noise(y, sr):
    # Only perform if audio is long enough
    if len(y) > sr: 
        return nr.reduce_noise(y=y, sr=sr)
    return y

# --- Step 3: Normalization ---
def normalize_audio(y):
    return librosa.util.normalize(y)

# --- Step 4: Segmentation ---
def segment_audio(y, sr, segment_length=3.0, overlap=0.5):
    seg_samples = int(segment_length * sr)
    step = int(seg_samples * (1 - overlap))
    segments = []
    
    # Pad if shorter than segment length
    if len(y) < seg_samples:
        return []

    for start in range(0, len(y) - seg_samples + 1, step):
        end = start + seg_samples
        segments.append(y[start:end])
    return segments

# --- Main Processing ---
def preprocess_audio(audio_path, transcript_path, out_dir, pid):
    print(f"Processing Participant {pid}...")

    try:
        # 1. Load Audio
        y, sr = librosa.load(audio_path, sr=16000)

        # 2. Isolate Participant
        y = isolate_participant_audio(y, sr, transcript_path)
        
        if len(y) == 0:
            print(f"⚠️ Participant {pid} has no audio data after isolation.")
            return

        # 3. Noise Reduction
        y = reduce_noise(y, sr)

        # 4. Silence Removal (Remove the gaps left between sentences)
        y = remove_silence(y, sr)

        # 5. Normalization
        y = normalize_audio(y)

        # 6. Segmentation
        segments = segment_audio(y, sr)

        # 7. Save
        if segments:
            participant_dir = os.path.join(out_dir, pid)
            os.makedirs(participant_dir, exist_ok=True)
            for i, seg in enumerate(segments):
                out_path = os.path.join(participant_dir, f"{pid}_seg{i}.wav")
                sf.write(out_path, seg, sr)
            print(f"Saved {len(segments)} segments for Participant {pid}")
        else:
            print(f"Participant {pid} resulted in 0 segments.")

    except Exception as e:
        print(f"Error processing {pid}: {e}")

# --- Batch Process ---
def process_dataset(dataset_dir, out_dir):
    print(f"Starting preprocessing for dataset: {dataset_dir}\n")
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("_AUDIO.wav"):
                pid = file.split("_")[0]
                audio_path = os.path.join(root, file)
                
                # Construct Transcript Path
                # Usually in same folder: "300_TRANSCRIPT.csv"
                transcript_file = f"{pid}_TRANSCRIPT.csv"
                transcript_path = os.path.join(root, transcript_file)
                
                preprocess_audio(audio_path, transcript_path, out_dir, pid)
                
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    # Update paths
    DATASET_DIR = r"C:\Users\User\Desktop\DAIC-WOZ"
    OUT_DIR = r"C:\Users\User\Desktop\processed"
    process_dataset(DATASET_DIR, OUT_DIR)