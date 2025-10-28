import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

# Optional: if you installed webrtcvad-wheels, you can use VAD for speech detection
try:
    import webrtcvad
    vad_available = True
except ImportError:
    vad_available = False
    print("⚠️ webrtcvad not installed. Skipping voice activity detection.")


# --- Step 1: Silence Removal ---
def remove_silence(y, sr, top_db=30):
    """Removes silent sections using librosa effects."""
    intervals = librosa.effects.split(y, top_db=top_db)
    non_silent = np.concatenate([y[start:end] for start, end in intervals])
    return non_silent


# --- Step 2: Noise Reduction ---
def reduce_noise(y, sr):
    """Reduces background noise using noisereduce."""
    reduced = nr.reduce_noise(y=y, sr=sr)
    return reduced


# --- Step 3: Normalization ---
def normalize_audio(y):
    """Peak normalization."""
    return librosa.util.normalize(y)


# --- Step 4: Segmentation ---
def segment_audio(y, sr, segment_length=3.0, overlap=0.5):
    """Split audio into segments of fixed length (seconds)."""
    seg_samples = int(segment_length * sr)
    step = int(seg_samples * (1 - overlap))
    segments = []
    for start in range(0, len(y) - seg_samples + 1, step):
        end = start + seg_samples
        segments.append(y[start:end])
    return segments


# --- Step 5: Voice Activity Detection ---
def apply_vad(audio_path, sr=16000, frame_duration=30):
    """Keep only voiced speech segments (remove interviewer + silences)."""
    if not vad_available:
        return librosa.load(audio_path, sr=sr)
    import struct
    vad = webrtcvad.Vad(2)
    y, sr = librosa.load(audio_path, sr=sr)
    int16_audio = (y * 32767).astype(np.int16).tobytes()
    frame_size = int(sr * frame_duration / 1000)
    frames = [int16_audio[i:i + frame_size * 2] for i in range(0, len(int16_audio), frame_size * 2)]
    voiced = b"".join([f for f in frames if len(f) == frame_size * 2 and vad.is_speech(f, sr)])
    y_voiced = np.frombuffer(voiced, dtype=np.int16).astype(np.float32) / 32767.0
    return y_voiced, sr


# --- Main Preprocessing Function ---
def preprocess_audio(audio_path, out_dir="processed", pid=""):
    """Applies all preprocessing steps to a single audio file."""
    print(f"🔄 Processing Participant {pid}...")

    # 1. Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # 2. Silence Removal
    y = remove_silence(y, sr)

    # 3. Noise Reduction
    y = reduce_noise(y, sr)

    # 4. Normalization
    y = normalize_audio(y)

    # 5. (Optional) VAD
    # y, sr = apply_vad(audio_path, sr=16000)

    # 6. Segmentation (3s, 50% overlap)
    segments = segment_audio(y, sr, segment_length=3.0, overlap=0.5)

    # 7. Save each segment
    os.makedirs(out_dir, exist_ok=True)
    participant_dir = os.path.join(out_dir, pid)
    os.makedirs(participant_dir, exist_ok=True)
    for i, seg in enumerate(segments):
        out_path = os.path.join(participant_dir, f"{pid}_seg{i}.wav")
        sf.write(out_path, seg, sr)
    print(f"✅ Saved {len(segments)} segments for Participant {pid}")


# --- Batch Process Entire Dataset ---
def process_dataset(dataset_dir, out_dir="processed"):
    """Automatically detects and preprocesses all *_AUDIO.wav files."""
    print(f"🚀 Starting preprocessing for dataset: {dataset_dir}\n")
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith("_AUDIO.wav"):
                pid = file.split("_")[0]  # e.g., '300' from '300_AUDIO.wav'
                audio_path = os.path.join(root, file)
                preprocess_audio(audio_path, out_dir, pid)
    print("\n🎉 Preprocessing complete for all participants!")


# --- Run Automatically ---
if __name__ == "__main__":
    # Change this path to your DAIC-WOZ root folder
    dataset_dir = r"C:\Users\User\Desktop\DIAC-WOZ"
    process_dataset(dataset_dir, out_dir="C:\\Users\\User\\Desktop\\DAIC-WOZ\\processed")
