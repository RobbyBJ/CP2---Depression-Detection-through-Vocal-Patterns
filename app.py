import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import librosa.display
from scipy.stats import skew, kurtosis

# ================= CONFIGURATION =================
# 1. Point to your Best Stacking Model
MODEL_PATH = "ensemble_models\stacking_ridge.pkl" 

# 2. Sensitivity Threshold (Optimized for 0.35)
THRESHOLD = 0.35 

# 3. Audio Settings
SEGMENT_DURATION = 3.0 
SR = 16000
# =================================================

# --- 1. SETUP & STYLE ---
st.set_page_config(page_title="Depression Detection AI", layout="wide")
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; color: white;}
    .depressed { background-color: #d32f2f; }
    .healthy { background-color: #388e3c; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND FUNCTIONS ---

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found: {MODEL_PATH}. Please make sure the .pkl file is in the same folder.")
        return None
    return joblib.load(MODEL_PATH)

def extract_features_v2(y, sr):
    """
    Extracts the V2 Advanced Features (130+ features).
    Matches 'extract_handcrafted_features_v2.py' exactly.
    """
    try:
        y_pre = librosa.effects.preemphasis(y)

        # --- 1. PROSODIC FEATURES ---
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            pitch_mean = np.mean(f0_clean)
            pitch_std = np.std(f0_clean)
            pitch_skew = skew(f0_clean)
            pitch_kurt = kurtosis(f0_clean)
            
            rms = librosa.feature.rms(y=y)[0]
            loudness_mean = np.mean(rms)
            loudness_std = np.std(rms)
            
            jitter = np.mean(np.abs(np.diff(f0_clean))) / pitch_mean
            shimmer = np.mean(np.abs(np.diff(rms))) / loudness_mean
        else:
            pitch_mean = pitch_std = pitch_skew = pitch_kurt = 0.0
            jitter = shimmer = loudness_mean = loudness_std = 0.0

        # --- 2. SPECTRAL FEATURES ---
        rms_frames = librosa.feature.rms(y=y)[0]
        voiced_mask = rms_frames > 0.005
        
        def get_stats(feature_vector, prefix):
            if len(feature_vector) == 0: return {}
            
            if feature_vector.shape[-1] == len(voiced_mask):
                if feature_vector.ndim == 1:
                    data = feature_vector[voiced_mask]
                else:
                    data = feature_vector[:, voiced_mask]
            else:
                data = feature_vector

            if data.size == 0: 
                return {f"{prefix}_mean": 0, f"{prefix}_std": 0, f"{prefix}_skew": 0, f"{prefix}_kurt": 0}

            return {
                f"{prefix}_mean": np.mean(data),
                f"{prefix}_std": np.std(data),
                f"{prefix}_skew": np.mean(skew(data, axis=-1, bias=False)),
                f"{prefix}_kurt": np.mean(kurtosis(data, axis=-1, bias=False))
            }

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr       = librosa.feature.zero_crossing_rate(y)[0]
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
            **get_stats(spec_cont, "spec_contrast"),
            **get_stats(spec_flat, "spec_flatness"),
        }

        # MFCC Stats
        if np.any(voiced_mask):
             mfcc_masked = mfcc[:, voiced_mask]
             d1_masked = mfcc_delta[:, voiced_mask]
             d2_masked = mfcc_delta2[:, voiced_mask]
        else:
             mfcc_masked = mfcc
             d1_masked = mfcc_delta
             d2_masked = mfcc_delta2

        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = np.mean(mfcc_masked[i])
            features[f"mfcc_{i+1}_std"]  = np.std(mfcc_masked[i])
            features[f"mfcc_{i+1}_skew"] = skew(mfcc_masked[i])
            features[f"mfcc_{i+1}_kurt"] = kurtosis(mfcc_masked[i])
            
            features[f"mfcc_d_{i+1}_mean"] = np.mean(d1_masked[i])
            features[f"mfcc_d_{i+1}_std"]  = np.std(d1_masked[i])
            
            features[f"mfcc_d2_{i+1}_mean"] = np.mean(d2_masked[i])
            features[f"mfcc_d2_{i+1}_std"]  = np.std(d2_masked[i])

        return features

    except Exception as e:
        return None

def process_and_predict(audio_file, model):
    # Load Audio
    y, sr = librosa.load(audio_file, sr=SR)
    
    # Segment Settings
    seg_samples = int(SEGMENT_DURATION * SR)
    probabilities = []
    
    # Progress Bar
    progress_bar = st.progress(0)
    num_segments = int(np.ceil(len(y) / seg_samples))
    
    # Store valid segments
    valid_features = []

    for i in range(num_segments):
        start = i * seg_samples
        end = min(start + seg_samples, len(y))
        y_seg = y[start:end]
        
        # Pad if short
        if len(y_seg) < seg_samples:
            y_seg = np.pad(y_seg, (0, seg_samples - len(y_seg)))
            
        # Extract V2 Features
        feats = extract_features_v2(y_seg, sr)
        
        if feats:
            valid_features.append(feats)
        
        progress_bar.progress((i + 1) / num_segments)
    
    if not valid_features:
        return 0, 0.0, y, sr, []

    # Create DataFrame
    df_seg = pd.DataFrame(valid_features)
    
    # --- GET PREDICTIONS (Stacking/Ridge Logic) ---
    # Ridge Classifier (Stacking Meta) uses decision_function, not predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_seg)[:, 1]
    else:
        # Normalize decision function to 0-1 for visualization
        scores = model.decision_function(df_seg)
        probs = 1 / (1 + np.exp(-scores)) # Sigmoid to convert score to probability
            
    # Aggregate Results (Average Probability across all segments)
    avg_prob = np.mean(probs)
    
    # --- APPLY THRESHOLD LOGIC ---
    final_pred = 1 if avg_prob >= THRESHOLD else 0
    
    return final_pred, avg_prob, y, sr, probs

# --- 3. FRONTEND UI ---
st.title("🧠 AI Depression Detector")
st.write("Upload a voice recording (.wav) to screen for potential depressive symptoms.")
st.caption(f"Using Advanced Feature Extraction • Sensitivity Threshold: {THRESHOLD}")

model = load_model()

if model:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Audio")
        uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Analyzing spectral texture & monotonicity..."):
                    pred, prob, y, sr, segment_probs = process_and_predict(uploaded_file, model)
                
                st.subheader("2. Result")
                
                # Dynamic Confidence Display based on Threshold
                if pred == 1:
                    # Scale confidence relative to threshold for display
                    display_conf = min(prob * 100, 99.9)
                    st.markdown(f"""
                        <div class="result-box depressed">
                            <h1>AT RISK</h1>
                            <p>Depression Signal Detected</p>
                            <p>Confidence Score: {display_conf:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.warning("⚠️ The model detected features consistent with flat affect and monotonicity.")
                else:
                    display_conf = (1 - prob) * 100
                    st.markdown(f"""
                        <div class="result-box healthy">
                            <h1>HEALTHY</h1>
                            <p>No Significant Indicators</p>
                            <p>Healthy Confidence: {display_conf:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ Voice patterns appear within standard ranges.")

    with col2:
        if uploaded_file is not None and 'y' in locals():
            st.subheader("3. Clinical Insights")
            
            st.markdown("**Waveform Analysis**")
            fig, ax = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax)
            st.pyplot(fig)
            
            st.markdown("**Risk Timeline (Per Segment)**")
            if len(segment_probs) > 0:
                chart_data = pd.DataFrame({
                    "Segment (3s)": range(1, len(segment_probs) + 1),
                    "Risk Score": segment_probs
                })
                st.line_chart(chart_data, x="Segment (3s)", y="Risk Score")
                st.caption(f"Values above {THRESHOLD} contribute to an 'At Risk' classification.")
            
            with st.expander("ℹ️ How does this work?"):
                st.write("""
                **Model:** Stacking Ensemble (Ridge Meta-Learner)
                **Features:** 130+ Acoustic Biomarkers
                **Key Indicators:**
                * **Monotonicity:** Low Standard Deviation in Pitch.
                * **Spectral Flatness:** Breathy/Weak voice quality.
                * **Jitter/Shimmer:** Micro-tremors in vocal cords.
                """)