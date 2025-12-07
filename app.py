import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import librosa.display

# ================= CONFIGURATION =================
MODEL_PATH = "best_tuned_smote_Logistic_Regression.pkl" # Update this if needed
SEGMENT_DURATION = 3.0 # Match your training segment length
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
        st.error(f"❌ Model not found: {MODEL_PATH}. Please run training first.")
        return None
    return joblib.load(MODEL_PATH)

def extract_features(y, sr):
    """
    Extracts ONLY the 21 features present in 'depression_dataset.csv'.
    """
    features = {}
    
    # 1. Prosodic Features
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) > 0:
        features['pitch_mean'] = np.mean(f0_clean)
        features['pitch_std'] = np.std(f0_clean)
        features['jitter'] = np.mean(np.abs(np.diff(f0_clean))) / np.mean(f0_clean)
    else:
        features['pitch_mean'] = 0.0
        features['pitch_std'] = 0.0
        features['jitter'] = 0.0

    # Simplified Shimmer (Amplitude variability)
    rmse = librosa.feature.rms(y=y)[0]
    if len(rmse) > 0 and np.mean(rmse) > 0:
        features['shimmer'] = np.mean(np.abs(np.diff(rmse))) / np.mean(rmse)
    else:
        features['shimmer'] = 0.0

    # 2. Spectral Features (Means only)
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Using Spectral Contrast (Band 0) as 'spectral_flux' proxy to match your dataset logic
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_flux'] = np.mean(contrast[0]) 
    
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))

    # 3. MFCCs (1-13 Means only)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = np.mean(mfcc[i])

    return features

def process_and_predict(audio_file, model):
    # Load Audio
    y, sr = librosa.load(audio_file, sr=SR)
    
    # Segment Settings
    seg_samples = int(SEGMENT_DURATION * SR)
    probabilities = []
    
    # Progress Bar
    progress_bar = st.progress(0)
    num_segments = int(np.ceil(len(y) / seg_samples))
    
    # Enforce Column Order (Critical for Scikit-Learn models)
    expected_cols = [
        'pitch_mean', 'pitch_std', 'jitter', 'shimmer', 
        'spectral_centroid', 'spectral_rolloff', 'spectral_flux', 'zcr',
        'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 
        'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13'
    ]

    for i in range(num_segments):
        start = i * seg_samples
        end = min(start + seg_samples, len(y))
        y_seg = y[start:end]
        
        # Pad if short
        if len(y_seg) < seg_samples:
            y_seg = np.pad(y_seg, (0, seg_samples - len(y_seg)))
            
        # Extract & Predict
        feats = extract_features(y_seg, sr)
        
        # Create DataFrame and Reorder Columns
        df_seg = pd.DataFrame([feats])
        df_seg = df_seg[expected_cols]  # <--- CRITICAL FIX
        
        try:
            prob = model.predict_proba(df_seg)[0][1]
            probabilities.append(prob)
        except:
            pred = model.predict(df_seg)[0]
            probabilities.append(float(pred))
            
        progress_bar.progress((i + 1) / num_segments)

    # Aggregate Results
    avg_prob = np.mean(probabilities)
    final_pred = 1 if avg_prob >= 0.5 else 0
    
    return final_pred, avg_prob, y, sr, probabilities

# --- 3. FRONTEND UI ---
st.title("🧠 AI Depression Detector (Voice Analysis)")
st.write("Upload a voice recording (.wav) to screen for potential depressive symptoms.")

model = load_model()

if model:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Audio")
        uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Analyzing voice biomarkers..."):
                    pred, prob, y, sr, segment_probs = process_and_predict(uploaded_file, model)
                
                st.subheader("2. Result")
                confidence = prob * 100 if pred == 1 else (1 - prob) * 100
                
                if pred == 1:
                    st.markdown(f"""
                        <div class="result-box depressed">
                            <h1>DETECTED</h1>
                            <p>Confidence: {confidence:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-box healthy">
                            <h1>HEALTHY</h1>
                            <p>Confidence: {confidence:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

    with col2:
        if uploaded_file is not None and 'y' in locals():
            st.subheader("3. Clinical Insights")
            
            st.markdown("**Waveform & Amplitude**")
            fig, ax = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax)
            st.pyplot(fig)
            
            st.markdown("**Analysis over Time**")
            chart_data = pd.DataFrame({
                "Segment": range(1, len(segment_probs) + 1),
                "Depression Risk": segment_probs
            })
            st.line_chart(chart_data, x="Segment", y="Depression Risk")
            st.caption("Values > 0.5 indicate depressive features.")
            
            st.info("💡 **Why?** The model detected acoustic patterns (Pitch Variability, MFCCs) consistent with the training data.")