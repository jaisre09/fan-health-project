import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

# Load your trained model and labels
model = tf.keras.models.load_model("fan_failure_model.h5")
label_map = np.load("label_map.npy", allow_pickle=True).item()

st.title("🛡️ Fan Health Diagnostic System")
st.write("Upload a .wav file to see the Speed, Health, and Failure Type.")

# File Uploader component
uploaded_file = st.file_uploader("Upload Fan Audio", type=["wav"])

if uploaded_file:
    # 1. Processing the audio
    signal, sr = librosa.load(uploaded_file, sr=22050, duration=3)
    
    # 2. Speed (Math Logic)
    energy = np.sqrt(np.mean(signal**2))
    speed = "LOW" if energy < 0.02 else "MEDIUM" if energy < 0.05 else "HIGH"
    
    # 3. Failure Type (CNN Logic)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mfcc, verbose=0)
    failure_type = label_map[np.argmax(prediction)]
    
    # 4. Health Score (Confidence Logic)
    confidence = np.max(prediction) * 100
    health_score = min(100, confidence + 5) if failure_type == "normal" else max(0, 100 - (100 - confidence) - 40)

    # UI Display
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Speed", speed)
    c2.metric("Health Score", f"{int(health_score)}%")
    c3.metric("Failure Type", failure_type.upper())