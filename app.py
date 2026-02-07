import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import os

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40

# =========================
# LOAD MODEL & LABELS
# =========================
model = tf.keras.models.load_model("fan_failure_model.h5")
label_map = np.load("label_map.npy", allow_pickle=True).item()

# =========================
# FUNCTIONS
# =========================
def extract_mfcc(file_path):
    signal, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T  # (time_steps, 40)

    # 🔑 THIS IS THE MOST IMPORTANT LINE
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    # Final shape: (1, time_steps, 40, 1)

    return mfcc


def get_speed_level(signal):
    rms = np.mean(librosa.feature.rms(y=signal))
    if rms < 0.02:
        return "LOW"
    elif rms < 0.05:
        return "MEDIUM"
    else:
        return "HIGH"


def get_health_score(pred_confidence):
    return int(pred_confidence * 100)


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Fan Health Diagnostic System")

st.title("🌀 Fan Health Diagnostic System")

uploaded_file = st.file_uploader(
    "Upload fan audio",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Display audio
    st.audio(uploaded_file)

    # Load signal again for speed calc
    signal, sr = librosa.load(
        temp_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    # Extract MFCC
    mfcc_input = extract_mfcc(temp_path)

    # Predict
    predictions = model.predict(mfcc_input)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    failure_type = label_map[predicted_index]
    speed_level = get_speed_level(signal)
    health_score = get_health_score(confidence)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("🔍 Diagnosis Result")

    st.write(f"**Failure Type:** {failure_type}")
    st.write(f"**Speed Level:** {speed_level}")
    st.write(f"**Health Score:** {health_score}%")

    # Cleanup
    os.remove(temp_path)

        
   






