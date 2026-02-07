import streamlit as st
import tensorflow as tf
import urllib.request
import os
import numpy as np
import librosa

# 1. Load Model logic
MODEL_URL = "https://github.com/jaisre09/fan-health-project/releases/download/v1.0/fan_failure_model.h5"
MODEL_PATH = "fan_failure_model.h5"

@st.cache_resource
def load_fan_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_fan_model()
labels = np.load("label_map.npy", allow_pickle=True)

# 2. UI Layout
st.title("Fan Health Diagnostic System")
uploaded_file = st.file_uploader("Upload fan audio...", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Analyzing audio frequencies..."):
        # Load audio and ensure it is float32
        audio, sample_rate = librosa.load(uploaded_file, sr=None) 
        audio = audio.astype(np.float32)

        # Extract MFCC features (The 'fingerprint' of the sound)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Calculate the mean
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # FIX: Reshape to 3D (1 sample, 40 features, 1 channel)
        # This resolves the ValueError in image_0d267d.png
        mfccs_reshaped = mfccs_scaled.reshape(1, 40, 1) 

        # Get Prediction
        prediction_probabilities = model.predict(mfccs_reshaped)
        predicted_index = np.argmax(prediction_probabilities)
        prediction_class = labels[predicted_index]

        # 3. Final Output Display
        st.header(f"Diagnostic Result: {prediction_class}")
        if prediction_class == "Normal":
            st.success("The fan is operating normally.")
        else:
            st.error(f"Warning: {prediction_class} Failure Detected!")
        
   





