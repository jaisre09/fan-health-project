import streamlit as st
import tensorflow as tf
import urllib.request
import os
import numpy as np
import librosa

# Load Model logic (Keep this exactly as it was)
MODEL_URL = "https://github.com/jaisre09/fan-health-project/releases/download/v1.0/fan_failure_model.h5"
MODEL_PATH = "fan_failure_model.h5"

@st.cache_resource
def load_fan_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_fan_model()
labels = np.load("label_map.npy", allow_pickle=True)

st.title("Fan Health Diagnostic System")
uploaded_file = st.file_uploader("Upload fan audio...", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    with st.spinner("Analyzing audio frequencies..."):
        # Fix: Using a more stable loading method
        audio, sample_rate = librosa.load(uploaded_file, sr=None) 
        
        # 1. Ensure audio is a float32 array (librosa requirement)
        audio = audio.astype(np.float32)

        # 2. Extract MFCC features with explicit parameters
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # 3. Scale and reshape for the model
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        mfccs_reshaped = mfccs_scaled.reshape(1, -1)

        # 4. Get Prediction
        with st.spinner("Analyzing audio frequencies..."):
        # Load and convert to float32
        audio, sample_rate = librosa.load(uploaded_file, sr=None) 
        audio = audio.astype(np.float32)

        # Extract 40 MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Calculate the mean across time
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # IMPORTANT: Reshape to (1, 40) so the model sees it as one sample
        mfccs_reshaped = np.array([mfccs_scaled]) 

        # Get Prediction
        prediction_probabilities = model.predict(mfccs_reshaped)
        predicted_index = np.argmax(prediction_probabilities)
        prediction_class = labels[predicted_index]
        
   



