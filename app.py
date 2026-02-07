import streamlit as st
import tensorflow as tf
import urllib.request
import os
import numpy as np
import librosa

# 1. Load Model (Already done in your previous step)
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
    
    # --- THIS IS THE MISSING PIECE ---
    with st.spinner("Analyzing audio frequencies..."):
        # 1. Load the audio file
        audio, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast') 
        
        # 2. Extract features (MFCCs)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfccs=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        # 3. Get Prediction from your AI model
        prediction_probabilities = model.predict(mfccs_scaled_features)
        predicted_label_index = np.argmax(prediction_probabilities, axis=1)
        prediction_class = labels[predicted_label_index][0]

    # --- DISPLAY RESULTS ---
    st.header(f"Result: {prediction_class}")
    
    if prediction_class == "Normal":
        st.success("The fan is operating within healthy parameters.")
    else:
        st.error(f"Warning: Potential {prediction_class} detected!")
   

