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
        # Load audio and convert to float32
        audio, sample_rate = librosa.load(uploaded_file, sr=None) 
        audio = audio.astype(np.float32)

        # Extract 40 MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Calculate the mean across time
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Reshape to (1, 40) for the model
        mfccs_reshaped = np.array([mfccs_scaled]) 

        # Get Prediction from model
        prediction_probabilities = model.predict(mfccs_reshaped)
        predicted_index = np.argmax(prediction_probabilities)
        prediction_class = labels[predicted_index]

        # 3. Display Results
        st.header(f"Result: {prediction_class}")
        if prediction_class == "Normal":
            st.success("The fan is operating within healthy parameters.")
        else:
            st.warning(f"Warning: Potential {prediction_class} detected!")
        
   




