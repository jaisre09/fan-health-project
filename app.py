import streamlit as st
import tensorflow as tf
import urllib.request
import os
import numpy as np

# 1. This is the link to the file you uploaded in image_015787.png
MODEL_URL = "https://github.com/jaisre09/fan-health-project/releases/download/v1.0/fan_failure_model.h5"
MODEL_PATH = "fan_failure_model.h5"

@st.cache_resource
def load_fan_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model... Please wait."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model and labels
model = load_fan_model()
labels = np.load("label_map.npy")

st.title("Fan Health Diagnostic System")
st.write("Upload audio to check fan health.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.success("Analyzing... (This is where your prediction logic goes)")
