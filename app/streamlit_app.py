# app/streamlit_app.py


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.predict import load_model, predict_image

st.set_page_config(page_title="🩻 Chest Cancer Classifier", layout="centered")
st.title("🧠 Chest X-ray Cancer Classifier")

MODEL_PATH = "model.pth"
CLASS_NAMES = ["adenocarcinoma", "normal"]  # Adjust if reversed

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, num_classes=2)

model = get_model()

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    label, confidence = predict_image("temp.jpg", model, CLASS_NAMES)

    st.success(f"Prediction: **{label.upper()}** ({confidence*100:.2f}% confidence)")

    os.remove("temp.jpg")
