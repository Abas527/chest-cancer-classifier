# app/streamlit_app.py

import os
import sys
import subprocess
import streamlit as st

# Set up path to access src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import load_model, predict_image

st.set_page_config(page_title="🩻 Chest Cancer Classifier", layout="centered")
st.title("🧠 Chest X-ray Cancer Classifier")

CLASS_NAMES = ["adenocarcinoma", "normal"]

# Step 1: Configure and pull from DVC (with DAGsHub auth)
def setup_dvc():
    if os.path.exists(".dvc") and os.path.exists("model.pth.dvc"):
        os.environ["DAGSHUB_USER"] = st.secrets["DAGSHUB_USER"]
        os.environ["DAGSHUB_TOKEN"] = st.secrets["DAGSHUB_TOKEN"]

        dvc_remote = st.secrets["DVC_REMOTE_URL"]

        # Configure DVC remote
        subprocess.run(["dvc", "remote", "modify", "origin", "url", dvc_remote], check=True)
        subprocess.run(["dvc", "remote", "modify", "origin", "auth", "basic"], check=True)
        subprocess.run(["dvc", "remote", "modify", "origin", "user", os.environ["DAGSHUB_USER"]], check=True)
        subprocess.run(["dvc", "remote", "modify", "origin", "password", os.environ["DAGSHUB_TOKEN"]], check=True)

        # Pull model
        subprocess.run(["dvc", "pull", "model.pth.dvc"], check=True)
    else:
        st.warning("DVC not initialized or model file not tracked.")

# Step 2: Cache the model loading
@st.cache_resource
def get_model():
    setup_dvc()
    return load_model("model.pth", num_classes=2)

model = get_model()

# Step 3: File upload and prediction
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    label, confidence = predict_image("temp.jpg", model, CLASS_NAMES)

    st.success(f"Prediction: **{label.upper()}** ({confidence*100:.2f}% confidence)")

    os.remove("temp.jpg")
