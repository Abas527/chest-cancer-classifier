

# 🧠 Chest Cancer Classification (Normal vs. Adenocarcinoma)

An end-to-end Deep Learning project for **chest cancer classification** using PyTorch, deployed with **FastAPI** for API inference, **Streamlit** for interactive UI, and enhanced with **MLflow, DVC, Docker, and CI/CD via GitHub Actions**. Trained and tracked via **DagsHub** with full experiment versioning and model reproducibility.

---

## 📌 Table of Contents

* [🔍 Problem Statement](#-problem-statement)
* [📂 Project Structure](#-project-structure)
* [🧰 Tech Stack](#-tech-stack)
* [🚀 Features](#-features)
* [⚙️ How to Run Locally](#️-how-to-run-locally)
* [🌐 Streamlit UI](#-streamlit-ui)
* [📦 Docker Support](#-docker-support)
* [📁 DVC + DagsHub](#-dvc--dagshub)
* [📊 MLflow Experiment Tracking](#-mlflow-experiment-tracking)
* [🔁 CI/CD via GitHub Actions](#-cicd-via-github-actions)
* [📈 Results](#-results)
* [🧪 Testing](#-testing)
* [🧑‍💻 Author](#-author)

---

## 🔍 Problem Statement

Early detection of **adenocarcinoma** in chest X-rays is vital for treatment. This project trains a deep learning classifier to distinguish between **normal** and **cancerous (adenocarcinoma)** chest X-rays using transfer learning.

---

## 📂 Project Structure

```bash
chest-cancer-classifier/
│
├── data/                      # Data loading, transformed by DVC
├── models/                    # Saved PyTorch models
├── src/
│   ├── data_loader.py         # Custom dataset & transforms
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation logic
│   └── predict.py             # Utility for model inference
│
├── app/
      |-streamlit_app.py       # Streamlit UI
├── Dockerfile                 # Docker image for app
├── requirements.txt
├── dvc.yaml
├── .dvc/                      # DVC config folder
├── mlruns/                    # MLflow logs
├── .github/workflows/ci.yml   # GitHub Actions CI/CD pipeline
└── README.md
```

---

## 🧰 Tech Stack

| Category            | Tools & Frameworks                                       |
| ------------------- | -------------------------------------------------------- |
| Deep Learning       | PyTorch, torchvision                                     |
| Experiment Tracking | MLflow, DagsHub                                          |
| Data Versioning     | DVC                                                      |
| UI Development      | Streamlit                                                |
| DevOps              | Docker, GitHub Actions CI/CD                             |
| Deployment          | Streamlit Community Cloud, Railway, DockerHub (optional) |

---

## 🚀 Features

* ✅ Image classification (Normal vs. Adenocarcinoma)
* ✅ Transfer Learning using pre-trained ResNet
* ✅ Track experiments using MLflow
* ✅ Version control data and models using DVC
* ✅ Interactive Streamlit app for user uploads
* ✅ Docker containerization
* ✅ GitHub Actions CI/CD for reproducibility

---

## ⚙️ How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/chest-cancer-classifier.git
cd chest-cancer-classifier
```

### 2. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### 3. Pull data & model via DVC

```bash
dvc pull
```

---

## 🌐 Streamlit UI

To launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

* Upload a chest X-ray image.
* View the predicted class and confidence.
* Deployed version (if applicable):
  **[🌐 Live Streamlit App](https://chest-cancer-classifier.streamlit.app)**

---


---

## 📦 Docker Support

Build and run the app using Docker:

```bash
# Build image
docker build -t chest-cancer-app .

# Run container
docker run -p 8000:8000 chest-cancer-app
```

Push to DockerHub:

```bash
docker tag chest-cancer-app yourusername/chest-cancer-app
docker push yourusername/chest-cancer-app
```

---

## 📁 DVC + DagsHub

* Uses DVC to version control datasets and models.
* Connected to [DagsHub](https://dagshub.com) for remote data/model storage.
* Add DVC credentials via GitHub Secrets (`DVC_USER`, `DVC_PASSWORD`, `DAGSHUB_TOKEN`).

```bash
dvc repro     # Reproduce entire ML pipeline
dvc pull      # Download latest data & model
```

---

## 📊 MLflow Experiment Tracking

* Integrated with MLflow + DagsHub backend.
* Logs metrics, hyperparameters, and artifacts.
* Automatically logs runs during training.

```bash
mlflow ui
```

---

## 🔁 CI/CD via GitHub Actions

* Automated pipeline on every push to `main`:

  * Installs dependencies
  * Pulls data via DVC
  * Runs training, evaluation
  * Pushes latest model & metrics to DagsHub

Example: `.github/workflows/ci.yml`

---

## 📈 Results

* Accuracy: \~98%
* ROC AUC: 0.99
* Precision/Recall balanced with early stopping
* Live predictions via  Streamlit

---

## 🧪 Testing

* Manual testing via Streamlit & Swagger UI
* Image samples tested from unseen validation set
* Docker container tested on localhost

---

## 🧑‍💻 Author

**👤 Anil khatiwada**
→ GitHub: [@cyberanil27](https://github.com/cyberanil27)
→ Email: [cyberanil27@gmail.com]

---

## 🏁 Final Note

This project is designed using **best MLOps practices** and serves as a great production-ready portfolio project for AI/ML roles.

> Feel free to fork, star, and contribute!

-