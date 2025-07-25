# template.py

import os

# Define the folder structure
folders = [
    "data/raw",               # Raw input images
    "data/processed",         # Preprocessed images
    "notebooks",              # Jupyter notebooks
    "src",                    # Source code (models, training, etc.)
    "app",                    # Streamlit or frontend code
    "config",                 # Configuration files
    "mlruns",                 # MLflow logs
    ".dvc",                   # DVC cache
]

# Define the base files to be created
files = [
    "src/data_loader.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "app/streamlit_app.py",
    "config/config.yaml",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "dvc.yaml",
]

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")
    
    for file in files:
        # Create the file and write a placeholder comment
        with open(file, "w") as f:
            f.write(f"# {file.split('/')[-1]}\n")
        print(f"Created file: {file}")

    print("\n✅ Project structure created successfully!")

if __name__ == "__main__":
    create_structure()
