# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import build_model
from src.data_loader import get_data_loaders
from src.evaluate import evaluate_model
from src.logger import get_logger
import dagshub

logger = get_logger("train")

def train(num_epochs=10, lr=1e-4, base_dir="data/raw", save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, _, class_names = get_data_loaders(base_dir)

    # Build model
    model = build_model(num_classes=len(class_names), pretrained=True).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info("Training started")

    ### 🔁 MLflow Logging Starts Here ###
    # Ensure experiment exists or create it
    dagshub.init(repo_owner='Abas527', repo_name='chest-cancer-classifier', mlflow=True)

    with mlflow.start_run():
        mlflow.log_param("model_architecture", "resnet18")
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", num_epochs)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)

                # Forward
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}: Training Loss = {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Evaluate on validation set
            val_acc = evaluate_model(model, val_loader, device, class_names, mode="val")
            logger.info(f"Validation Accuracy: {val_acc:.2f}%")
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Save the trained model
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

        # Log the model to MLflow
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()
