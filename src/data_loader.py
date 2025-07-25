# src/data_loader.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(base_dir, image_size=224, batch_size=32, num_workers=2):
    """
    Load pre-split data from train/val/test directories.
    """

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Adjust for RGB if needed
    ])

    test_val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(base_dir, 'val'), transform=test_val_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=test_val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
