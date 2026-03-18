"""
Data Preprocessing and DataLoader for ISIC Skin Cancer Dataset.

The ISIC dataset is organized in folder-based structure:
    Train/<class_name>/image.jpg
    Test/<class_name>/image.jpg

Converts multi-class labels into binary classification:
    Malignant: melanoma, basal cell carcinoma, actinic keratosis,
               squamous cell carcinoma
    Benign:    nevus, pigmented benign keratosis, seborrheic keratosis,
               dermatofibroma, vascular lesion
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from dataset.data_loader import get_train_transforms, get_val_transforms


# ---------- ISIC Class → Binary Label Mapping ----------

ISIC_MALIGNANT_CLASSES = [
    "melanoma",
    "basal cell carcinoma",
    "actinic keratosis",
    "squamous cell carcinoma",
]

ISIC_BENIGN_CLASSES = [
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "dermatofibroma",
    "vascular lesion",
]


def isic_class_to_binary(class_name):
    """Map ISIC folder name to binary label: 1 = malignant, 0 = benign."""
    if class_name.lower() in [c.lower() for c in ISIC_MALIGNANT_CLASSES]:
        return 1
    return 0


# ---------- Custom Dataset ----------

class ISICDataset(Dataset):
    """PyTorch Dataset for ISIC skin cancer images (folder-based)."""

    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe: pandas DataFrame with columns 'image_path', 'label',
                       'class_name'.
            transform: torchvision transforms to apply.
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------- Build DataFrame from folder structure ----------

def build_isic_dataframe(data_dir):
    """
    Scan the ISIC folder structure and build a DataFrame.

    Args:
        data_dir: path to Train/ or Test/ folder containing class subfolders.

    Returns:
        DataFrame with columns: image_path, class_name, label
    """
    records = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        label = isic_class_to_binary(class_name)

        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "image_path": os.path.join(class_dir, img_file),
                    "class_name": class_name,
                    "label": label,
                })

    df = pd.DataFrame(records)
    return df


# ---------- Helper to build loaders ----------

def prepare_isic_dataloaders(
    isic_base_dir,
    batch_size=32,
    val_split=0.2,
    seed=42,
    img_size=224,
    num_workers=2,
):
    """
    Build train/val DataLoaders for the ISIC dataset.

    Uses the Train/ folder for training data (with a stratified val split).
    The Test/ folder is used as a separate held-out test set.
    """
    train_dir = os.path.join(isic_base_dir, "Train")
    test_dir = os.path.join(isic_base_dir, "Test")

    # Build full dataframe from Train folder
    full_df = build_isic_dataframe(train_dir)

    print(f"\n{'='*50}")
    print(f"ISIC Dataset Summary (Train folder)")
    print(f"{'='*50}")
    print(f"Total images : {len(full_df)}")
    print(f"Malignant (1): {(full_df['label'] == 1).sum()}")
    print(f"Benign    (0): {(full_df['label'] == 0).sum()}")
    print(f"\nClass breakdown:")
    for cls in sorted(full_df['class_name'].unique()):
        n = len(full_df[full_df['class_name'] == cls])
        lbl = "Malignant" if isic_class_to_binary(cls) else "Benign"
        print(f"  {cls}: {n} ({lbl})")

    # Stratified train/val split
    train_df, val_df = train_test_split(
        full_df, test_size=val_split, random_state=seed, stratify=full_df["label"]
    )

    # Build test DataFrame
    test_df = build_isic_dataframe(test_dir)
    print(f"\nISIC Test set: {len(test_df)} images")
    print(f"  Malignant: {(test_df['label'] == 1).sum()}")
    print(f"  Benign:    {(test_df['label'] == 0).sum()}")

    # Create datasets
    train_dataset = ISICDataset(train_df, transform=get_train_transforms(img_size))
    val_dataset = ISICDataset(val_df, transform=get_val_transforms(img_size))
    test_dataset = ISICDataset(test_df, transform=get_val_transforms(img_size))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print(f"Test samples : {len(test_dataset)}")

    return train_loader, val_loader, test_loader, full_df
