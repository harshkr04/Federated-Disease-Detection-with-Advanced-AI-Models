"""
Data Preprocessing and DataLoader for HAM10000 Skin Lesion Dataset.

Converts multi-class labels into binary classification:
    Malignant: melanoma (mel), bcc, akiec
    Benign:    nv, bkl, df, vasc
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ---------- Label Mapping ----------
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]
BENIGN_CLASSES = ["nv", "bkl", "df", "vasc"]


def label_to_binary(dx):
    """Map diagnosis string to binary label: 1 = malignant, 0 = benign."""
    return 1 if dx in MALIGNANT_CLASSES else 0


# ---------- Transforms ----------
def get_train_transforms(img_size=224):
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size=224):
    """Validation / test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------- Custom Dataset ----------
class SkinLesionDataset(Dataset):
    """PyTorch Dataset for HAM10000 skin lesion images."""

    def __init__(self, dataframe, image_dirs, transform=None):
        """
        Args:
            dataframe: pandas DataFrame with columns 'image_id' and 'label'.
            image_dirs: list of directories that contain the .jpg images.
            transform: torchvision transforms to apply.
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dirs = image_dirs
        self.transform = transform

    def _find_image_path(self, image_id):
        """Search for image_id.jpg inside the given directories."""
        for d in self.image_dirs:
            path = os.path.join(d, f"{image_id}.jpg")
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._find_image_path(row["image_id"])

        if img_path is None:
            raise FileNotFoundError(f"Image {row['image_id']}.jpg not found.")

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------- Helper to build loaders ----------
def prepare_dataloaders(
    metadata_csv,
    image_dirs,
    batch_size=32,
    val_split=0.2,
    seed=42,
    img_size=224,
    num_workers=2,
):
    """
    Read metadata CSV, create binary labels, split into train/val,
    and return DataLoaders.
    """
    df = pd.read_csv(metadata_csv)
    df["label"] = df["dx"].apply(label_to_binary)

    print(f"Total images : {len(df)}")
    print(f"Malignant (1): {(df['label'] == 1).sum()}")
    print(f"Benign    (0): {(df['label'] == 0).sum()}")

    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df["label"]
    )

    train_dataset = SkinLesionDataset(train_df, image_dirs, transform=get_train_transforms(img_size))
    val_dataset = SkinLesionDataset(val_df, image_dirs, transform=get_val_transforms(img_size))

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

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")
    return train_loader, val_loader, df
