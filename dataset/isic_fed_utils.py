"""
Federated Learning utilities for the ISIC dataset.

Implements Non-IID dataset splitting for 3 hospitals using ISIC data.
Reuses FedAvg, FedProx, and MOON from the existing federated/ module.
"""

import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.isic_data_loader import (
    ISICDataset,
    build_isic_dataframe,
    isic_class_to_binary,
)
from dataset.data_loader import get_train_transforms, get_val_transforms


def create_isic_non_iid_splits(df, seed=42):
    """
    Split ISIC dataset into 3 Non-IID hospital partitions.

    Hospital_A → mostly benign   (~80% benign, ~20% malignant)
    Hospital_B → mostly malignant (~80% malignant, ~20% benign)
    Hospital_C → mixed           (remaining data)
    """
    rng = np.random.RandomState(seed)

    benign_df = df[df["label"] == 0].copy()
    malignant_df = df[df["label"] == 1].copy()

    # Shuffle
    benign_df = benign_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    malignant_df = malignant_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_benign = len(benign_df)
    n_malignant = len(malignant_df)

    # Hospital A: mostly benign — 50% of benign, 15% of malignant
    hosp_a_benign = benign_df.iloc[: int(n_benign * 0.5)]
    hosp_a_malig = malignant_df.iloc[: int(n_malignant * 0.15)]
    hosp_a = pd.concat([hosp_a_benign, hosp_a_malig]).sample(
        frac=1, random_state=seed
    )

    # Hospital B: mostly malignant — 15% of benign, 50% of malignant
    hosp_b_benign = benign_df.iloc[int(n_benign * 0.5) : int(n_benign * 0.65)]
    hosp_b_malig = malignant_df.iloc[
        int(n_malignant * 0.15) : int(n_malignant * 0.65)
    ]
    hosp_b = pd.concat([hosp_b_benign, hosp_b_malig]).sample(
        frac=1, random_state=seed
    )

    # Hospital C: mixed — remaining data
    hosp_c_benign = benign_df.iloc[int(n_benign * 0.65) :]
    hosp_c_malig = malignant_df.iloc[int(n_malignant * 0.65) :]
    hosp_c = pd.concat([hosp_c_benign, hosp_c_malig]).sample(
        frac=1, random_state=seed
    )

    splits = {
        "Hospital_A": hosp_a,
        "Hospital_B": hosp_b,
        "Hospital_C": hosp_c,
    }

    print("\nISIC Non-IID Hospital Splits:")
    for name, split_df in splits.items():
        n_b = (split_df["label"] == 0).sum()
        n_m = (split_df["label"] == 1).sum()
        total = len(split_df)
        print(
            f"  {name}: {total} samples "
            f"(benign={n_b} [{n_b/total*100:.1f}%], "
            f"malignant={n_m} [{n_m/total*100:.1f}%])"
        )

    return splits


def create_isic_hospital_loaders(
    isic_base_dir,
    batch_size=32,
    img_size=224,
    num_workers=2,
    seed=42,
):
    """
    Create hospital DataLoaders and a global validation DataLoader
    for the ISIC dataset federated simulation.
    """
    import os
    from sklearn.model_selection import train_test_split

    train_dir = os.path.join(isic_base_dir, "Train")
    full_df = build_isic_dataframe(train_dir)

    print(f"\nISIC Dataset for Federated Training:")
    print(f"  Total images : {len(full_df)}")
    print(f"  Malignant (1): {(full_df['label'] == 1).sum()}")
    print(f"  Benign    (0): {(full_df['label'] == 0).sum()}")

    # Non-IID hospital splits
    hospital_splits = create_isic_non_iid_splits(full_df, seed=seed)

    hospital_loaders = {}
    for name, split_df in hospital_splits.items():
        dataset = ISICDataset(
            split_df, transform=get_train_transforms(img_size)
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        hospital_loaders[name] = loader

    # Global validation set (20% from entire training data)
    _, val_df = train_test_split(
        full_df, test_size=0.2, random_state=seed, stratify=full_df["label"]
    )
    val_dataset = ISICDataset(
        val_df, transform=get_val_transforms(img_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return hospital_loaders, val_loader, full_df
