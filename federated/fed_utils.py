"""
Federated Learning utilities.

Implements:
  - Non-IID dataset splitting for 3 hospitals
  - FedAvg (Federated Averaging) algorithm
  - Local client training
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset.data_loader import (
    SkinLesionDataset,
    get_train_transforms,
    get_val_transforms,
    label_to_binary,
)
from training.trainer import train_one_epoch, validate


# ---------- Non-IID Splitting ----------

def create_non_iid_splits(df, seed=42):
    """
    Split dataset into 3 Non-IID hospital partitions.

    Hospital_A → mostly benign   (80% benign, 20% malignant)
    Hospital_B → mostly malignant (80% malignant, 20% benign)
    Hospital_C → mixed           (50/50 from remaining)
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
    hosp_a = pd.concat([hosp_a_benign, hosp_a_malig]).sample(frac=1, random_state=seed)

    # Hospital B: mostly malignant — 15% of benign, 50% of malignant
    hosp_b_benign = benign_df.iloc[int(n_benign * 0.5) : int(n_benign * 0.65)]
    hosp_b_malig = malignant_df.iloc[int(n_malignant * 0.15) : int(n_malignant * 0.65)]
    hosp_b = pd.concat([hosp_b_benign, hosp_b_malig]).sample(frac=1, random_state=seed)

    # Hospital C: mixed — remaining data
    hosp_c_benign = benign_df.iloc[int(n_benign * 0.65) :]
    hosp_c_malig = malignant_df.iloc[int(n_malignant * 0.65) :]
    hosp_c = pd.concat([hosp_c_benign, hosp_c_malig]).sample(frac=1, random_state=seed)

    splits = {
        "Hospital_A": hosp_a,
        "Hospital_B": hosp_b,
        "Hospital_C": hosp_c,
    }

    for name, split_df in splits.items():
        n_b = (split_df["label"] == 0).sum()
        n_m = (split_df["label"] == 1).sum()
        print(f"  {name}: {len(split_df)} samples (benign={n_b}, malignant={n_m})")

    return splits


# ---------- FedAvg ----------

def federated_average(global_model, client_models):
    """
    Average the weights from all client models (FedAvg).
    Updates global_model in-place.
    """
    global_dict = global_model.state_dict()
    n_clients = len(client_models)

    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client.state_dict()[key].float() for client in client_models], dim=0
        ).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


# ---------- Local Training ----------

def train_client(model, train_loader, device, epochs=1, lr=1e-4):
    """
    Train a local client model for a given number of epochs.
    Returns the updated model.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

    return model, train_loss, train_acc
