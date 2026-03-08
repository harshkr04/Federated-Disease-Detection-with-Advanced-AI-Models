"""
Federated Learning Simulation — Train with FedAvg across 3 hospitals.

Usage:
    python federated_train.py
"""

import os
import sys
import copy
import time

import torch
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.data_loader import (
    SkinLesionDataset,
    get_train_transforms,
    get_val_transforms,
    prepare_dataloaders,
    label_to_binary,
)
from models.hybrid_model import build_hybrid_model
from training.trainer import get_device, validate
from federated.fed_utils import (
    create_non_iid_splits,
    federated_average,
    train_client,
)
from torch.utils.data import DataLoader


def main():
    # ---------- Configuration ----------
    METADATA_CSV = "HAM10000_metadata.csv"
    IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    BATCH_SIZE = 32
    FED_ROUNDS = 5         # number of federated rounds
    LOCAL_EPOCHS = 2       # each client trains for 2 epochs per round
    LEARNING_RATE = 1e-4
    IMG_SIZE = 224
    NUM_WORKERS = 2

    MODEL_SAVE_PATH = "weights/federated_global.pth"
    WEIGHTS_SAVE_PATH = "weights/federated_global_weights.pth"

    print("=" * 60)
    print("FEDERATED LEARNING SIMULATION (FedAvg)")
    print("=" * 60)

    # ---------- Load & prepare data ----------
    df = pd.read_csv(METADATA_CSV)
    df["label"] = df["dx"].apply(label_to_binary)

    print(f"\nTotal images : {len(df)}")
    print(f"Malignant (1): {(df['label'] == 1).sum()}")
    print(f"Benign    (0): {(df['label'] == 0).sum()}")

    # Create Non-IID splits for 3 hospitals
    print("\nCreating Non-IID hospital splits...")
    hospital_splits = create_non_iid_splits(df, seed=42)

    # Create DataLoaders for each hospital
    hospital_loaders = {}
    for name, split_df in hospital_splits.items():
        dataset = SkinLesionDataset(
            split_df, IMAGE_DIRS, transform=get_train_transforms(IMG_SIZE)
        )
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        hospital_loaders[name] = loader

    # Create a global validation set (20% of full data)
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_dataset = SkinLesionDataset(
        val_df, IMAGE_DIRS, transform=get_val_transforms(IMG_SIZE)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ---------- Initialize global model ----------
    device = get_device()
    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    print(f"\nGlobal Model: HybridCNNTransformer")
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # ---------- Federated Training ----------
    print(f"\nStarting {FED_ROUNDS} federated rounds...")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Clients: {list(hospital_loaders.keys())}\n")

    for round_num in range(1, FED_ROUNDS + 1):
        round_start = time.time()
        print(f"--- Round {round_num}/{FED_ROUNDS} ---")

        client_models = []

        for hosp_name, hosp_loader in hospital_loaders.items():
            # Copy global model for this client
            client_model = copy.deepcopy(global_model)

            # Train locally
            client_model, loss, acc = train_client(
                client_model, hosp_loader, device,
                epochs=LOCAL_EPOCHS, lr=LEARNING_RATE,
            )
            print(f"  {hosp_name}: loss={loss:.4f}, acc={acc:.4f}")
            client_models.append(client_model)

        # FedAvg — average client weights
        global_model = federated_average(global_model, client_models)

        # Validate global model
        global_model = global_model.to(device)
        val_loss, val_acc = validate(global_model, val_loader, criterion, device)
        elapsed = time.time() - round_start

        print(f"  Global Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model, MODEL_SAVE_PATH)
            torch.save(global_model.state_dict(), WEIGHTS_SAVE_PATH)
            print(f"  ✓ Best global model saved (val_acc={val_acc:.4f})")

        print()

    print(f"Federated training complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Full model : {MODEL_SAVE_PATH}")
    print(f"  Weights    : {WEIGHTS_SAVE_PATH}")


if __name__ == "__main__":
    main()
