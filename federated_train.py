"""
Federated Learning Simulation — Train with FedAvg, FedProx, or MOON
across 3 hospital clients.

Usage:
    python federated_train.py                  # runs all algorithms
    python federated_train.py --algorithm fedavg
    python federated_train.py --algorithm fedprox
    python federated_train.py --algorithm moon
"""

import os
import sys
import copy
import time
import argparse

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
from federated.fedprox_utils import train_fedprox_client
from federated.moon_utils import train_moon_client
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# Shared setup
# ============================================================

def prepare_data(batch_size=32, img_size=224, num_workers=2):
    """Load dataset and create hospital splits + validation set."""
    METADATA_CSV = "HAM10000_metadata.csv"
    IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

    df = pd.read_csv(METADATA_CSV)
    df["label"] = df["dx"].apply(label_to_binary)

    print(f"\nTotal images : {len(df)}")
    print(f"Malignant (1): {(df['label'] == 1).sum()}")
    print(f"Benign    (0): {(df['label'] == 0).sum()}")

    # Non-IID hospital splits
    print("\nCreating Non-IID hospital splits...")
    hospital_splits = create_non_iid_splits(df, seed=42)

    hospital_loaders = {}
    for name, split_df in hospital_splits.items():
        dataset = SkinLesionDataset(
            split_df, IMAGE_DIRS, transform=get_train_transforms(img_size)
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        hospital_loaders[name] = loader

    # Global validation set (20%)
    _, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_dataset = SkinLesionDataset(
        val_df, IMAGE_DIRS, transform=get_val_transforms(img_size)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return hospital_loaders, val_loader


# ============================================================
# FedAvg training
# ============================================================

def run_fedavg(hospital_loaders, val_loader, device,
               fed_rounds=5, local_epochs=2, lr=1e-4):
    """Run Federated Averaging training."""
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING — FedAvg")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0

    model_save_path = "weights/fedavg_model.pth"
    weights_save_path = "weights/fedavg_weights.pth"

    print(f"\nStarting {fed_rounds} federated rounds (FedAvg)...")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Clients: {list(hospital_loaders.keys())}\n")

    for round_num in range(1, fed_rounds + 1):
        round_start = time.time()
        print(f"--- Round {round_num}/{fed_rounds} ---")

        client_models = []

        for hosp_name, hosp_loader in hospital_loaders.items():
            client_model = copy.deepcopy(global_model)
            client_model, loss, acc = train_client(
                client_model, hosp_loader, device,
                epochs=local_epochs, lr=lr,
            )
            print(f"  {hosp_name}: loss={loss:.4f}, acc={acc:.4f}")
            client_models.append(client_model)

        global_model = federated_average(global_model, client_models)

        global_model = global_model.to(device)
        val_loss, val_acc = validate(global_model, val_loader, criterion, device)
        elapsed = time.time() - round_start

        print(f"  Global Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("weights", exist_ok=True)
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

        print()

    print(f"FedAvg complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model: {model_save_path}")
    return global_model, best_val_acc


# ============================================================
# FedProx training
# ============================================================

def run_fedprox(hospital_loaders, val_loader, device,
                fed_rounds=5, local_epochs=2, lr=1e-4, mu=0.01):
    """Run Federated Proximal (FedProx) training."""
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING — FedProx")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0

    model_save_path = "weights/fedprox_model.pth"
    weights_save_path = "weights/fedprox_weights.pth"

    print(f"\nStarting {fed_rounds} federated rounds (FedProx, mu={mu})...")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Clients: {list(hospital_loaders.keys())}\n")

    for round_num in range(1, fed_rounds + 1):
        round_start = time.time()
        print(f"--- Round {round_num}/{fed_rounds} ---")

        client_models = []

        for hosp_name, hosp_loader in hospital_loaders.items():
            client_model = copy.deepcopy(global_model)
            client_model, loss, acc = train_fedprox_client(
                client_model, global_model, hosp_loader, device,
                mu=mu, epochs=local_epochs, lr=lr,
            )
            print(f"  {hosp_name}: loss={loss:.4f}, acc={acc:.4f}")
            client_models.append(client_model)

        global_model = federated_average(global_model, client_models)

        global_model = global_model.to(device)
        val_loss, val_acc = validate(global_model, val_loader, criterion, device)
        elapsed = time.time() - round_start

        print(f"  Global Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("weights", exist_ok=True)
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

        print()

    print(f"FedProx complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model: {model_save_path}")
    return global_model, best_val_acc


# ============================================================
# MOON training
# ============================================================

def run_moon(hospital_loaders, val_loader, device,
             fed_rounds=5, local_epochs=2, lr=1e-4,
             lam=0.5, temperature=0.5):
    """Run MOON (Model-Contrastive Federated Learning) training."""
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING — MOON")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0

    model_save_path = "weights/moon_model.pth"
    weights_save_path = "weights/moon_weights.pth"

    # Track previous local models per client (initialized to global)
    prev_client_models = {
        name: copy.deepcopy(global_model)
        for name in hospital_loaders.keys()
    }

    print(f"\nStarting {fed_rounds} federated rounds (MOON, λ={lam}, τ={temperature})...")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Clients: {list(hospital_loaders.keys())}\n")

    for round_num in range(1, fed_rounds + 1):
        round_start = time.time()
        print(f"--- Round {round_num}/{fed_rounds} ---")

        client_models = []

        for hosp_name, hosp_loader in hospital_loaders.items():
            client_model = copy.deepcopy(global_model)
            previous_model = prev_client_models[hosp_name]

            client_model, loss, acc = train_moon_client(
                client_model, global_model, previous_model,
                hosp_loader, device,
                lam=lam, temperature=temperature,
                epochs=local_epochs, lr=lr,
            )
            print(f"  {hosp_name}: loss={loss:.4f}, acc={acc:.4f}")
            client_models.append(client_model)

            # Store as previous model for next round
            prev_client_models[hosp_name] = copy.deepcopy(client_model)

        global_model = federated_average(global_model, client_models)

        global_model = global_model.to(device)
        val_loss, val_acc = validate(global_model, val_loader, criterion, device)
        elapsed = time.time() - round_start

        print(f"  Global Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("weights", exist_ok=True)
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

        print()

    print(f"MOON complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model: {model_save_path}")
    return global_model, best_val_acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Simulation"
    )
    parser.add_argument(
        "--algorithm", type=str, default="all",
        choices=["fedavg", "fedprox", "moon", "all"],
        help="Which federated algorithm to run (default: all)",
    )
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=2,
                        help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--mu", type=float, default=0.01,
                        help="FedProx proximal coefficient")
    parser.add_argument("--moon-lambda", type=float, default=0.5,
                        help="MOON contrastive loss weight")
    parser.add_argument("--moon-temp", type=float, default=0.5,
                        help="MOON temperature")
    args = parser.parse_args()

    # Setup
    device = get_device()
    hospital_loaders, val_loader = prepare_data()

    algorithms = (
        ["fedavg", "fedprox", "moon"] if args.algorithm == "all"
        else [args.algorithm]
    )

    results = {}

    for algo in algorithms:
        if algo == "fedavg":
            model, acc = run_fedavg(
                hospital_loaders, val_loader, device,
                fed_rounds=args.rounds, local_epochs=args.local_epochs,
                lr=args.lr,
            )
            results["FedAvg"] = acc

        elif algo == "fedprox":
            model, acc = run_fedprox(
                hospital_loaders, val_loader, device,
                fed_rounds=args.rounds, local_epochs=args.local_epochs,
                lr=args.lr, mu=args.mu,
            )
            results["FedProx"] = acc

        elif algo == "moon":
            model, acc = run_moon(
                hospital_loaders, val_loader, device,
                fed_rounds=args.rounds, local_epochs=args.local_epochs,
                lr=args.lr, lam=args.moon_lambda,
                temperature=args.moon_temp,
            )
            results["MOON"] = acc

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("FEDERATED TRAINING SUMMARY")
        print("=" * 60)
        for name, acc in results.items():
            print(f"  {name:>10s}: Best Val Acc = {acc:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
