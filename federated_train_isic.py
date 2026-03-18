"""
Federated Learning Simulation on ISIC Dataset —
Train with FedAvg, FedProx, or MOON across 3 hospital clients.

Mirrors the HAM10000 `federated_train.py` with identical hyperparameters.
Results are saved separately under weights/isic/ and results/isic/.

Usage:
    python federated_train_isic.py                      # runs all algorithms
    python federated_train_isic.py --algorithm fedavg
    python federated_train_isic.py --algorithm fedprox
    python federated_train_isic.py --algorithm moon
"""

import os
import sys
import copy
import time
import argparse

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.isic_fed_utils import create_isic_hospital_loaders
from models.hybrid_model import build_hybrid_model
from training.trainer import get_device, validate
from federated.fed_utils import federated_average, train_client
from federated.fedprox_utils import train_fedprox_client
from federated.moon_utils import train_moon_client


# ============================================================
# Configuration (same as HAM10000)
# ============================================================

ISIC_BASE_DIR = "Skin cancer ISIC The International Skin Imaging Collaboration"
BATCH_SIZE = 32
IMG_SIZE = 224
FED_ROUNDS = 5
LOCAL_EPOCHS = 2
LR = 1e-4
MU = 0.01          # FedProx
MOON_LAMBDA = 0.5  # MOON
MOON_TEMP = 0.5    # MOON


# ============================================================
# FedAvg training on ISIC
# ============================================================

def run_fedavg_isic(hospital_loaders, val_loader, device,
                    fed_rounds=5, local_epochs=2, lr=1e-4):
    """Run Federated Averaging training on ISIC dataset."""
    print("\n" + "=" * 60)
    print("ISIC — FEDERATED LEARNING — FedAvg")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    os.makedirs("weights/isic", exist_ok=True)
    model_save_path = "weights/isic/fedavg_model.pth"
    weights_save_path = "weights/isic/fedavg_weights.pth"

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
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

        print()

    print(f"FedAvg (ISIC) complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model: {model_save_path}")
    return global_model, best_val_acc


# ============================================================
# FedProx training on ISIC
# ============================================================

def run_fedprox_isic(hospital_loaders, val_loader, device,
                     fed_rounds=5, local_epochs=2, lr=1e-4, mu=0.01):
    """Run FedProx training on ISIC dataset."""
    print("\n" + "=" * 60)
    print("ISIC — FEDERATED LEARNING — FedProx")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    os.makedirs("weights/isic", exist_ok=True)
    model_save_path = "weights/isic/fedprox_model.pth"
    weights_save_path = "weights/isic/fedprox_weights.pth"

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
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

        print()

    print(f"FedProx (ISIC) complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model: {model_save_path}")
    return global_model, best_val_acc


# ============================================================
# MOON training on ISIC
# ============================================================

def run_moon_isic(hospital_loaders, val_loader, device,
                  fed_rounds=5, local_epochs=2, lr=1e-4,
                  lam=0.5, temperature=0.5):
    """Run MOON (Model-Contrastive FL) training on ISIC dataset."""
    print("\n" + "=" * 60)
    print("ISIC — FEDERATED LEARNING — MOON")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    os.makedirs("weights/isic", exist_ok=True)
    model_save_path = "weights/isic/moon_model.pth"
    weights_save_path = "weights/isic/moon_weights.pth"

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
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

        print()

    print(f"MOON (ISIC) complete. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model: {model_save_path}")
    return global_model, best_val_acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning Simulation on ISIC Dataset"
    )
    parser.add_argument(
        "--algorithm", type=str, default="all",
        choices=["fedavg", "fedprox", "moon", "all"],
        help="Which federated algorithm to run (default: all)",
    )
    parser.add_argument("--rounds", type=int, default=FED_ROUNDS,
                        help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS,
                        help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate")
    parser.add_argument("--mu", type=float, default=MU,
                        help="FedProx proximal coefficient")
    parser.add_argument("--moon-lambda", type=float, default=MOON_LAMBDA,
                        help="MOON contrastive loss weight")
    parser.add_argument("--moon-temp", type=float, default=MOON_TEMP,
                        help="MOON temperature")
    args = parser.parse_args()

    # Setup
    device = get_device()

    print("=" * 60)
    print("FEDERATED LEARNING SIMULATION — ISIC DATASET")
    print("=" * 60)

    hospital_loaders, val_loader, _ = create_isic_hospital_loaders(
        isic_base_dir=ISIC_BASE_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
    )

    algorithms = (
        ["fedavg", "fedprox", "moon"] if args.algorithm == "all"
        else [args.algorithm]
    )

    results = {}

    for algo in algorithms:
        if algo == "fedavg":
            model, acc = run_fedavg_isic(
                hospital_loaders, val_loader, device,
                fed_rounds=args.rounds, local_epochs=args.local_epochs,
                lr=args.lr,
            )
            results["FedAvg"] = acc

        elif algo == "fedprox":
            model, acc = run_fedprox_isic(
                hospital_loaders, val_loader, device,
                fed_rounds=args.rounds, local_epochs=args.local_epochs,
                lr=args.lr, mu=args.mu,
            )
            results["FedProx"] = acc

        elif algo == "moon":
            model, acc = run_moon_isic(
                hospital_loaders, val_loader, device,
                fed_rounds=args.rounds, local_epochs=args.local_epochs,
                lr=args.lr, lam=args.moon_lambda,
                temperature=args.moon_temp,
            )
            results["MOON"] = acc

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("FEDERATED TRAINING SUMMARY — ISIC DATASET")
        print("=" * 60)
        for name, acc in results.items():
            print(f"  {name:>10s}: Best Val Acc = {acc:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
