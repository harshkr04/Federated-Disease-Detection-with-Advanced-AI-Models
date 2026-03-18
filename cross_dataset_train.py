"""
Cross-Dataset Validation: Train and Evaluate All Models on the ISIC Dataset.

Trains the same 5 models used for HAM10000 on the ISIC dataset:
  1. Centralized CNN (ResNet50)
  2. Centralized Hybrid CNN-Transformer
  3. Federated FedAvg Hybrid
  4. Federated FedProx Hybrid
  5. Federated MOON Hybrid

Produces:
  - Trained weights in weights/isic/
  - Metrics, confusion matrices, ROC curves in results/isic/
  - Cross-dataset comparison tables and charts in results/cross_dataset/

Usage:
    python cross_dataset_train.py                  # run everything
    python cross_dataset_train.py --skip-training  # only generate results
    python cross_dataset_train.py --model cnn      # train only CNN
    python cross_dataset_train.py --model hybrid    # train only Hybrid
    python cross_dataset_train.py --model fedavg    # train only FedAvg
    python cross_dataset_train.py --model fedprox   # train only FedProx
    python cross_dataset_train.py --model moon      # train only MOON
"""

import os
import sys
import copy
import time
import json
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.isic_data_loader import prepare_isic_dataloaders
from dataset.isic_fed_utils import create_isic_hospital_loaders
from models.cnn_model import build_cnn_model
from models.hybrid_model import build_hybrid_model
from training.trainer import get_device, train_model, validate
from federated.fed_utils import federated_average, train_client
from federated.fedprox_utils import train_fedprox_client
from federated.moon_utils import train_moon_client


# ============================================================
# Configuration
# ============================================================

ISIC_BASE_DIR = "Skin cancer ISIC The International Skin Imaging Collaboration"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
FED_ROUNDS = 5
LOCAL_EPOCHS = 2
LR = 1e-4
MU = 0.01          # FedProx
MOON_LAMBDA = 0.5  # MOON
MOON_TEMP = 0.5    # MOON


# ============================================================
# Helpers
# ============================================================

def get_predictions(model, dataloader, device):
    """Run inference and collect predictions."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_all_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }


# ============================================================
# Training Functions
# ============================================================

def train_centralized_cnn(train_loader, val_loader, device):
    """Train Centralized CNN (ResNet50) on ISIC dataset."""
    print("\n" + "=" * 60)
    print("ISIC — CENTRALIZED CNN TRAINING (ResNet50)")
    print("=" * 60)

    model = build_cnn_model(num_classes=2, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BaselineCNN (ResNet50)")
    print(f"Total parameters: {total_params:,}")

    os.makedirs("weights/isic", exist_ok=True)
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        model_save_path="weights/isic/cnn_model.pth",
        weights_save_path="weights/isic/cnn_weights.pth",
    )
    return model, history


def train_centralized_hybrid(train_loader, val_loader, device):
    """Train Centralized Hybrid CNN-Transformer on ISIC dataset."""
    print("\n" + "=" * 60)
    print("ISIC — CENTRALIZED HYBRID CNN-TRANSFORMER TRAINING")
    print("=" * 60)

    model = build_hybrid_model(num_classes=2, pretrained=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: HybridCNNTransformer (ResNet50 + Transformer)")
    print(f"Total parameters: {total_params:,}")

    os.makedirs("weights/isic", exist_ok=True)
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        model_save_path="weights/isic/hybrid_model.pth",
        weights_save_path="weights/isic/hybrid_weights.pth",
    )
    return model, history


def train_fedavg_isic(hospital_loaders, val_loader, device):
    """Run FedAvg on ISIC dataset."""
    print("\n" + "=" * 60)
    print("ISIC — FEDERATED LEARNING — FedAvg")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    os.makedirs("weights/isic", exist_ok=True)
    model_save_path = "weights/isic/fedavg_model.pth"
    weights_save_path = "weights/isic/fedavg_weights.pth"

    print(f"\nStarting {FED_ROUNDS} federated rounds (FedAvg)...")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Clients: {list(hospital_loaders.keys())}\n")

    history = {"round": [], "val_acc": [], "val_loss": []}

    for round_num in range(1, FED_ROUNDS + 1):
        round_start = time.time()
        print(f"--- Round {round_num}/{FED_ROUNDS} ---")

        client_models = []
        for hosp_name, hosp_loader in hospital_loaders.items():
            client_model = copy.deepcopy(global_model)
            client_model, loss, acc = train_client(
                client_model, hosp_loader, device,
                epochs=LOCAL_EPOCHS, lr=LR,
            )
            print(f"  {hosp_name}: loss={loss:.4f}, acc={acc:.4f}")
            client_models.append(client_model)

        global_model = federated_average(global_model, client_models)
        global_model = global_model.to(device)
        val_loss, val_acc = validate(global_model, val_loader, criterion, device)
        elapsed = time.time() - round_start

        history["round"].append(round_num)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(f"  Global Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        print()

    print(f"FedAvg (ISIC) complete. Best Val Accuracy: {best_val_acc:.4f}")
    return global_model, history


def train_fedprox_isic(hospital_loaders, val_loader, device):
    """Run FedProx on ISIC dataset."""
    print("\n" + "=" * 60)
    print("ISIC — FEDERATED LEARNING — FedProx")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    os.makedirs("weights/isic", exist_ok=True)
    model_save_path = "weights/isic/fedprox_model.pth"
    weights_save_path = "weights/isic/fedprox_weights.pth"

    print(f"\nStarting {FED_ROUNDS} federated rounds (FedProx, mu={MU})...")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Clients: {list(hospital_loaders.keys())}\n")

    history = {"round": [], "val_acc": [], "val_loss": []}

    for round_num in range(1, FED_ROUNDS + 1):
        round_start = time.time()
        print(f"--- Round {round_num}/{FED_ROUNDS} ---")

        client_models = []
        for hosp_name, hosp_loader in hospital_loaders.items():
            client_model = copy.deepcopy(global_model)
            client_model, loss, acc = train_fedprox_client(
                client_model, global_model, hosp_loader, device,
                mu=MU, epochs=LOCAL_EPOCHS, lr=LR,
            )
            print(f"  {hosp_name}: loss={loss:.4f}, acc={acc:.4f}")
            client_models.append(client_model)

        global_model = federated_average(global_model, client_models)
        global_model = global_model.to(device)
        val_loss, val_acc = validate(global_model, val_loader, criterion, device)
        elapsed = time.time() - round_start

        history["round"].append(round_num)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(f"  Global Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        print()

    print(f"FedProx (ISIC) complete. Best Val Accuracy: {best_val_acc:.4f}")
    return global_model, history


def train_moon_isic(hospital_loaders, val_loader, device):
    """Run MOON on ISIC dataset."""
    print("\n" + "=" * 60)
    print("ISIC — FEDERATED LEARNING — MOON")
    print("=" * 60)

    global_model = build_hybrid_model(num_classes=2, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    os.makedirs("weights/isic", exist_ok=True)
    model_save_path = "weights/isic/moon_model.pth"
    weights_save_path = "weights/isic/moon_weights.pth"

    prev_client_models = {
        name: copy.deepcopy(global_model)
        for name in hospital_loaders.keys()
    }

    print(f"\nStarting {FED_ROUNDS} federated rounds (MOON, λ={MOON_LAMBDA}, τ={MOON_TEMP})...")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Clients: {list(hospital_loaders.keys())}\n")

    history = {"round": [], "val_acc": [], "val_loss": []}

    for round_num in range(1, FED_ROUNDS + 1):
        round_start = time.time()
        print(f"--- Round {round_num}/{FED_ROUNDS} ---")

        client_models = []
        for hosp_name, hosp_loader in hospital_loaders.items():
            client_model = copy.deepcopy(global_model)
            previous_model = prev_client_models[hosp_name]

            client_model, loss, acc = train_moon_client(
                client_model, global_model, previous_model,
                hosp_loader, device,
                lam=MOON_LAMBDA, temperature=MOON_TEMP,
                epochs=LOCAL_EPOCHS, lr=LR,
            )
            print(f"  {hosp_name}: loss={loss:.4f}, acc={acc:.4f}")
            client_models.append(client_model)
            prev_client_models[hosp_name] = copy.deepcopy(client_model)

        global_model = federated_average(global_model, client_models)
        global_model = global_model.to(device)
        val_loss, val_acc = validate(global_model, val_loader, criterion, device)
        elapsed = time.time() - round_start

        history["round"].append(round_num)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(f"  Global Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model, model_save_path)
            torch.save(global_model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")
        print()

    print(f"MOON (ISIC) complete. Best Val Accuracy: {best_val_acc:.4f}")
    return global_model, history


# ============================================================
# Evaluation & Plotting
# ============================================================

def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Benign", "Malignant"]
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           ylabel="True Label", xlabel="Predicted Label",
           title=title)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path, title="ROC Curve", label="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_comparison(results_dict, save_path, title="ROC Curve Comparison"):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]
    for idx, (name, data) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        auc = data["metrics"]["auc_roc"]
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_dataset_comparison(ham_metrics, isic_metrics, save_dir):
    """Generate cross-dataset comparison charts."""
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(ham_metrics.keys())
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("auc_roc", "AUC-ROC"),
        ("f1_score", "F1-Score"),
    ]

    for metric_key, metric_label in metrics_to_plot:
        ham_vals = [ham_metrics[m][metric_key] * 100 for m in model_names]
        isic_vals = [isic_metrics[m][metric_key] * 100 for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width / 2, ham_vals, width,
                       label="HAM10000", color="#3498db", edgecolor="white")
        bars2 = ax.bar(x + width / 2, isic_vals, width,
                       label="ISIC", color="#e74c3c", edgecolor="white")

        # Value labels
        for bar, val in zip(bars1, ham_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")
        for bar, val in zip(bars2, isic_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

        ax.set_ylabel(f"{metric_label} (%)", fontsize=12)
        ax.set_title(f"Cross-Dataset {metric_label} Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"cross_{metric_key}_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # Combined grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx]
        ham_vals = [ham_metrics[m][metric_key] * 100 for m in model_names]
        isic_vals = [isic_metrics[m][metric_key] * 100 for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width / 2, ham_vals, width,
               label="HAM10000", color="#3498db")
        ax.bar(x + width / 2, isic_vals, width,
               label="ISIC", color="#e74c3c")

        ax.set_ylabel(f"{metric_label} (%)")
        ax.set_title(f"{metric_label}")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" Hybrid", "\nHybrid") for n in model_names],
                           fontsize=7)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-Dataset Performance Comparison", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "cross_dataset_combined.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def evaluate_all_isic_models(val_loader, device):
    """Evaluate all ISIC-trained models and save results."""
    isic_models = [
        ("weights/isic/cnn_model.pth",     "Centralized CNN",    "results/isic/centralized_cnn"),
        ("weights/isic/hybrid_model.pth",   "Centralized Hybrid", "results/isic/centralized_hybrid"),
        ("weights/isic/fedavg_model.pth",   "FedAvg Hybrid",      "results/isic/fedavg"),
        ("weights/isic/fedprox_model.pth",  "FedProx Hybrid",     "results/isic/fedprox"),
        ("weights/isic/moon_model.pth",     "MOON Hybrid",        "results/isic/moon"),
    ]

    all_results = {}

    for model_path, model_name, result_dir in isic_models:
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name} (ISIC)")
        print(f"{'='*50}")

        if not os.path.exists(model_path):
            print(f"  ⚠ Not found: {model_path} — skipping.")
            continue

        model = torch.load(model_path, map_location=device, weights_only=False)
        model = model.to(device)
        model.eval()

        y_true, y_pred, y_prob = get_predictions(model, val_loader, device)
        metrics = compute_all_metrics(y_true, y_pred, y_prob)

        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1 Score:  {metrics['f1_score']*100:.2f}%")
        print(f"  AUC-ROC:   {metrics['auc_roc']*100:.2f}%")

        # Save metrics
        with open(os.path.join(result_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save classification report
        report = classification_report(
            y_true, y_pred, target_names=["Benign", "Malignant"],
            output_dict=True,
        )
        with open(os.path.join(result_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        # Plots
        plot_confusion_matrix(
            y_true, y_pred,
            os.path.join(result_dir, "confusion_matrix.png"),
            title=f"Confusion Matrix — {model_name} (ISIC)",
        )
        plot_roc_curve(
            y_true, y_prob,
            os.path.join(result_dir, "roc_curve.png"),
            title=f"ROC Curve — {model_name} (ISIC)",
            label=model_name,
        )

        all_results[model_name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    # ISIC ROC comparison
    if len(all_results) > 1:
        os.makedirs("results/isic/comparison", exist_ok=True)
        plot_roc_comparison(
            all_results,
            "results/isic/comparison/roc_comparison.png",
            title="ROC Curve Comparison (ISIC Dataset)",
        )

        # Save ISIC comparison CSV
        rows = []
        for name, data in all_results.items():
            m = data["metrics"]
            rows.append({
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1_score"], 4),
                "AUC-ROC": round(m["auc_roc"], 4),
            })
        df = pd.DataFrame(rows)
        df.to_csv("results/isic/model_comparison.csv", index=False)
        print(f"\n  Saved: results/isic/model_comparison.csv")

    return all_results


def generate_cross_dataset_analysis(isic_results):
    """Load HAM10000 results and compare with ISIC results."""
    print("\n" + "=" * 60)
    print("CROSS-DATASET ANALYSIS")
    print("=" * 60)

    # Load HAM10000 metrics
    ham_model_dirs = {
        "Centralized CNN": "results/centralized",
        "Centralized Hybrid": "results/federated_cnn",
        "FedAvg Hybrid": "results/federated_hybrid",
        "FedProx Hybrid": "results/federated_fedprox",
        "MOON Hybrid": "results/federated_moon",
    }

    ham_metrics = {}
    for name, result_dir in ham_model_dirs.items():
        json_path = os.path.join(result_dir, "metrics.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                ham_metrics[name] = json.load(f)
        else:
            print(f"  ⚠ HAM10000 metrics not found for {name}")

    isic_metrics = {
        name: data["metrics"] for name, data in isic_results.items()
    }

    # Only compare models that have results for both datasets
    common_models = sorted(
        set(ham_metrics.keys()) & set(isic_metrics.keys()),
        key=lambda x: list(ham_metrics.keys()).index(x)
    )

    if not common_models:
        print("  ⚠ No common models found for comparison.")
        return

    ham_filtered = {m: ham_metrics[m] for m in common_models}
    isic_filtered = {m: isic_metrics[m] for m in common_models}

    # Generate comparison charts
    print("\nGenerating cross-dataset comparison charts...")
    plot_cross_dataset_comparison(
        ham_filtered, isic_filtered, "results/cross_dataset"
    )

    # Generate cross-dataset comparison table
    os.makedirs("results/cross_dataset", exist_ok=True)

    rows = []
    for model_name in common_models:
        h = ham_filtered[model_name]
        i = isic_filtered[model_name]
        rows.append({
            "Model": model_name,
            "HAM_Accuracy": round(h["accuracy"], 4),
            "ISIC_Accuracy": round(i["accuracy"], 4),
            "Diff_Accuracy": round(i["accuracy"] - h["accuracy"], 4),
            "HAM_AUC": round(h["auc_roc"], 4),
            "ISIC_AUC": round(i["auc_roc"], 4),
            "Diff_AUC": round(i["auc_roc"] - h["auc_roc"], 4),
            "HAM_F1": round(h["f1_score"], 4),
            "ISIC_F1": round(i["f1_score"], 4),
            "Diff_F1": round(i["f1_score"] - h["f1_score"], 4),
            "HAM_Precision": round(h["precision"], 4),
            "ISIC_Precision": round(i["precision"], 4),
            "HAM_Recall": round(h["recall"], 4),
            "ISIC_Recall": round(i["recall"], 4),
        })

    df = pd.DataFrame(rows)
    csv_path = "results/cross_dataset/cross_dataset_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print("CROSS-DATASET COMPARISON SUMMARY")
    print(f"{'='*90}")
    header = (
        f"{'Model':<20s} "
        f"{'HAM Acc':>8s} {'ISIC Acc':>9s} {'Diff':>6s} | "
        f"{'HAM AUC':>8s} {'ISIC AUC':>9s} {'Diff':>6s} | "
        f"{'HAM F1':>7s} {'ISIC F1':>8s} {'Diff':>6s}"
    )
    print(header)
    print("-" * 90)
    for row in rows:
        print(
            f"{row['Model']:<20s} "
            f"{row['HAM_Accuracy']*100:>7.2f}% {row['ISIC_Accuracy']*100:>8.2f}% "
            f"{row['Diff_Accuracy']*100:>+5.2f}% | "
            f"{row['HAM_AUC']*100:>7.2f}% {row['ISIC_AUC']*100:>8.2f}% "
            f"{row['Diff_AUC']*100:>+5.2f}% | "
            f"{row['HAM_F1']*100:>6.2f}% {row['ISIC_F1']*100:>7.2f}% "
            f"{row['Diff_F1']*100:>+5.2f}%"
        )
    print("=" * 90)

    # Save analysis text
    analysis_path = "results/cross_dataset/analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("CROSS-DATASET VALIDATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Datasets: HAM10000 (~10,015 images) vs ISIC (~2,239 images)\n")
        f.write("Task: Binary Classification (Benign vs Malignant)\n\n")

        for row in rows:
            f.write(f"Model: {row['Model']}\n")
            f.write(f"  HAM10000:  Acc={row['HAM_Accuracy']*100:.2f}%, "
                    f"AUC={row['HAM_AUC']*100:.2f}%, "
                    f"F1={row['HAM_F1']*100:.2f}%\n")
            f.write(f"  ISIC:      Acc={row['ISIC_Accuracy']*100:.2f}%, "
                    f"AUC={row['ISIC_AUC']*100:.2f}%, "
                    f"F1={row['ISIC_F1']*100:.2f}%\n")
            f.write(f"  Diff:      Acc={row['Diff_Accuracy']*100:+.2f}%, "
                    f"AUC={row['Diff_AUC']*100:+.2f}%, "
                    f"F1={row['Diff_F1']*100:+.2f}%\n\n")

    print(f"\n  Saved analysis: {analysis_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Validation on ISIC Dataset"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["cnn", "hybrid", "fedavg", "fedprox", "moon", "all"],
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training, only run evaluation and comparison",
    )
    parser.add_argument(
        "--rounds", type=int, default=FED_ROUNDS,
        help="Number of federated rounds",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Number of centralized training epochs",
    )
    args = parser.parse_args()

    # Update module-level config from CLI args
    import cross_dataset_train as _self
    _self.FED_ROUNDS = args.rounds
    _self.EPOCHS = args.epochs

    device = get_device()

    # Create output directories
    dirs = [
        "weights/isic",
        "results/isic",
        "results/isic/centralized_cnn",
        "results/isic/centralized_hybrid",
        "results/isic/fedavg",
        "results/isic/fedprox",
        "results/isic/moon",
        "results/isic/comparison",
        "results/cross_dataset",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    if not args.skip_training:
        # ---------- Centralized Training ----------
        if args.model in ("all", "cnn", "hybrid"):
            train_loader, val_loader, test_loader, _ = prepare_isic_dataloaders(
                ISIC_BASE_DIR,
                batch_size=BATCH_SIZE,
                img_size=IMG_SIZE,
            )

            if args.model in ("all", "cnn"):
                train_centralized_cnn(train_loader, val_loader, device)

            if args.model in ("all", "hybrid"):
                train_centralized_hybrid(train_loader, val_loader, device)

        # ---------- Federated Training ----------
        if args.model in ("all", "fedavg", "fedprox", "moon"):
            hospital_loaders, fed_val_loader, _ = create_isic_hospital_loaders(
                ISIC_BASE_DIR,
                batch_size=BATCH_SIZE,
                img_size=IMG_SIZE,
            )

            if args.model in ("all", "fedavg"):
                train_fedavg_isic(hospital_loaders, fed_val_loader, device)

            if args.model in ("all", "fedprox"):
                train_fedprox_isic(hospital_loaders, fed_val_loader, device)

            if args.model in ("all", "moon"):
                train_moon_isic(hospital_loaders, fed_val_loader, device)

    # ---------- Evaluation ----------
    print("\n" + "=" * 60)
    print("EVALUATING ALL ISIC MODELS")
    print("=" * 60)

    # Use the ISIC test set for final evaluation (held-out Test/ folder)
    _, _, test_loader, _ = prepare_isic_dataloaders(
        ISIC_BASE_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
    )

    isic_results = evaluate_all_isic_models(test_loader, device)

    # ---------- Cross-Dataset Analysis ----------
    if isic_results:
        generate_cross_dataset_analysis(isic_results)

    print("\n✓ Cross-dataset validation complete!")
    print("  Results saved to: results/isic/ and results/cross_dataset/")


if __name__ == "__main__":
    main()
