"""
Evaluate all ISIC-trained models on the ISIC test set.

Evaluates:
  1. Centralized CNN (ResNet50)
  2. Centralized Hybrid (CNN + Transformer)
  3. Federated Hybrid — FedAvg
  4. Federated Hybrid — FedProx
  5. Federated Hybrid — MOON

Saves results (metrics, confusion matrices, ROC curves) to results/isic/.

Usage:
    python evaluate_isic.py
    python evaluate_isic.py --use-test-set   # evaluate on ISIC Test/ folder
"""

import os
import sys
import json
import argparse

import torch
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

from dataset.isic_data_loader import prepare_isic_dataloaders, build_isic_dataframe, ISICDataset
from dataset.data_loader import get_val_transforms
from training.trainer import get_device


# ============================================================
# Configuration
# ============================================================

ISIC_BASE_DIR = "Skin cancer ISIC The International Skin Imaging Collaboration"
BATCH_SIZE = 32
IMG_SIZE = 224


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


# ============================================================
# Main Evaluation
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ISIC-trained models"
    )
    parser.add_argument(
        "--use-test-set", action="store_true",
        help="Evaluate on the ISIC Test/ folder instead of the val split",
    )
    args = parser.parse_args()

    device = get_device()

    # Prepare evaluation data
    if args.use_test_set:
        print("Using ISIC Test set for evaluation...\n")
        test_dir = os.path.join(ISIC_BASE_DIR, "Test")
        test_df = build_isic_dataframe(test_dir)
        test_dataset = ISICDataset(test_df, transform=get_val_transforms(IMG_SIZE))
        eval_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        eval_set_name = "Test"
        print(f"Test samples: {len(test_dataset)}")
        print(f"  Malignant: {(test_df['label'] == 1).sum()}")
        print(f"  Benign:    {(test_df['label'] == 0).sum()}")
    else:
        print("Using ISIC validation split for evaluation...\n")
        _, eval_loader, _, _ = prepare_isic_dataloaders(
            ISIC_BASE_DIR,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
        )
        eval_set_name = "Val"

    # Models to evaluate
    isic_models = [
        ("weights/isic/cnn_model.pth",     "Centralized CNN",    "results/isic/centralized_cnn"),
        ("weights/isic/hybrid_model.pth",  "Centralized Hybrid", "results/isic/centralized_hybrid"),
        ("weights/isic/fedavg_model.pth",  "FedAvg Hybrid",      "results/isic/fedavg"),
        ("weights/isic/fedprox_model.pth", "FedProx Hybrid",     "results/isic/fedprox"),
        ("weights/isic/moon_model.pth",    "MOON Hybrid",        "results/isic/moon"),
    ]

    all_results = {}

    for model_path, model_name, result_dir in isic_models:
        os.makedirs(result_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name} (ISIC {eval_set_name})")
        print(f"{'='*50}")

        if not os.path.exists(model_path):
            print(f"  ⚠ Not found: {model_path} — skipping.")
            continue

        model = torch.load(model_path, map_location=device, weights_only=False)
        model = model.to(device)
        model.eval()

        y_true, y_pred, y_prob = get_predictions(model, eval_loader, device)
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

    # ROC comparison across all ISIC models
    if len(all_results) > 1:
        os.makedirs("results/isic/comparison", exist_ok=True)
        plot_roc_comparison(
            all_results,
            "results/isic/comparison/roc_comparison.png",
            title=f"ROC Curve Comparison (ISIC {eval_set_name} Set)",
        )

        # ISIC comparison CSV
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
        csv_path = "results/isic/model_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved to {csv_path}")

    # Comparison Summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print(f"COMPARISON SUMMARY — ISIC {eval_set_name.upper()} SET")
        print("=" * 70)
        header = (
            f"{'Model':<25s} {'Acc':>7s} {'Prec':>7s} "
            f"{'Recall':>7s} {'F1':>7s} {'AUC':>7s}"
        )
        print(header)
        print("-" * 70)
        for name, data in all_results.items():
            m = data["metrics"]
            row = (
                f"{name:<25s} "
                f"{m['accuracy']:>7.4f} "
                f"{m['precision']:>7.4f} "
                f"{m['recall']:>7.4f} "
                f"{m['f1_score']:>7.4f} "
                f"{m['auc_roc']:>7.4f}"
            )
            print(row)
        print("=" * 70)


if __name__ == "__main__":
    main()
