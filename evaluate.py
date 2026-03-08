"""
Evaluate trained models on the validation set.

Compares:
  1. Centralized CNN (ResNet50)
  2. Centralized Hybrid (CNN + Transformer)
  3. Federated Hybrid model

Usage:
    python evaluate.py
"""

import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.data_loader import prepare_dataloaders
from training.trainer import get_device
from evaluation.metrics import (
    get_predictions,
    compute_metrics,
    print_metrics,
    print_classification_report,
    print_confusion_matrix,
)


def evaluate_model(model_path, val_loader, device, model_name="Model"):
    """Load a saved model and evaluate it."""
    if not os.path.exists(model_path):
        print(f"  ⚠ Model not found: {model_path} — skipping.")
        return None

    print(f"\nLoading {model_name} from {model_path}...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    y_true, y_pred, y_prob = get_predictions(model, val_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print_metrics(metrics, model_name)
    print_classification_report(y_true, y_pred)
    print_confusion_matrix(y_true, y_pred)

    return metrics


def main():
    # ---------- Configuration ----------
    METADATA_CSV = "HAM10000_metadata.csv"
    IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    BATCH_SIZE = 32
    IMG_SIZE = 224

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Prepare validation data
    _, val_loader, _ = prepare_dataloaders(
        metadata_csv=METADATA_CSV,
        image_dirs=IMAGE_DIRS,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        img_size=IMG_SIZE,
    )

    device = get_device()

    # Models to evaluate
    models_to_eval = [
        ("weights/cnn_model.pth", "Baseline CNN (ResNet50)"),
        ("weights/hybrid_model.pth", "Hybrid CNN + Transformer"),
        ("weights/federated_global.pth", "Federated Hybrid (FedAvg)"),
    ]

    all_results = {}

    for model_path, model_name in models_to_eval:
        metrics = evaluate_model(model_path, val_loader, device, model_name)
        if metrics:
            all_results[model_name] = metrics

    # ---------- Comparison Summary ----------
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        header = f"{'Model':<35s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AUC':>7s}"
        print(header)
        print("-" * 70)
        for name, m in all_results.items():
            row = (
                f"{name:<35s} "
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
