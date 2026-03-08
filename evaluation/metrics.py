"""
Evaluation utilities — metrics for model performance.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


def get_predictions(model, dataloader, device):
    """
    Run inference on the dataloader and collect all predictions,
    probabilities, and true labels.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of malignant

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
    }
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*50}")
    print(f"Evaluation Results — {model_name}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"  {key:>12s}: {value:.4f}")
    print(f"{'='*50}")


def print_classification_report(y_true, y_pred):
    """Print sklearn classification report."""
    target_names = ["Benign (0)", "Malignant (1)"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))


def print_confusion_matrix(y_true, y_pred):
    """Print confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"  {'':>12s}  Pred:0  Pred:1")
    print(f"  {'Actual:0':>12s}  {cm[0][0]:>6d}  {cm[0][1]:>6d}")
    print(f"  {'Actual:1':>12s}  {cm[1][0]:>6d}  {cm[1][1]:>6d}")
