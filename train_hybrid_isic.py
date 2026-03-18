"""
Train the Hybrid CNN + Transformer model on ISIC dataset for binary classification.

Mirrors the HAM10000 `train_hybrid.py` with identical hyperparameters.
Results are saved separately under weights/isic/ and results/isic/.

Usage:
    python train_hybrid_isic.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.isic_data_loader import prepare_isic_dataloaders
from models.hybrid_model import build_hybrid_model
from training.trainer import get_device, train_model


def main():
    # ---------- Configuration (same as HAM10000) ----------
    ISIC_BASE_DIR = "Skin cancer ISIC The International Skin Imaging Collaboration"
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    IMG_SIZE = 224

    # ISIC-specific output paths
    os.makedirs("weights/isic", exist_ok=True)
    MODEL_SAVE_PATH = "weights/isic/hybrid_model.pth"
    WEIGHTS_SAVE_PATH = "weights/isic/hybrid_weights.pth"

    # ---------- Prepare data ----------
    print("=" * 60)
    print("HYBRID CNN + TRANSFORMER TRAINING — ISIC DATASET")
    print("=" * 60)

    train_loader, val_loader, test_loader, _ = prepare_isic_dataloaders(
        isic_base_dir=ISIC_BASE_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
    )

    # ---------- Build model ----------
    device = get_device()
    model = build_hybrid_model(num_classes=2, pretrained=True)
    print(f"Model: HybridCNNTransformer (ResNet50 + Transformer)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ---------- Train ----------
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        model_save_path=MODEL_SAVE_PATH,
        weights_save_path=WEIGHTS_SAVE_PATH,
    )

    print("\nDone! Model saved to:")
    print(f"  Full model : {MODEL_SAVE_PATH}")
    print(f"  Weights    : {WEIGHTS_SAVE_PATH}")


if __name__ == "__main__":
    main()
