"""
Train the Baseline CNN (ResNet50) on HAM10000 for binary classification.

Usage:
    python train_cnn.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.data_loader import prepare_dataloaders
from models.cnn_model import build_cnn_model
from training.trainer import get_device, train_model


def main():
    # ---------- Configuration ----------
    METADATA_CSV = "HAM10000_metadata.csv"
    IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2
    IMG_SIZE = 224

    MODEL_SAVE_PATH = "weights/cnn_model.pth"
    WEIGHTS_SAVE_PATH = "weights/cnn_weights.pth"

    # ---------- Prepare data ----------
    print("=" * 60)
    print("BASELINE CNN TRAINING (ResNet50)")
    print("=" * 60)

    train_loader, val_loader, _ = prepare_dataloaders(
        metadata_csv=METADATA_CSV,
        image_dirs=IMAGE_DIRS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        img_size=IMG_SIZE,
    )

    # ---------- Build model ----------
    device = get_device()
    model = build_cnn_model(num_classes=2, pretrained=True)
    print(f"Model: BaselineCNN (ResNet50)")
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
