"""
Training utilities — shared training / validation loops and helpers.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import time


def get_device():
    """Return the best available device (GPU if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        # Enable cuDNN benchmark for faster training with fixed-size inputs
        torch.backends.cudnn.benchmark = True
    else:
        print("CUDA not available — using CPU")

    return device


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct / total)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=1e-4,
    model_save_path="weights/model.pth",
    weights_save_path="weights/weights.pth",
):
    """
    Full training loop with validation.
    Saves both the full model and just the state_dict (weights).
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save full model (architecture + weights)
            torch.save(model, model_save_path)
            # Save only weights
            torch.save(model.state_dict(), weights_save_path)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")
    return model, history
