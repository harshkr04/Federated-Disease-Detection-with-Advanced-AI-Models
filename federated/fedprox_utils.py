"""
FedProx — Federated Proximal Optimization.

Implements:
  - FedProx local client training with proximal term
  - Loss = CrossEntropyLoss + (mu / 2) * ||w_local - w_global||^2

Reference:
  Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
"""

import copy
import torch
import torch.nn as nn


def proximal_term(model, global_model):
    """
    Compute the proximal term: ||w_local - w_global||^2.

    Args:
        model: current local model being trained.
        global_model: reference global model (frozen).

    Returns:
        Scalar tensor representing the squared L2 norm of parameter differences.
    """
    prox = torch.tensor(0.0, device=next(model.parameters()).device)
    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
        prox += (local_param - global_param.detach()).pow(2).sum()
    return prox


def train_fedprox_client(model, global_model, dataloader, device, mu=0.01,
                         epochs=1, lr=1e-4):
    """
    Train a local client model using FedProx.

    Loss = CrossEntropyLoss + (mu / 2) * ||w_local - w_global||^2

    Args:
        model: local model to train (will be modified in-place).
        global_model: reference global model (used for proximal term, not updated).
        dataloader: training DataLoader for this client.
        device: torch device (cuda / cpu).
        mu: proximal term coefficient (default: 0.01).
        epochs: number of local training epochs.
        lr: learning rate.

    Returns:
        model: updated local model.
        avg_loss: average training loss over the last epoch.
        avg_acc: average training accuracy over the last epoch.
    """
    model = model.to(device)
    global_model = global_model.to(device)
    global_model.eval()  # freeze global model — used only as reference

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    avg_loss = 0.0
    avg_acc = 0.0

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = criterion(outputs, labels)

            # Proximal term
            prox_loss = proximal_term(model, global_model)
            loss = ce_loss + (mu / 2.0) * prox_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        avg_acc = correct / total

    return model, avg_loss, avg_acc
