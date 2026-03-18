"""
MOON — Model-Contrastive Federated Learning.

Implements:
  - Contrastive loss between current local, global, and previous local
    model embeddings.
  - Loss = CrossEntropyLoss + lambda * ContrastiveLoss

Reference:
  Li et al., "Model-Contrastive Federated Learning", CVPR 2021.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def model_contrastive_loss(z_local, z_global, z_prev, temperature=0.5):
    """
    Compute the MOON contrastive loss.

    Positive pair: (z_local, z_global)
    Negative pair: (z_local, z_prev)

    L = -log( exp(sim(z_local, z_global) / tau)
             / (exp(sim(z_local, z_global) / tau)
              + exp(sim(z_local, z_prev) / tau)) )

    Args:
        z_local: embeddings from current local model (B, d).
        z_global: embeddings from global model (B, d).
        z_prev: embeddings from previous local model (B, d).
        temperature: temperature scaling factor (tau).

    Returns:
        Scalar contrastive loss.
    """
    # Normalize embeddings
    z_local = F.normalize(z_local, dim=1)
    z_global = F.normalize(z_global, dim=1)
    z_prev = F.normalize(z_prev, dim=1)

    # Cosine similarities
    pos_sim = (z_local * z_global).sum(dim=1) / temperature  # (B,)
    neg_sim = (z_local * z_prev).sum(dim=1) / temperature    # (B,)

    # Contrastive loss: -log(exp(pos) / (exp(pos) + exp(neg)))
    logits = torch.stack([pos_sim, neg_sim], dim=1)  # (B, 2)
    labels = torch.zeros(z_local.size(0), dtype=torch.long,
                         device=z_local.device)  # positive is index 0
    loss = F.cross_entropy(logits, labels)

    return loss


def train_moon_client(model, global_model, previous_model, dataloader, device,
                      lam=0.5, temperature=0.5, epochs=1, lr=1e-4):
    """
    Train a local client model using MOON (Model-Contrastive FL).

    Loss = CrossEntropyLoss + lambda * ContrastiveLoss

    Args:
        model: local model to train (will be modified in-place).
        global_model: reference global model (for positive pair).
        previous_model: previous round's local model (for negative pair).
        dataloader: training DataLoader for this client.
        device: torch device.
        lam: weight for contrastive loss (default: 0.5).
        temperature: temperature for contrastive similarity (default: 0.5).
        epochs: number of local training epochs.
        lr: learning rate.

    Returns:
        model: updated local model.
        avg_loss: average training loss over the last epoch.
        avg_acc: average training accuracy over the last epoch.
    """
    model = model.to(device)
    global_model = global_model.to(device)
    previous_model = previous_model.to(device)

    global_model.eval()
    previous_model.eval()

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

            # Forward pass with embeddings
            outputs, z_local = model(images, return_embedding=True)
            ce_loss = criterion(outputs, labels)

            # Get global and previous embeddings (no gradient)
            with torch.no_grad():
                _, z_global = global_model(images, return_embedding=True)
                _, z_prev = previous_model(images, return_embedding=True)

            # Contrastive loss
            con_loss = model_contrastive_loss(
                z_local, z_global, z_prev, temperature=temperature
            )

            loss = ce_loss + lam * con_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        avg_acc = correct / total

    return model, avg_loss, avg_acc
