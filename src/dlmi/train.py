import torch
import numpy as np
from tqdm.notebook import tqdm


def train_one_epoch(model, dataloader, optimizer, criterion, metric, device):
    """Run one training epoch. Returns average loss and metric."""
    model.train()
    losses, metrics = [], []
    for x, y in tqdm(dataloader, leave=False, desc="Training"):
        optimizer.zero_grad()
        pred = model(x.to(device))
        loss = criterion(pred, y.to(device))
        loss.backward()
        optimizer.step()
        losses.extend([loss.item()] * len(y))
        m = metric(pred.cpu(), y.int().cpu())
        metrics.extend([m.item()] * len(y))
    return np.mean(losses), np.mean(metrics)


@torch.no_grad()
def validate(model, dataloader, criterion, metric, device):
    """Run validation. Returns average loss and metric."""
    model.eval()
    losses, metrics = [], []
    for x, y in tqdm(dataloader, leave=False, desc="Validating"):
        pred = model(x.to(device))
        loss = criterion(pred, y.to(device))
        losses.extend([loss.item()] * len(y))
        m = metric(pred.cpu(), y.int().cpu())
        metrics.extend([m.item()] * len(y))
    return np.mean(losses), np.mean(metrics)


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    metric,
    device,
    num_epochs=100,
    patience=10,
    save_path=None,
):
    """Full training loop with early stopping.

    Returns:
        dict with training history (train_losses, val_losses, train_metrics, val_metrics).
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metric": [],
        "val_metric": [],
    }
    min_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        train_loss, train_metric = train_one_epoch(
            model, train_dataloader, optimizer, criterion, metric, device
        )
        val_loss, val_metric = validate(
            model, val_dataloader, criterion, metric, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metric"].append(train_metric)
        history["val_metric"].append(val_metric)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_metric:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_metric:.4f}"
        )

        if val_loss < min_loss:
            print(f"  -> New best val loss: {min_loss:.4f} -> {val_loss:.4f}")
            min_loss = val_loss
            best_epoch = epoch
            if save_path:
                torch.save(model.state_dict(), save_path)

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return history
