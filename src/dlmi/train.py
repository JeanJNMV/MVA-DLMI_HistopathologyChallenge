import torch
import numpy as np
from tqdm.notebook import tqdm


def train_one_epoch(model, dataloader, optimizer, criterion, metric, device):
    """Run one training epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    dataloader : torch.utils.data.DataLoader
        Training dataloader yielding "(inputs, labels)" batches.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates.
    criterion : callable
        Loss function.
    metric : callable
        Metric function returning a scalar score.
    device : torch.device
        Device to run training on.

    Returns
    -------
    avg_loss : float
        Mean training loss over the epoch.
    avg_metric : float
        Mean training metric over the epoch.
    """
    model.train()
    losses, metrics = [], []
    for x, y in tqdm(dataloader, leave=False, desc="Training"):
        optimizer.zero_grad()
        pred = model(x.to(device)).squeeze(1)
        loss = criterion(pred, y.float().to(device))
        loss.backward()
        optimizer.step()
        losses.extend([loss.item()] * len(y))
        m = metric(pred.cpu(), y.int().cpu())
        metrics.extend([m.item()] * len(y))
    return np.mean(losses), np.mean(metrics)


@torch.no_grad()
def validate(model, dataloader, criterion, metric, device):
    """Run validation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    dataloader : torch.utils.data.DataLoader
        Validation dataloader yielding "(inputs, labels)" batches.
    criterion : callable
        Loss function.
    metric : callable
        Metric function returning a scalar score.
    device : torch.device
        Device to run evaluation on.

    Returns
    -------
    avg_loss : float
        Mean validation loss.
    avg_metric : float
        Mean validation metric.
    """
    model.eval()
    losses, metrics = [], []
    for x, y in tqdm(dataloader, leave=False, desc="Validating"):
        pred = model(x.to(device)).squeeze(1)
        loss = criterion(pred, y.float().to(device))
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

    Training stops when the validation loss does not improve for
    "patience" consecutive epochs.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_dataloader : torch.utils.data.DataLoader
        Training dataloader.
    val_dataloader : torch.utils.data.DataLoader
        Validation dataloader.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates.
    criterion : callable
        Loss function.
    metric : callable
        Metric function returning a scalar score.
    device : torch.device
        Device to run training on.
    num_epochs : int, optional
        Maximum number of training epochs.
    patience : int, optional
        Number of epochs without improvement before stopping.
    save_path : str or None, optional
        Path to save the best model weights. No saving when "None".

    Returns
    -------
    dict
        Training history with keys "'train_loss'", "'val_loss'",
        "'train_metric'", and "'val_metric'".
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
