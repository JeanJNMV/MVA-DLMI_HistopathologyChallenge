import torch
import numpy as np
from tqdm.auto import tqdm


def _forward_with_loss(model, x, y, criterion, device):
    """Compute predictions and loss, using logits when the criterion expects them."""
    x = x.to(device)
    y = y.float().to(device)

    if isinstance(criterion, torch.nn.BCEWithLogitsLoss) and hasattr(
        model, "forward_logits"
    ):
        logits = model.forward_logits(x).squeeze(1)
        pred = torch.sigmoid(logits)
        loss = criterion(logits, y)
    else:
        pred = model(x).squeeze(1)
        loss = criterion(pred, y)

    return pred, loss


def _compute_epoch_metric(metric, preds, targets):
    """Compute a scalar metric over a full epoch when supported by the metric object."""
    if all(hasattr(metric, attr) for attr in ("reset", "update", "compute")):
        metric.reset()
        metric.update(preds, targets)
        value = metric.compute().item()
        metric.reset()
        return value

    return (preds.round().int() == targets).float().mean().item()


def train_one_epoch(model, dataloader, optimizer, criterion, metric, device, scaler=None):
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
    scaler : torch.amp.GradScaler or None, optional
        AMP gradient scaler. When provided, the forward pass runs in float16.

    Returns
    -------
    avg_loss : float
        Mean training loss over the epoch.
    avg_metric : float
        Mean training metric over the epoch.
    """
    model.train()
    losses = []
    preds_epoch, targets_epoch = [], []
    for x, y in tqdm(dataloader, leave=False, desc="Training"):
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                pred, loss = _forward_with_loss(model, x, y, criterion, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, loss = _forward_with_loss(model, x, y, criterion, device)
            loss.backward()
            optimizer.step()
        losses.extend([loss.item()] * len(y))
        preds_epoch.append(pred.detach().cpu())
        targets_epoch.append(y.int().cpu())

    preds_epoch = torch.cat(preds_epoch)
    targets_epoch = torch.cat(targets_epoch)
    return np.mean(losses), _compute_epoch_metric(metric, preds_epoch, targets_epoch)


@torch.no_grad()
def validate(model, dataloader, criterion, metric, device, use_amp=False):
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
    use_amp : bool, optional
        Whether to run the forward pass in float16 via AMP.

    Returns
    -------
    avg_loss : float
        Mean validation loss.
    avg_metric : float
        Mean validation metric.
    """
    model.eval()
    losses = []
    preds_epoch, targets_epoch = [], []
    for x, y in tqdm(dataloader, leave=False, desc="Validating"):
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                pred, loss = _forward_with_loss(model, x, y, criterion, device)
        else:
            pred, loss = _forward_with_loss(model, x, y, criterion, device)
        losses.extend([loss.item()] * len(y))
        preds_epoch.append(pred.detach().cpu())
        targets_epoch.append(y.int().cpu())

    preds_epoch = torch.cat(preds_epoch)
    targets_epoch = torch.cat(targets_epoch)
    return np.mean(losses), _compute_epoch_metric(metric, preds_epoch, targets_epoch)


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
    use_amp=True,
):
    """Full training loop with early stopping, cosine LR scheduling, and AMP.

    Training stops when the validation loss does not improve for
    "patience" consecutive epochs. The learning rate follows a cosine
    annealing schedule over "num_epochs". Mixed-precision (float16) is
    enabled automatically when a CUDA device is used and "use_amp=True".

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
    use_amp : bool, optional
        Enable automatic mixed precision on CUDA (default True).

    Returns
    -------
    dict
        Training history with keys "'train_loss'", "'val_loss'",
        "'train_metric'", and "'val_metric'".
    """
    # AMP: active only on CUDA
    scaler = (
        torch.amp.GradScaler(device="cuda")
        if (use_amp and device.type == "cuda")
        else None
    )
    if scaler is not None:
        print("AMP enabled (float16 mixed precision)")

    # Cosine annealing: LR decays from its initial value to ~0 over num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

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
            model, train_dataloader, optimizer, criterion, metric, device,
            scaler=scaler,
        )
        val_loss, val_metric = validate(
            model, val_dataloader, criterion, metric, device,
            use_amp=(scaler is not None),
        )

        scheduler.step()

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
