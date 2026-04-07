import torch
import numpy as np
from tqdm.auto import tqdm


def _forward_with_loss(model, x, y, criterion, device):
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
    if all(hasattr(metric, attr) for attr in ("reset", "update", "compute")):
        metric.reset()
        metric.update(preds, targets)
        value = metric.compute().item()
        metric.reset()
        return value

    return (preds.round().int() == targets).float().mean().item()


def train_one_epoch(model, dataloader, optimizer, criterion, metric, device, scaler=None):
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
    scaler = (
        torch.amp.GradScaler(device="cuda")
        if (use_amp and device.type == "cuda")
        else None
    )
    if scaler is not None:
        print("AMP enabled (float16 mixed precision)")

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
