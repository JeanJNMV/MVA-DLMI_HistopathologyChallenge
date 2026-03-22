import random
import torch
import numpy as np
import pandas as pd


def set_seed(seed=0):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_submission(ids, preds, path, threshold=0.5):
    """Save predictions to a CSV submission file.

    Args:
        ids: list of image IDs
        preds: list/array of predicted probabilities
        path: output CSV path
        threshold: decision threshold for binary classification
    """
    df = pd.DataFrame(
        {"ID": ids, "Pred": [int(p > threshold) for p in preds]}
    ).set_index("ID")
    df.to_csv(path)
    return df
