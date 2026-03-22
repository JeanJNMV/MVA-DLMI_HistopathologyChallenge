import torch
import torch.nn as nn


def get_feature_extractor(model_name="dinov2_vits14", device=None):
    """Load a pretrained DINOv2 feature extractor from torch hub."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
    model.eval()
    return model


def get_linear_probe(input_dim, num_classes=1, device=None):
    """Create a linear probing head (sigmoid output for binary classification)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(input_dim, num_classes), nn.Sigmoid()).to(device)
    return model
