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


def get_finetunable_dinov2(
    model_name="dinov2_vits14", num_blocks_to_unfreeze=2, device=None
):
    """
    DINOv2 with the last N transformer blocks unfrozen.
    Returns the full model (backbone + linear head) ready for end-to-end training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = torch.hub.load("facebookresearch/dinov2", model_name)

    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False

    # Unfreeze the last N blocks
    # DINOv2-ViTS14 has 12 blocks: backbone.blocks[0] ... backbone.blocks[11]
    for block in backbone.blocks[-num_blocks_to_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True

    # Also unfreeze the final norm layer
    for param in backbone.norm.parameters():
        param.requires_grad = True

    # Full model: backbone + classification head
    class DinoWithHead(nn.Module):
        def __init__(self, backbone, feat_dim):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

        def forward(self, x):
            features = self.backbone(x)  # (B, 384) for ViTS14
            return self.head(features)

    model = DinoWithHead(backbone, backbone.num_features).to(device)
    return model
