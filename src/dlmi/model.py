import torch
import torch.nn as nn

# Hibou pathology foundation models (ViT-*/14, pretrained on histopathology).
# HistAI recommends loading them via transformers + trust_remote_code.
_HIBOU_MODELS = {
    "hibou-b": "histai/hibou-b",
    "hibou-l": "histai/hibou-L",
}


def is_hibou_model(model_name):
    """Return True when the requested backbone is a Hibou foundation model."""
    return model_name in _HIBOU_MODELS


def _resolve_attr(root, *paths):
    """Return the first nested attribute path found on ``root``."""
    for path in paths:
        obj = root
        found = True
        for name in path.split("."):
            obj = getattr(obj, name, None)
            if obj is None:
                found = False
                break
        if found:
            return obj
    return None


def get_feature_extractor(model_name="dinov2_vits14", device=None):
    """Load a pretrained DINOv2 feature extractor from torch hub.

    Parameters
    ----------
    model_name : str, optional
        Name of the DINOv2 model variant to load.
    device : torch.device or None, optional
        Device to place the model on. Uses CUDA if available when "None".

    Returns
    -------
    torch.nn.Module
        Pretrained feature extractor in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_hibou_model(model_name):
        model = get_finetunable_dinov2(
            model_name=model_name,
            num_blocks_to_unfreeze=0,
            device=device,
        ).backbone
    else:
        model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)

    model.eval()
    return model


def get_linear_probe(input_dim, num_classes=1, device=None):
    """Create a linear probing head with sigmoid output for binary classification.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector.
    num_classes : int, optional
        Number of output units.
    device : torch.device or None, optional
        Device to place the model on. Uses CUDA if available when "None".

    Returns
    -------
    torch.nn.Sequential
        Linear layer followed by a sigmoid activation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(input_dim, num_classes), nn.Sigmoid()).to(device)
    return model


def get_finetunable_dinov2(
    model_name="dinov2_vits14", num_blocks_to_unfreeze=2, device=None
):
    """Build a DINOv2 backbone with the last N transformer blocks unfrozen.

    Returns the full model (backbone + linear head) ready for end-to-end
    training.

    Parameters
    ----------
    model_name : str, optional
        Name of the DINOv2 model variant to load.
    num_blocks_to_unfreeze : int, optional
        Number of trailing transformer blocks to unfreeze.
    device : torch.device or None, optional
        Device to place the model on. Uses CUDA if available when "None".

    Returns
    -------
    DinoWithHead
        Model composed of the DINOv2 backbone and a binary classification head.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_hibou_model(model_name):
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers is required for Hibou models. Install with: pip install transformers"
            )

        _hf_model = AutoModel.from_pretrained(
            _HIBOU_MODELS[model_name], trust_remote_code=True
        )

        class _HibouBackbone(nn.Module):
            """Adapter exposing a DINO-like interface around the HF Hibou model."""

            def __init__(self, m):
                super().__init__()
                self._m = m
                self._blocks = _resolve_attr(
                    m,
                    "encoder.layer",
                    "encoder.layers",
                    "dinov2.encoder.layer",
                    "dinov2.encoder.layers",
                    "blocks",
                )
                self._norm = _resolve_attr(
                    m,
                    "layernorm",
                    "norm",
                    "dinov2.layernorm",
                    "dinov2.norm",
                )
                self._num_features = getattr(getattr(m, "config", None), "hidden_size", None)

                if self._blocks is None or self._norm is None or self._num_features is None:
                    raise AttributeError(
                        "Unsupported Hibou backbone structure returned by transformers."
                    )

            @property
            def blocks(self):
                return self._blocks

            @property
            def norm(self):
                return self._norm

            @property
            def num_features(self):
                return self._num_features

            def forward(self, x):
                outputs = self._m(pixel_values=x)
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is not None:
                    return pooled

                last_hidden = getattr(outputs, "last_hidden_state", None)
                if last_hidden is not None:
                    return last_hidden[:, 0]

                if isinstance(outputs, (tuple, list)) and outputs:
                    return outputs[0][:, 0]

                raise AttributeError("Unexpected Hibou output structure.")

        backbone = _HibouBackbone(_hf_model)
    else:
        backbone = torch.hub.load("facebookresearch/dinov2", model_name)

    n_blocks = len(backbone.blocks)

    if not 0 <= num_blocks_to_unfreeze <= n_blocks:
        raise ValueError(
            f"num_blocks_to_unfreeze must be in [0, {n_blocks}], "
            f"got {num_blocks_to_unfreeze}."
        )

    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False

    # Unfreeze the last N blocks
    # DINOv2-ViTS14 has 12 blocks: backbone.blocks[0] ... backbone.blocks[11]
    if num_blocks_to_unfreeze > 0:
        for block in backbone.blocks[-num_blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True

        # Keep the final normalization trainable when fine-tuning the backbone.
        for param in backbone.norm.parameters():
            param.requires_grad = True

    # Full model: backbone + classification head
    class DinoWithHead(nn.Module):
        """DINOv2 backbone with a binary classification head.

        Parameters
        ----------
        backbone : torch.nn.Module
            DINOv2 feature extractor.
        feat_dim : int
            Dimension of the backbone output features.
        """

        def __init__(self, backbone, feat_dim):
            super().__init__()
            self.backbone = backbone
            # Keep a single linear layer so old checkpoints remain loadable via
            # the same "head.0.*" parameter names.
            self.head = nn.Sequential(nn.Linear(feat_dim, 1))

        def forward_logits(self, x):
            """Forward pass returning raw logits before the sigmoid."""
            features = self.backbone(x)
            return self.head(features)

        def forward(self, x):
            """Forward pass through backbone and classification head.

            Parameters
            ----------
            x : torch.Tensor
                Input image batch of shape "(B, C, H, W)".

            Returns
            -------
            torch.Tensor
                Predicted probabilities of shape "(B, 1)".
            """
            return torch.sigmoid(self.forward_logits(x))

    model = DinoWithHead(backbone, backbone.num_features).to(device)
    return model
