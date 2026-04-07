import torch
import torch.nn as nn

_HIBOU_MODELS = {
    "hibou-b": "histai/hibou-b",
    "hibou-l": "histai/hibou-L",
}


def is_hibou_model(model_name):
    return model_name in _HIBOU_MODELS


def _resolve_attr(root, *paths):
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


class HibouBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._blocks = _resolve_attr(
            model,
            "encoder.layer",
            "encoder.layers",
            "dinov2.encoder.layer",
            "dinov2.encoder.layers",
            "blocks",
        )
        self._norm = _resolve_attr(
            model,
            "layernorm",
            "norm",
            "dinov2.layernorm",
            "dinov2.norm",
        )
        self._num_features = getattr(getattr(model, "config", None), "hidden_size", None)

        if self._blocks is None or self._norm is None or self._num_features is None:
            raise AttributeError("Unsupported Hibou backbone structure returned by transformers.")

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
        outputs = self.model(pixel_values=x)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden[:, 0]

        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0][:, 0]

        raise AttributeError("Unexpected Hibou output structure.")


class DinoWithHead(nn.Module):
    def __init__(self, backbone, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(feat_dim, 1))

    def forward_logits(self, x):
        return self.head(self.backbone(x))

    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x))


def _default_device(device):
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_extractor(model_name="dinov2_vits14", device=None):
    device = _default_device(device)
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
    return nn.Sequential(nn.Linear(input_dim, num_classes), nn.Sigmoid()).to(
        _default_device(device)
    )


def get_finetunable_dinov2(
    model_name="dinov2_vits14", num_blocks_to_unfreeze=2, device=None
):
    device = _default_device(device)

    if is_hibou_model(model_name):
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers is required for Hibou models. Install with: pip install transformers"
            )

        hf_model = AutoModel.from_pretrained(
            _HIBOU_MODELS[model_name], trust_remote_code=True
        )
        backbone = HibouBackbone(hf_model)
    else:
        backbone = torch.hub.load("facebookresearch/dinov2", model_name)

    n_blocks = len(backbone.blocks)

    if not 0 <= num_blocks_to_unfreeze <= n_blocks:
        raise ValueError(
            f"num_blocks_to_unfreeze must be in [0, {n_blocks}], "
            f"got {num_blocks_to_unfreeze}."
        )

    for param in backbone.parameters():
        param.requires_grad = False

    if num_blocks_to_unfreeze > 0:
        for block in backbone.blocks[-num_blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True

        for param in backbone.norm.parameters():
            param.requires_grad = True

    return DinoWithHead(backbone, backbone.num_features).to(device)
