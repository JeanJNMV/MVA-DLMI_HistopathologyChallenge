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
        self._num_features = getattr(
            getattr(model, "config", None), "hidden_size", None
        )

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
    model_name="dinov2_vits14",
    num_blocks_to_unfreeze=2,
    device=None,
    use_mixstyle=False,
    mixstyle_p=0.5,
    mixstyle_alpha=0.1,
):
    """Create a finetunable DINOv2 model with optional MixStyle and binary head.

    Parameters
    ----------
    model_name : str, optional
        Name of the DINOv2 model variant to load (default "dinov2_vits14").
    num_blocks_to_unfreeze : int, optional
        Number of final transformer blocks to unfreeze for fine-tuning.
    device : torch.device or None, optional
        Device to place the model on. Uses CUDA if available when "None".
    use_mixstyle : bool, optional
        Whether to insert MixStyle modules as forward hooks into the backbone.
    mixstyle_p : float, optional
        Probability of applying MixStyle during training (default 0.5).
    mixstyle_alpha : float, optional
        Beta distribution concentration parameter controlling mixing strength.

    Returns
    -------
    torch.nn.Module
        A model wrapping the DINOv2 backbone with a sigmoid binary head.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    for block in backbone.blocks[-num_blocks_to_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in backbone.norm.parameters():
        param.requires_grad = True

    # MixStyle module definition
    class MixStyle(nn.Module):
        """Mix instance-level feature statistics across samples in a batch.

        Targets domain shift by interpolating channel-wise mean and std
        between randomly paired samples, simulating unseen domain styles.

        Applied in token space: (B, N_tokens, D) from ViT blocks.

        Parameters
        ----------
        p : float
            Probability of applying MixStyle on a given forward pass.
        alpha : float
            Beta distribution concentration — higher = more mixing.

        References
        ----------
        Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
        """

        def __init__(self, p=0.5, alpha=0.1):
            super().__init__()
            self.p = p
            self.alpha = alpha

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply MixStyle to a token sequence.

            Parameters
            ----------
            x : torch.Tensor
                Token tensor of shape (B, N_tokens, D).

            Returns
            -------
            torch.Tensor
                Style-mixed tensor of shape (B, N_tokens, D).
            """
            if not self.training or torch.rand(1).item() > self.p:
                return x

            B, N, D = x.shape

            # Compute per-sample statistics across the token dimension
            mu = x.mean(dim=1, keepdim=True)  # (B, 1, D)
            sigma = x.std(dim=1, keepdim=True) + 1e-6  # (B, 1, D)
            x_norm = (x - mu) / sigma

            # Sample Beta mixing weights, bias toward stronger of the two
            lam = (
                torch.distributions.Beta(self.alpha, self.alpha)
                .sample((B,))
                .to(x.device)
            )  # (B,)
            lam = torch.max(lam, 1 - lam).view(B, 1, 1)  # (B, 1, 1)

            # Random pairing within the batch
            perm = torch.randperm(B, device=x.device)
            mu2, sigma2 = mu[perm], sigma[perm]

            # Interpolate statistics and re-apply
            mu_mix = lam * mu + (1 - lam) * mu2
            sigma_mix = lam * sigma + (1 - lam) * sigma2
            return x_norm * sigma_mix + mu_mix

    class DinoWithHead(nn.Module):
        """DINOv2 backbone with a binary classification head.

        Parameters
        ----------
        backbone : torch.nn.Module
            DINOv2 feature extractor.
        feat_dim : int
            Dimension of the backbone output features.
        mixstyle_blocks : list[int]
            Indices of transformer blocks after which MixStyle is applied.
        """

        def __init__(self, backbone, feat_dim, mixstyle_blocks):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

            # Register one MixStyle module per hook location
            self.mixstyles = nn.ModuleList(
                [MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) for _ in mixstyle_blocks]
            )

            # Attach hooks — capture the MixStyle ref in the closure
            for ms, block_idx in zip(self.mixstyles, mixstyle_blocks):
                backbone.blocks[block_idx].register_forward_hook(self._make_hook(ms))

        @staticmethod
        def _make_hook(ms_module):
            """Closure so each hook captures its own MixStyle instance."""

            def hook(module, input, output):
                return ms_module(output)

            return hook

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    # Apply MixStyle after blocks 3 and 6 (early-to-mid network)
    mixstyle_blocks = [3, 6] if use_mixstyle else []

    model = DinoWithHead(backbone, backbone.num_features, mixstyle_blocks).to(device)
    return model
