"""Alternative embedding backbones for the Siamese wrapper.

Every backbone here obeys the same contract as SmallCNN, so it can be
passed straight into SiameseNetwork(backbone=...) without touching the
wrapper or the training loop:

    Input:  (B, in_channels, 150, 220) float32 in [0, 1]
    Output: (B, embedding_dim)

Usage:
    from sigver.models.backbones import build_backbone, BACKBONE_NAMES
    backbone = build_backbone("resnet18", embedding_dim=128, pretrained=True)
    model = SiameseNetwork(backbone=backbone)

Design notes
------------
1. Grayscale input. The torchvision backbones expect 3 channels. Rather
   than replicating the image across channels, the first conv's
   pretrained weights are SUMMED over the input-channel axis. For an
   image that would otherwise be replicated 3x this gives an identical
   response, at a third of the first-layer cost.

2. Embedding scale. SmallCNN emits unnormalised embeddings and
   ContrastiveLoss uses margin 1.0, so the distance scale is not free:
   if a backbone emits distances far above the margin, negative pairs
   sit in the flat region of max(0, margin - d)^2 and stop producing
   gradient. `l2_normalize` is available but defaults to False to keep
   these runs comparable with the Progress Report 2 baseline. WATCH the
   val_threshold column in history.json for the first epoch of any new
   backbone: prior runs sat near 0.48-0.53. A value near 0 or far above
   1.0 means the margin needs retuning for that architecture, and the
   EER is not comparable until it is.

3. Input normalisation. The v2 pipeline emits strictly binary {0, 1}
   inputs, whereas the ImageNet-pretrained weights were fitted on
   channel-standardised natural images. `input_norm` exposes this as an
   explicit choice; the default is "none", which keeps the input
   contract byte-identical to the baseline. Whether "imagenet" or
   "symmetric" transfers better on binary signature images is an open
   empirical question here, not something this module asserts - if a
   pretrained backbone underperforms scratch, this is the first thing
   to ablate.

4. Stochastic depth in a Siamese setting. EfficientNet-B0 ships 16
   StochasticDepth layers, active only in training mode. Because the
   two arms of a Siamese pair are two separate forward passes, each
   samples a DIFFERENT random path, so two identical images produce
   different embeddings and a genuine pair can never be driven to
   distance 0. That puts a noise floor under the positive term of the
   contrastive loss during training; it disappears at eval, so
   validation EER looks unaffected while the training signal is
   quietly noisier than the other arms. `stochastic_depth_prob`
   therefore defaults to 0.0 here, which keeps the comparison
   like-for-like and matches the Progress Report 2 finding that
   overfitting is not the binding constraint. Set it to 0.2 (the
   torchvision default) only as a deliberate regularisation ablation.

5. ViT and input size. torchvision's vit_b_16 has a fixed 224x224 input
   and no adaptive pooling, so 150x220 inputs must be resized. The
   resize uses NEAREST interpolation deliberately: bilinear would
   reintroduce intermediate grey values into a strictly binary input,
   which is precisely the anti-aliasing leak that PREPROCESSING_VERSION
   2 exists to eliminate. ResNet and EfficientNet both end in adaptive
   pooling and take 150x220 unchanged, which is one reason to run them
   first.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sigver.models.siamese import SmallCNN

__all__ = ["build_backbone", "BACKBONE_NAMES", "BACKBONES"]


# ImageNet channel statistics, used only when input_norm="imagenet".
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _InputNorm(nn.Module):
    """Optional input standardisation, applied before the backbone.

    mode="none"       pass through unchanged (default; matches baseline)
    mode="symmetric"  [0, 1] -> [-1, 1]
    mode="imagenet"   per-channel ImageNet standardisation. For
                      in_channels=1 the three ImageNet channel statistics
                      are averaged, which is the grayscale equivalent.
    """

    def __init__(self, mode: str = "none", in_channels: int = 1):
        super().__init__()
        if mode not in {"none", "symmetric", "imagenet"}:
            raise ValueError(
                f"input_norm must be one of 'none', 'symmetric', 'imagenet'; "
                f"got {mode!r}"
            )
        self.mode = mode
        if mode == "imagenet":
            if in_channels == 1:
                mean = (sum(_IMAGENET_MEAN) / 3.0,)
                std = (sum(_IMAGENET_STD) / 3.0,)
            elif in_channels == 3:
                mean, std = _IMAGENET_MEAN, _IMAGENET_STD
            else:
                raise ValueError(
                    f"input_norm='imagenet' supports in_channels 1 or 3; "
                    f"got {in_channels}"
                )
            # Buffers so they move with .to(device) and land in the state dict.
            self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        if self.mode == "symmetric":
            return x * 2.0 - 1.0
        return (x - self.mean) / self.std

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Rebuild a Conv2d for `in_channels`, summing pretrained weights.

    Summing over the input-channel axis makes the response to a
    single-channel image identical to the response the original layer
    would give that image replicated across 3 channels.
    """
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        w = conv.weight.sum(dim=1, keepdim=True)          # (out, 1, kh, kw)
        new_conv.weight.copy_(w.repeat(1, in_channels, 1, 1) / in_channels
                              if in_channels > 1 else w)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class _TorchvisionBackbone(nn.Module):
    """Wraps a torchvision model whose classifier head is replaced.

    `net` must already emit (B, embedding_dim). `resize_to` is used only
    by fixed-input-size architectures (ViT).
    """

    def __init__(self, net: nn.Module, norm: _InputNorm,
                 l2_normalize: bool = False,
                 resize_to: tuple[int, int] | None = None):
        super().__init__()
        self.norm = norm
        self.net = net
        self.l2_normalize = l2_normalize
        self.resize_to = resize_to

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.resize_to is not None and x.shape[-2:] != self.resize_to:
            # NEAREST: preserves the strictly binary {0, 1} input contract.
            x = F.interpolate(x, size=self.resize_to, mode="nearest")
        emb = self.net(x)
        if self.l2_normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb


def _load_torchvision(name: str, pretrained: bool, **kwargs):
    """Fetch a torchvision constructor and its default ImageNet weights.

    Uses the `weights=` API (torchvision >= 0.13). Raises with an
    actionable message rather than silently falling back to random
    initialisation, since a run that trains on random weights while the
    config says pretrained=True is exactly the kind of result that has
    to be thrown away later.
    """
    try:
        import torchvision.models as tvm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torchvision is required for the pretrained backbones. "
            "Install it in the training environment."
        ) from exc

    constructors: dict[str, tuple[Callable, str]] = {
        "resnet18": (tvm.resnet18, "ResNet18_Weights"),
        "resnet34": (tvm.resnet34, "ResNet34_Weights"),
        "efficientnet_b0": (tvm.efficientnet_b0, "EfficientNet_B0_Weights"),
        "vit_b_16": (tvm.vit_b_16, "ViT_B_16_Weights"),
        "vit_b_32": (tvm.vit_b_32, "ViT_B_32_Weights"),
    }
    ctor, weights_enum_name = constructors[name]

    if not pretrained:
        return ctor(weights=None, **kwargs)

    weights_enum = getattr(tvm, weights_enum_name, None)
    if weights_enum is None:  # pragma: no cover
        raise RuntimeError(
            f"torchvision.models.{weights_enum_name} not found. This module "
            f"targets the torchvision `weights=` API (>= 0.13). Check the "
            f"installed torchvision version before running."
        )
    return ctor(weights=weights_enum.DEFAULT, **kwargs)


def _build_resnet(name: str, embedding_dim: int, pretrained: bool,
                  in_channels: int, l2_normalize: bool,
                  input_norm: str) -> nn.Module:
    net = _load_torchvision(name, pretrained)
    net.conv1 = _adapt_first_conv(net.conv1, in_channels)
    net.fc = nn.Linear(net.fc.in_features, embedding_dim)
    return _TorchvisionBackbone(net, _InputNorm(input_norm, in_channels),
                                l2_normalize)


def _build_efficientnet(name: str, embedding_dim: int, pretrained: bool,
                        in_channels: int, l2_normalize: bool,
                        input_norm: str,
                        stochastic_depth_prob: float = 0.0) -> nn.Module:
    # stochastic_depth_prob=0.0 (not torchvision's 0.2): see note 4 in the
    # module docstring. Replacing net.classifier below also drops the
    # classifier Dropout(p=0.2), so with this override the module is fully
    # deterministic in train mode, like the other arms.
    net = _load_torchvision(name, pretrained,
                            stochastic_depth_prob=stochastic_depth_prob)
    stem_conv = net.features[0][0]
    net.features[0][0] = _adapt_first_conv(stem_conv, in_channels)
    in_features = net.classifier[-1].in_features
    net.classifier = nn.Linear(in_features, embedding_dim)
    return _TorchvisionBackbone(net, _InputNorm(input_norm, in_channels),
                                l2_normalize)


def _build_vit(name: str, embedding_dim: int, pretrained: bool,
               in_channels: int, l2_normalize: bool,
               input_norm: str) -> nn.Module:
    net = _load_torchvision(name, pretrained)
    net.conv_proj = _adapt_first_conv(net.conv_proj, in_channels)
    in_features = net.heads.head.in_features
    net.heads = nn.Linear(in_features, embedding_dim)
    return _TorchvisionBackbone(net, _InputNorm(input_norm, in_channels),
                                l2_normalize,
                                resize_to=(net.image_size, net.image_size))


def _build_smallcnn(name: str, embedding_dim: int, pretrained: bool,
                    in_channels: int, l2_normalize: bool,
                    input_norm: str) -> nn.Module:
    """The Progress Report 2 baseline, reachable through the same factory.

    Kept in the registry so `--backbone smallcnn` reproduces the
    published baseline through exactly the same code path as every
    comparison arm.
    """
    if pretrained:
        raise ValueError("smallcnn has no pretrained weights; pass pretrained=False")
    if in_channels != 1:
        raise ValueError(f"smallcnn is fixed at in_channels=1; got {in_channels}")
    net = SmallCNN(embedding_dim=embedding_dim)
    if input_norm == "none" and not l2_normalize:
        return net  # byte-identical to the baseline construction
    return _TorchvisionBackbone(net, _InputNorm(input_norm, in_channels),
                                l2_normalize)


BACKBONES: dict[str, Callable[..., nn.Module]] = {
    "smallcnn": _build_smallcnn,
    "resnet18": _build_resnet,
    "resnet34": _build_resnet,
    "efficientnet_b0": _build_efficientnet,
    "vit_b_16": _build_vit,
    "vit_b_32": _build_vit,
}

BACKBONE_NAMES = tuple(BACKBONES)


def build_backbone(name: str,
                   embedding_dim: int = 128,
                   pretrained: bool = True,
                   in_channels: int = 1,
                   l2_normalize: bool = False,
                   input_norm: str = "none") -> nn.Module:
    """Construct an embedding backbone for SiameseNetwork.

    Parameters
    ----------
    name
        One of BACKBONE_NAMES.
    embedding_dim
        Output dimensionality; 128 matches the baseline.
    pretrained
        Load ImageNet weights. Ignored (and rejected) for "smallcnn".
    in_channels
        1 for the v2 pipeline's single-channel output.
    l2_normalize
        Off by default, matching the baseline. See module docstring
        before switching it on - it changes the distance scale and so
        the meaning of ContrastiveLoss's margin.
    input_norm
        "none" (default, baseline-identical), "symmetric", or "imagenet".

    Returns
    -------
    nn.Module mapping (B, in_channels, 150, 220) -> (B, embedding_dim).
    """
    if name not in BACKBONES:
        raise ValueError(
            f"unknown backbone {name!r}; available: {', '.join(BACKBONE_NAMES)}"
        )
    return BACKBONES[name](name, embedding_dim, pretrained, in_channels,
                           l2_normalize, input_norm)