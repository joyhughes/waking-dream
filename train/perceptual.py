"""VGG-16 features, and the two things that get measured on them.

Both training paths lean on the same observation: comparing images pixel by pixel asks for the wrong
thing. Two frames can differ everywhere in pixels and be the same picture, and the safest way to
minimize a pixel loss over a set of equally-valid answers is to average them, which is a blur. A
network's intermediate features are invariant to exactly the differences that do not matter, so a
loss measured on them asks for the right thing instead.

  - **Content**: features at one layer, position by position. "Show the same things in the same
    places", with no opinion about the exact pixels.
  - **Style**: the Gram matrix at several layers -- which features co-occur, averaged over the whole
    frame with position thrown away entirely. "Use the same vocabulary of strokes and textures",
    with no opinion about where.

That the Gram matrix captures style is Gatys et al. 2015; using both as a training loss for a
feed-forward network is Johnson et al. 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Indices into torchvision's `vgg16().features` for the outputs of each block's last ReLU. These are
# the layers Gatys and Johnson both use; the names are the conventional ones.
VGG_LAYERS: dict[str, int] = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_3": 15,
    "relu4_3": 22,
}

DEFAULT_STYLE_LAYERS = ("relu1_2", "relu2_2", "relu3_3", "relu4_3")
DEFAULT_CONTENT_LAYER = "relu2_2"


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Normalized Gram matrix of a `[N, C, H, W]` feature map.

    Dividing by `C * H * W` rather than leaving the raw sum is what makes a style weight mean the
    same thing at 256px as at 512px, and what lets the same weight be reused when the style image is
    re-scaled. Without it the loss silently changes magnitude with every resolution change.
    """
    batch, channels, height, width = features.shape
    flat = features.reshape(batch, channels, height * width)
    return flat @ flat.transpose(1, 2) / (channels * height * width)


class VggFeatures(nn.Module):
    """A frozen VGG-16 truncated after the deepest layer anyone asks for."""

    def __init__(self, device: torch.device, layers: tuple[str, ...]):
        super().__init__()
        unknown = [name for name in layers if name not in VGG_LAYERS]
        if unknown:
            raise ValueError(f"Unknown VGG layers: {unknown}. Known: {sorted(VGG_LAYERS)}")

        self.wanted = {VGG_LAYERS[name]: name for name in layers}
        depth = max(self.wanted) + 1

        features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:depth]
        features.eval().to(device)
        for parameter in features.parameters():
            parameter.requires_grad_(False)
        self.features = features

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, images01: torch.Tensor) -> dict[str, torch.Tensor]:
        """Takes images in [0, 1] and returns the requested activations by name."""
        h = (images01 - self.mean) / self.std
        captured: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.features):
            h = layer(h)
            name = self.wanted.get(index)
            if name is not None:
                captured[name] = h
        return captured


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """Mean squared gradient magnitude, as a smoothness penalty.

    A small amount of this is worth having in style transfer: the style loss is happy to be satisfied
    by high-frequency noise that happens to have the right feature statistics, and on video that
    noise is a different noise every frame, which reads as crawling grain.
    """
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dy.pow(2).mean() + dx.pow(2).mean()


# How the equivariance warp samples outside the frame. Reflection rather than "border": MPS does
# not implement border padding for grid_sample at all, and zero padding would drag black in from
# outside every edge, teaching the network that frames are surrounded by darkness.
WARP_PADDING_MODE = "reflection"


def random_warp_grid(batch: int, height: int, width: int, device: torch.device, strength: float = 0.06) -> torch.Tensor:
    """A small random similarity transform per example, as a sampling grid.

    Used by both trainers for the equivariance term. Deliberately small: a large warp would ask the
    network to be stable under motion no two consecutive video frames could contain, trading detail
    quality for a robustness nothing needs.
    """
    angle = (torch.rand(batch, device=device) - 0.5) * 2 * strength
    scale = 1.0 + (torch.rand(batch, device=device) - 0.5) * 2 * strength
    shift = (torch.rand(batch, 2, device=device) - 0.5) * 2 * strength

    cos = torch.cos(angle) * scale
    sin = torch.sin(angle) * scale
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, :, 2] = shift

    return F.affine_grid(theta, (batch, 3, height, width), align_corners=False)
