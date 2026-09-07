"""The slow DeepDream that the fast network is trained to imitate.

This is ordinary gradient-ascent DeepDream and it is meant to stay ordinary. Its job is to be a
*correct* reference, not a fast one -- every trick that would speed it up at the cost of fidelity
would be teaching the student the wrong thing. It runs once per training example, offline, and its
cost is paid in the dataset build rather than at sixty frames a second.

The technique is Mordvintsev et al. 2015. The pieces that matter for quality, all of which come from
that work and the notebooks around it:

  - an octave pyramid, so features are drawn at several scales rather than only the finest one;
  - Laplacian-pyramid gradient normalization, which stops the ascent from spending its whole step
    budget on the highest frequency band and is the single largest quality difference here;
  - random spatial jitter before each step, which is what keeps the network's receptive-field grid
    from printing itself into the result as a tiling artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import models

# GoogLeNet is the network the original DeepDream ran on, and its inception layers still give the
# most recognisable version of the effect. Everything here addresses layers by attribute name.
LAYER_STACK: tuple[str, ...] = (
    "conv2",
    "inception3a",
    "inception3b",
    "inception4a",
    "inception4b",
    "inception4c",
    "inception4d",
    "inception4e",
    "inception5a",
    "inception5b",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DreamSettings:
    """One teacher configuration, derived from a control vector by `settings_from_controls`."""

    layers: tuple[str, ...]
    layer_weights: tuple[float, ...]
    steps: int
    step_size: float
    octaves: int
    octave_scale: float
    pattern_scale: float
    jitter: int
    tv_weight: float


def settings_from_controls(controls: list[float], *, steps: int = 10, octaves: int = 3) -> DreamSettings:
    """Maps the [0,1] control vector onto actual teacher settings.

    The layer control indexes into `LAYER_STACK` with a two-layer span rather than picking one, so
    neighbouring control values give overlapping targets. That overlap is what makes the conditioned
    student's slider continuous instead of a set of discrete jumps between trained points.
    """
    layer_control, strength, scale = controls

    centre = layer_control * (len(LAYER_STACK) - 1)
    low = int(math.floor(centre))
    high = min(low + 1, len(LAYER_STACK) - 1)
    blend = centre - low

    return DreamSettings(
        layers=(LAYER_STACK[low], LAYER_STACK[high]),
        layer_weights=(1.0 - blend, blend) if high != low else (1.0, 0.0),
        steps=steps,
        # Strength moves the step size rather than the step count, so every example costs the same
        # to generate and a dataset build has a predictable runtime.
        step_size=0.02 + 0.10 * strength,
        octaves=octaves,
        octave_scale=1.4,
        pattern_scale=1.0 + 2.5 * scale,
        jitter=16,
        tv_weight=0.08,
    )


def _binomial_kernel(channels: int, device: torch.device) -> torch.Tensor:
    row = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device)
    kernel = torch.outer(row, row)
    kernel = kernel / kernel.sum()
    return kernel.expand(channels, 1, 5, 5).contiguous()


def _laplacian_normalize(gradient: torch.Tensor, levels: int) -> torch.Tensor:
    """Normalizes each frequency band of the gradient to unit deviation, then reassembles it.

    Without this the ascent is dominated by the finest band, because that is where a convolutional
    network's gradient has the most energy, and the result is high-frequency fur over a picture that
    never develops any large structure. Splitting the gradient into bands, scaling each to the same
    magnitude, and adding them back gives every scale an equal say in the step.
    """
    if levels <= 1:
        return gradient / (gradient.std() + 1e-8)

    channels = gradient.shape[1]
    kernel = _binomial_kernel(channels, gradient.device)

    current = gradient
    bands: list[torch.Tensor] = []
    for _ in range(levels - 1):
        padded = F.pad(current, (2, 2, 2, 2), mode="reflect")
        smoothed = F.conv2d(padded, kernel, groups=channels)
        downsampled = smoothed[:, :, ::2, ::2]
        upsampled = F.interpolate(downsampled, size=current.shape[-2:], mode="bilinear", align_corners=False)
        bands.append(current - upsampled)
        current = downsampled
    bands.append(current)

    total = torch.zeros_like(gradient)
    for band in bands:
        normalized = band / (band.std() + 1e-8)
        total = total + F.interpolate(normalized, size=gradient.shape[-2:], mode="bilinear", align_corners=False)

    return total / (total.std() + 1e-8)


def _total_variation_gradient(image: torch.Tensor) -> torch.Tensor:
    """Closed-form gradient of the squared total-variation penalty: a Laplacian, edge-padded.

    Edge padding rather than zero padding, because a zero-padded Laplacian reads the drop from the
    border pixel to imaginary black as real detail and pulls the whole frame toward a vignette.
    """
    channels = image.shape[1]
    stencil = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=image.device
    ).expand(channels, 1, 3, 3).contiguous()
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    return 2.0 * F.conv2d(padded, stencil, groups=channels)


class DeepDreamTeacher:
    """A frozen GoogLeNet used only for its intermediate activations."""

    def __init__(self, device: torch.device, backbone: str = "googlenet"):
        self.device = device
        if backbone != "googlenet":
            raise ValueError(f"Unsupported teacher backbone: {backbone}")

        # `transform_input` rescales the input for the original Caffe-era preprocessing; the input
        # here is already ImageNet-normalized, so it has to stay off or the activations shift.
        self.net = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1, transform_input=False)
        self.net.eval().to(device)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

        self._captured: dict[str, torch.Tensor] = {}
        for name in LAYER_STACK:
            module = getattr(self.net, name)
            module.register_forward_hook(self._capture(name))

    def _capture(self, name: str):
        def hook(_module, _inputs, output):
            self._captured[name] = output
        return hook

    def _activation_loss(self, image: torch.Tensor, settings: DreamSettings) -> torch.Tensor:
        self._captured.clear()
        self.net((image - self.mean) / self.std)

        total = image.new_zeros(())
        for name, weight in zip(settings.layers, settings.layer_weights):
            if weight == 0.0:
                continue
            # Mean of squares rather than mean: it rewards a strong response of either sign, which
            # is what produces the characteristic amplification instead of a global brightening.
            total = total + weight * self._captured[name].pow(2).mean()
        return total

    @torch.no_grad()
    def _resize(self, image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(image, size=size, mode="bilinear", align_corners=False)

    def dream(self, image: torch.Tensor, settings: DreamSettings) -> torch.Tensor:
        """Runs the full octave sweep on a `[1, 3, H, W]` image in [0, 1] and returns the result."""
        height, width = image.shape[-2:]

        # Pattern scale works by giving the network fewer pixels to draw on. The features it draws
        # are a fixed size in its own input, so a coarser working resolution makes them cover more
        # of the final frame -- the same lever octaves pull, held at one setting.
        work_height = max(64, int(round(height / settings.pattern_scale)))
        work_width = max(64, int(round(width / settings.pattern_scale)))

        current = self._resize(image, (work_height, work_width))

        for octave in range(settings.octaves):
            factor = settings.octave_scale ** (settings.octaves - 1 - octave)
            octave_size = (max(48, int(work_height / factor)), max(48, int(work_width / factor)))
            current = self._resize(current, octave_size)

            for _ in range(settings.steps):
                # Jitter shifts the image before the gradient is taken and shifts it back after, so
                # no pixel sits at the same offset within the network's stride grid twice.
                shift_y = int(torch.randint(-settings.jitter, settings.jitter + 1, (1,)).item())
                shift_x = int(torch.randint(-settings.jitter, settings.jitter + 1, (1,)).item())
                current = torch.roll(current, shifts=(shift_y, shift_x), dims=(2, 3))

                probe = current.detach().requires_grad_(True)
                loss = self._activation_loss(probe, settings)
                (gradient,) = torch.autograd.grad(loss, probe)

                direction = _laplacian_normalize(gradient, levels=4)
                if settings.tv_weight > 0:
                    smoothing = _total_variation_gradient(current)
                    smoothing = smoothing / (smoothing.std() + 1e-8)
                    direction = direction - settings.tv_weight * smoothing

                current = (current + settings.step_size * direction).clamp(0, 1)
                current = torch.roll(current, shifts=(-shift_y, -shift_x), dims=(2, 3))

        return self._resize(current, (height, width)).clamp(0, 1)
