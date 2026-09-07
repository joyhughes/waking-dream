"""The student: the feed-forward network that replaces the teacher's gradient ascent.

The shape is the fast-style-transfer network of Johnson et al. 2016 -- wide input convolution, two
stride-2 downsamples, residual blocks, two nearest-neighbour upsamples, wide output convolution.
The reason to keep that shape is arithmetic rather than tradition: the residual stack, which is
nearly all the work, runs at a quarter of the input resolution in each direction and so costs a
sixteenth of what it would at full size. That is the whole margin real time is bought with.

What is different is the target. This is not fitted to a style image's Gram statistics; it is fitted
to the actual output of `teacher.py` on the actual input, so one forward pass lands where a few
hundred ascent steps would have.

Every layer here has a counterpart in `src/gpu/ops.ts`, and the two have to agree exactly:

  - padding is replicate, matching the shader's clamped sampling coordinate;
  - upsampling is nearest-then-convolve rather than a transposed convolution, matching `resize`;
  - the output is tanh mapped to [0, 1], matching the `tanh01` activation;
  - instance normalization is per-image and per-channel with eps 1e-5, matching `instanceNorm`.

`export.py` walks this module in a fixed order. If the architecture here changes, that walk changes
with it, and the self-test's reference comparison is what catches it if they drift apart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CONTROL_DIMS


class ConvLayer(nn.Module):
    """Convolution with replicate padding, which is what the runtime's coordinate clamp amounts to."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad,) * 4, mode="replicate"))


class ConditionalInstanceNorm(nn.Module):
    """Instance normalization whose affine is a base vector plus a projection of the control vector.

    This is where the live controls come from. The convolution weights are fixed at export and could
    not be changed per frame at any sensible cost, but the per-channel scale and shift a
    normalization layer already applies can be recomputed for pennies -- so a single trained network
    covers a continuum of teacher configurations rather than one point in it.

    Conditional instance normalization is Dumoulin et al. 2017; the general conditioner in front of
    it is the FiLM framing of Perez et al. 2018. `src/model/conditioning.ts` evaluates the identical
    arithmetic on the CPU each frame.
    """

    def __init__(self, channels: int, film_hidden: int):
        super().__init__()
        self.channels = channels
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma_projection = nn.Linear(film_hidden, channels, bias=False)
        self.beta_projection = nn.Linear(film_hidden, channels, bias=False)

        # Zero-initialized, so the network starts as an unconditioned one and the conditioning grows
        # from there. Random projections would inject noise into every normalization before the
        # control vector means anything, which is a much worse place to start optimizing from.
        nn.init.zeros_(self.gamma_projection.weight)
        nn.init.zeros_(self.beta_projection.weight)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.unsqueeze(0) + self.gamma_projection(hidden)
        beta = self.beta.unsqueeze(0) + self.beta_projection(hidden)

        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        normalized = (x - mean) * torch.rsqrt(variance + 1e-5)

        return normalized * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)


class DreamNet(nn.Module):
    def __init__(self, width: int = 16, blocks: int = 5, film_hidden: int = 32):
        super().__init__()
        self.width = width
        self.blocks = blocks
        self.film_hidden = film_hidden

        w1, w2, w4 = width, width * 2, width * 4

        self.film = nn.Linear(CONTROL_DIMS, film_hidden)

        # Held in flat lists in execution order, because that is the order export.py walks.
        self.convs = nn.ModuleList(
            [
                ConvLayer(3, w1, 9, stride=1),
                ConvLayer(w1, w2, 3, stride=2),
                ConvLayer(w2, w4, 3, stride=2),
            ]
            + [ConvLayer(w4, w4, 3) for _ in range(2 * blocks)]
            + [
                ConvLayer(w4, w2, 3),
                ConvLayer(w2, w1, 3),
                ConvLayer(w1, 3, 9),
            ]
        )

        norm_channels = [w1, w2, w4] + [w4] * (2 * blocks) + [w2, w1]
        self.norms = nn.ModuleList(ConditionalInstanceNorm(c, film_hidden) for c in norm_channels)

    @property
    def output_conv_index(self) -> int:
        return len(self.convs) - 1

    def forward(self, x: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.film(controls))

        h = F.relu(self.norms[0](self.convs[0](x), hidden))
        h = F.relu(self.norms[1](self.convs[1](h), hidden))
        h = F.relu(self.norms[2](self.convs[2](h), hidden))

        for block in range(self.blocks):
            first, second = 3 + 2 * block, 4 + 2 * block
            residual = h
            h = F.relu(self.norms[first](self.convs[first](h), hidden))
            # The block's second normalization adds the skip and does not rectify, so the identity
            # path stays linear all the way through the stack.
            h = self.norms[second](self.convs[second](h), hidden) + residual

        upsample_first = 3 + 2 * self.blocks
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = F.relu(self.norms[upsample_first](self.convs[upsample_first](h), hidden))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = F.relu(self.norms[upsample_first + 1](self.convs[upsample_first + 1](h), hidden))

        return torch.tanh(self.convs[self.output_conv_index](h)) * 0.5 + 0.5

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
