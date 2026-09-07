"""Weight layout, kept apart from PyTorch on purpose.

The one thing that must match the browser exactly is how a convolution's weights are laid out in
the texture the shader indexes. Isolating that here means it can be checked on its own -- see the
`__main__` block, which verifies the vectorized version against the index formula written out
literally -- rather than only being exercised at the end of a training run.

The shader (`src/gpu/shaders.ts`) reads weight texel `(tap * paddedIn + cin) * groupsOut + group`,
where each texel holds four consecutive output channels and both channel counts are padded up to a
multiple of four. `src/model/pack.ts` is the same layout on the browser side.
"""

from __future__ import annotations

import numpy as np


def groups_for(channels: int) -> int:
    return (channels + 3) // 4


def pack_conv_array(weight_oihw: np.ndarray) -> np.ndarray:
    """`[out, in, kh, kw]` into the shader's texel order, as an `[n, 4]` float32 array."""
    out_channels, in_channels, kh, kw = weight_oihw.shape
    if kh != kw:
        raise ValueError("Only square kernels are supported.")

    # To [tap, in, out], the order the shader walks.
    hwio = np.ascontiguousarray(weight_oihw.transpose(2, 3, 1, 0)).reshape(kh * kw, in_channels, out_channels)

    padded_in = groups_for(in_channels) * 4
    groups_out = groups_for(out_channels)
    padded = np.zeros((kh * kw, padded_in, groups_out * 4), dtype=np.float32)
    padded[:, :in_channels, :out_channels] = hwio

    # Splitting the padded output axis into (group, component) and flattening the leading three axes
    # in C order gives exactly (tap * padded_in + cin) * groups_out + group.
    return padded.reshape(kh * kw, padded_in, groups_out, 4).reshape(-1, 4)


def pack_bias_array(bias: np.ndarray) -> np.ndarray:
    channels = bias.shape[0]
    padded = np.zeros(groups_for(channels) * 4, dtype=np.float32)
    padded[:channels] = bias
    return padded.reshape(-1, 4)


def _reference_pack(weight_oihw: np.ndarray) -> np.ndarray:
    """The same layout written out as the literal index formula, for the check below."""
    out_channels, in_channels, kh, _ = weight_oihw.shape
    padded_in = groups_for(in_channels) * 4
    groups_out = groups_for(out_channels)

    texels = np.zeros((kh * kh * padded_in * groups_out, 4), dtype=np.float32)
    for ky in range(kh):
        for kx in range(kh):
            tap = ky * kh + kx
            for cin in range(padded_in):
                for group in range(groups_out):
                    texel = (tap * padded_in + cin) * groups_out + group
                    for j in range(4):
                        cout = group * 4 + j
                        if cin < in_channels and cout < out_channels:
                            texels[texel, j] = weight_oihw[cout, cin, ky, kx]
    return texels


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for out_channels, in_channels, kernel in [(16, 3, 9), (32, 8, 3), (12, 6, 3), (3, 16, 9), (64, 64, 3)]:
        weight = rng.standard_normal((out_channels, in_channels, kernel, kernel)).astype(np.float32)
        fast = pack_conv_array(weight)
        slow = _reference_pack(weight)
        assert fast.shape == slow.shape, (fast.shape, slow.shape)
        assert np.array_equal(fast, slow), f"mismatch for {out_channels}x{in_channels}x{kernel}"
        print(f"ok  {in_channels:3d} -> {out_channels:3d}  k{kernel}  {fast.shape[0]} texels")
    print("packing layout matches the index formula the shader uses.")
