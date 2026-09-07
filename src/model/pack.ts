import { groupsFor } from '../gpu/gl';

/**
 * Repacking a convolution kernel into the texel order the shader indexes.
 *
 * The shader computes its weight texel as `(tap * paddedIn + cin) * groupsOut + groupOut`, and each
 * texel carries four consecutive output channels. Both channel counts round up to a multiple of
 * four, padded with zeros, which is what makes a three-channel RGB kernel cost the same as a
 * four-channel one.
 *
 * `train/export.py` performs the identical repack in NumPy. The two implementations have to agree
 * exactly, so this one is kept small and the self-test checks a packed convolution against a CPU
 * reference rather than trusting either.
 */
export function packConv(
  kernel: Float32Array,
  kernelSize: number,
  inChannels: number,
  outChannels: number,
): { texels: Float32Array; texelCount: number } {
  const paddedIn = groupsFor(inChannels) * 4;
  const groupsOut = groupsFor(outChannels);
  const taps = kernelSize * kernelSize;
  const texelCount = taps * paddedIn * groupsOut;
  const texels = new Float32Array(texelCount * 4);

  for (let tap = 0; tap < taps; tap++) {
    for (let cin = 0; cin < paddedIn; cin++) {
      for (let group = 0; group < groupsOut; group++) {
        const texel = (tap * paddedIn + cin) * groupsOut + group;
        for (let j = 0; j < 4; j++) {
          const cout = group * 4 + j;
          if (cin >= inChannels || cout >= outChannels) continue;
          texels[texel * 4 + j] = kernel[(tap * inChannels + cin) * outChannels + cout];
        }
      }
    }
  }

  return { texels, texelCount };
}

/** Biases, four to a texel, zero-padded to a whole number of channel groups. */
export function packBias(bias: Float32Array, outChannels: number): { texels: Float32Array; texelCount: number } {
  const texelCount = groupsFor(outChannels);
  const texels = new Float32Array(texelCount * 4);
  texels.set(bias.subarray(0, outChannels));
  return { texels, texelCount };
}
