import type { GlContext } from './gl';
import type { GpuTensor } from './tensor';

/**
 * Readback helpers used only by the self-test.
 *
 * Nothing in the render loop may call these — a `readPixels` stalls the pipeline until the GPU has
 * caught up, which is exactly the cost the whole runtime is arranged to avoid. They live in their
 * own module so that rule is visible from the import list rather than only from a comment.
 */

/** Pulls a tensor back as a flat `[height][width][channels]` float array, channels unpadded. */
export function readTensor(ctx: GlContext, tensor: GpuTensor): Float32Array {
  const { gl } = ctx;
  const framebuffer = gl.createFramebuffer();
  if (!framebuffer) throw new Error('Could not create a readback framebuffer.');

  const out = new Float32Array(tensor.width * tensor.height * tensor.channels);
  const layer = new Float32Array(tensor.width * tensor.height * 4);

  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  try {
    for (let group = 0; group < tensor.groups; group++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, tensor.texture, 0, group);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);

      const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (complete !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`Readback framebuffer incomplete: 0x${complete.toString(16)}`);
      }

      gl.readPixels(0, 0, tensor.width, tensor.height, gl.RGBA, gl.FLOAT, layer);

      for (let y = 0; y < tensor.height; y++) {
        for (let x = 0; x < tensor.width; x++) {
          for (let j = 0; j < 4; j++) {
            const channel = group * 4 + j;
            if (channel >= tensor.channels) break;
            out[(y * tensor.width + x) * tensor.channels + channel] = layer[(y * tensor.width + x) * 4 + j];
          }
        }
      }
    }
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);
  }

  return out;
}

/** Uploads a flat `[height][width][channels]` array into a tensor, padding channels with zeros. */
export function writeTensor(ctx: GlContext, tensor: GpuTensor, data: Float32Array): void {
  const { gl } = ctx;
  const layer = new Float32Array(tensor.width * tensor.height * 4);

  gl.bindTexture(gl.TEXTURE_2D_ARRAY, tensor.texture);
  for (let group = 0; group < tensor.groups; group++) {
    layer.fill(0);
    for (let y = 0; y < tensor.height; y++) {
      for (let x = 0; x < tensor.width; x++) {
        for (let j = 0; j < 4; j++) {
          const channel = group * 4 + j;
          if (channel >= tensor.channels) break;
          layer[(y * tensor.width + x) * 4 + j] = data[(y * tensor.width + x) * tensor.channels + channel];
        }
      }
    }
    gl.texSubImage3D(
      gl.TEXTURE_2D_ARRAY, 0, 0, 0, group, tensor.width, tensor.height, 1, gl.RGBA, gl.FLOAT, layer,
    );
  }
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
}
