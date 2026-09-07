/**
 * WebGL2 context setup and the capability probe every other module in `gpu/` reads from.
 *
 * Two extensions decide whether this app can run at all. `EXT_color_buffer_float` makes float
 * textures renderable, which is the whole premise here — a network's intermediate activations are
 * signed and unbounded, and an 8-bit render target would quantize them into mush by the second
 * residual block. `OES_texture_float_linear` only affects 32-bit filtering; activations are
 * half-float, which WebGL2 filters natively, so its absence costs nothing.
 */

export interface GpuCaps {
  /** Colour attachments a single draw may write. Convolutions write four channels per attachment,
   *  so this is directly how many output channels one pass can produce. */
  maxDrawBuffers: number;
  /** Layers a `TEXTURE_2D_ARRAY` may have. Every tensor here is one, so this caps channel count. */
  maxArrayLayers: number;
  maxTextureSize: number;
  /** Present when the driver will hand back real GPU timings rather than wall-clock guesses. */
  timerQuery: EXT_disjoint_timer_query_webgl2 | null;
  rendererName: string;
}

export interface EXT_disjoint_timer_query_webgl2 {
  TIME_ELAPSED_EXT: number;
  GPU_DISJOINT_EXT: number;
}

export interface GlContext {
  gl: WebGL2RenderingContext;
  caps: GpuCaps;
  canvas: HTMLCanvasElement;
}

export class GpuUnsupportedError extends Error {}

export function createGlContext(canvas: HTMLCanvasElement): GlContext {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    // The output canvas is read back by MediaRecorder and by "save frame", both of which need the
    // drawing buffer to still hold the last frame when they ask for it.
    preserveDrawingBuffer: true,
    powerPreference: 'high-performance',
    premultipliedAlpha: false,
  });

  if (!gl) {
    throw new GpuUnsupportedError('This browser does not support WebGL2.');
  }

  if (!gl.getExtension('EXT_color_buffer_float')) {
    throw new GpuUnsupportedError(
      'This GPU cannot render to float textures (EXT_color_buffer_float is missing), which the network needs.',
    );
  }
  // Not required, but 32-bit render targets are used for the instance-norm reduction and some
  // drivers only expose them through this extension rather than the one above.
  gl.getExtension('EXT_float_blend');
  gl.getExtension('OES_texture_float_linear');

  const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const rendererName = debugInfo
    ? String(gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL))
    : String(gl.getParameter(gl.RENDERER));

  const caps: GpuCaps = {
    maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS) as number,
    maxArrayLayers: gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number,
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
    timerQuery: gl.getExtension('EXT_disjoint_timer_query_webgl2') as EXT_disjoint_timer_query_webgl2 | null,
    rendererName,
  };

  return { gl, caps, canvas };
}

/**
 * Channels are carried four to a texture layer, so every channel count the runtime deals with is
 * rounded up to a multiple of four. Exported weights are zero-padded to match, which is why a
 * three-channel RGB input costs the same as a four-channel one.
 */
export function groupsFor(channels: number): number {
  return Math.ceil(channels / 4);
}
