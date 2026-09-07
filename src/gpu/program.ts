import type { GlContext } from './gl';
import type { GpuTensor } from './tensor';

/**
 * Everything in this runtime is a full-screen quad drawn into a float render target, so the plumbing
 * around that — compiling a program once and reusing it, binding array-texture layers as colour
 * attachments, setting uniforms without a `getUniformLocation` call per frame — lives here rather
 * than being repeated in every op.
 */

const VERTEX_SHADER = `#version 300 es
// A single triangle that covers the viewport. Cheaper than two, and with no shared edge there is no
// diagonal seam for the rasterizer to double-shade.
void main() {
  vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Could not create a shader.');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? 'unknown error';
    gl.deleteShader(shader);
    const numbered = source
      .split('\n')
      .map((line, i) => `${String(i + 1).padStart(3)} | ${line}`)
      .join('\n');
    throw new Error(`Shader failed to compile: ${log}\n${numbered}`);
  }
  return shader;
}

export class Program {
  private readonly uniforms = new Map<string, WebGLUniformLocation | null>();
  private textureUnit = 0;

  constructor(
    private readonly ctx: GlContext,
    readonly handle: WebGLProgram,
  ) {}

  private location(name: string): WebGLUniformLocation | null {
    if (!this.uniforms.has(name)) {
      this.uniforms.set(name, this.ctx.gl.getUniformLocation(this.handle, name));
    }
    return this.uniforms.get(name) ?? null;
  }

  use(): this {
    this.ctx.gl.useProgram(this.handle);
    this.textureUnit = 0;
    return this;
  }

  int(name: string, value: number): this {
    const loc = this.location(name);
    if (loc) this.ctx.gl.uniform1i(loc, value);
    return this;
  }

  float(name: string, value: number): this {
    const loc = this.location(name);
    if (loc) this.ctx.gl.uniform1f(loc, value);
    return this;
  }

  vec2(name: string, x: number, y: number): this {
    const loc = this.location(name);
    if (loc) this.ctx.gl.uniform2f(loc, x, y);
    return this;
  }

  vec4(name: string, x: number, y: number, z: number, w: number): this {
    const loc = this.location(name);
    if (loc) this.ctx.gl.uniform4f(loc, x, y, z, w);
    return this;
  }

  ivec2(name: string, x: number, y: number): this {
    const loc = this.location(name);
    if (loc) this.ctx.gl.uniform2i(loc, x, y);
    return this;
  }

  /** Binds a tensor to the next free texture unit. Units are handed out in `use()` order. */
  tensor(name: string, tensor: GpuTensor): this {
    const { gl } = this.ctx;
    const unit = this.textureUnit++;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tensor.texture);
    this.int(name, unit);
    return this;
  }

  /** Binds a plain 2D texture (weights, conditioning, video frames) to the next free unit. */
  texture2d(name: string, texture: WebGLTexture): this {
    const { gl } = this.ctx;
    const unit = this.textureUnit++;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    this.int(name, unit);
    return this;
  }
}

/** Compiles and caches programs by their fragment source, since generated shaders repeat a lot. */
export class ProgramCache {
  private readonly programs = new Map<string, Program>();
  private vertexShader: WebGLShader | null = null;

  constructor(private readonly ctx: GlContext) {}

  get(fragmentSource: string): Program {
    const existing = this.programs.get(fragmentSource);
    if (existing) return existing;

    const { gl } = this.ctx;
    this.vertexShader ??= compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragment = compile(gl, gl.FRAGMENT_SHADER, fragmentSource);

    const handle = gl.createProgram();
    if (!handle) throw new Error('Could not create a program.');
    gl.attachShader(handle, this.vertexShader);
    gl.attachShader(handle, fragment);
    gl.linkProgram(handle);
    gl.deleteShader(fragment);

    if (!gl.getProgramParameter(handle, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(handle) ?? 'unknown error';
      gl.deleteProgram(handle);
      throw new Error(`Program failed to link: ${log}`);
    }

    const program = new Program(this.ctx, handle);
    this.programs.set(fragmentSource, program);
    return program;
  }

  get size(): number {
    return this.programs.size;
  }

  dispose(): void {
    const { gl } = this.ctx;
    for (const program of this.programs.values()) gl.deleteProgram(program.handle);
    this.programs.clear();
    if (this.vertexShader) gl.deleteShader(this.vertexShader);
    this.vertexShader = null;
  }
}

/**
 * Where a draw goes. A pass writes one or more consecutive channel groups of a tensor, or the
 * canvas. Grouping several layers into one draw is the difference between reading each input
 * channel once per output channel and reading it once per four of them, so the multi-attachment
 * path is the normal one, not an optimization bolted on later.
 */
export class RenderTargets {
  private readonly framebuffer: WebGLFramebuffer;
  /** Vertex-array state has to be bound even though the vertex shader reads no attributes. */
  private readonly vao: WebGLVertexArrayObject;

  constructor(private readonly ctx: GlContext) {
    const { gl } = ctx;
    const framebuffer = gl.createFramebuffer();
    const vao = gl.createVertexArray();
    if (!framebuffer || !vao) throw new Error('Could not create framebuffer state.');
    this.framebuffer = framebuffer;
    this.vao = vao;
  }

  /** Points the framebuffer at `count` layers of `tensor` starting at `firstGroup`, then draws. */
  drawInto(tensor: GpuTensor, firstGroup: number, count: number, draw: () => void): void {
    const { gl } = this.ctx;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);

    const buffers: number[] = [];
    for (let i = 0; i < count; i++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, tensor.texture, 0, firstGroup + i);
      buffers.push(gl.COLOR_ATTACHMENT0 + i);
    }
    // Attachments left over from a previous, wider pass would otherwise still be written.
    for (let i = count; i < this.ctx.caps.maxDrawBuffers; i++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, null, 0, 0);
    }
    gl.drawBuffers(buffers);
    gl.viewport(0, 0, tensor.width, tensor.height);

    this.runDraw(draw);
  }

  /** Draws into a plain 2D texture, which is how the instance-norm reduction accumulates. */
  drawInto2D(textures: WebGLTexture[], width: number, height: number, draw: () => void): void {
    const { gl } = this.ctx;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);

    const buffers: number[] = [];
    textures.forEach((texture, i) => {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texture, 0);
      buffers.push(gl.COLOR_ATTACHMENT0 + i);
    });
    for (let i = textures.length; i < this.ctx.caps.maxDrawBuffers; i++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, null, 0, 0);
    }
    gl.drawBuffers(buffers);
    gl.viewport(0, 0, width, height);

    this.runDraw(draw);
  }

  /** Draws to the canvas itself. Only the final composite does this. */
  drawToCanvas(width: number, height: number, draw: () => void): void {
    const { gl } = this.ctx;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    this.runDraw(draw);
  }

  private runDraw(draw: () => void): void {
    const { gl } = this.ctx;
    gl.bindVertexArray(this.vao);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    draw();
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  dispose(): void {
    const { gl } = this.ctx;
    gl.deleteFramebuffer(this.framebuffer);
    gl.deleteVertexArray(this.vao);
  }
}
