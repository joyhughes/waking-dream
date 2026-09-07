import type { EXT_disjoint_timer_query_webgl2, GlContext } from '../gpu/gl';

/**
 * Frame timing.
 *
 * Two numbers matter and they are not the same. Wall-clock frame interval tells you what the user
 * sees, but on a display-synced loop it is pinned to the refresh rate and says nothing about how
 * much headroom is left — a pipeline using 3ms and one using 15ms both read as 60fps. GPU time,
 * when the driver will report it, is the number that actually moves when the capture size changes,
 * and it is the one the size/speed tradeoff has to be read off.
 */

const EMA_WEIGHT = 0.1;

export interface TimingSnapshot {
  fps: number;
  frameMs: number;
  /** Milliseconds the GPU spent on the frame, or null when the driver will not say. */
  gpuMs: number | null;
  /** Milliseconds of CPU time spent submitting the frame. */
  cpuMs: number;
}

export class FrameTimer {
  private lastFrameAt: number | null = null;
  private frameMs = 0;
  private cpuMs = 0;
  private gpuMs: number | null = null;

  private readonly ext: EXT_disjoint_timer_query_webgl2 | null;
  /** Queries in flight. A result is typically two or three frames behind the frame it measured. */
  private readonly pending: WebGLQuery[] = [];
  private active: WebGLQuery | null = null;

  constructor(private readonly ctx: GlContext) {
    this.ext = ctx.caps.timerQuery;
  }

  get supportsGpuTiming(): boolean {
    return this.ext !== null;
  }

  beginFrame(now: number): number {
    if (this.lastFrameAt !== null) {
      const delta = now - this.lastFrameAt;
      this.frameMs = this.frameMs === 0 ? delta : this.frameMs + (delta - this.frameMs) * EMA_WEIGHT;
    }
    this.lastFrameAt = now;

    const { gl } = this.ctx;
    if (this.ext && !this.active) {
      const query = gl.createQuery();
      if (query) {
        gl.beginQuery(this.ext.TIME_ELAPSED_EXT, query);
        this.active = query;
      }
    }

    return performance.now();
  }

  endFrame(cpuStart: number): void {
    const cpu = performance.now() - cpuStart;
    this.cpuMs = this.cpuMs === 0 ? cpu : this.cpuMs + (cpu - this.cpuMs) * EMA_WEIGHT;

    const { gl } = this.ctx;
    if (this.ext && this.active) {
      gl.endQuery(this.ext.TIME_ELAPSED_EXT);
      this.pending.push(this.active);
      this.active = null;
    }

    this.collect();
  }

  /**
   * Drains whatever query results are ready.
   *
   * `GPU_DISJOINT_EXT` going true means the GPU was preempted mid-measurement — another tab, a
   * power state change — and every outstanding result is then meaningless rather than merely noisy,
   * so they are all discarded rather than averaged in.
   */
  private collect(): void {
    const { gl } = this.ctx;
    if (!this.ext) return;

    if (gl.getParameter(this.ext.GPU_DISJOINT_EXT)) {
      for (const query of this.pending) gl.deleteQuery(query);
      this.pending.length = 0;
      return;
    }

    while (this.pending.length > 0) {
      const query = this.pending[0];
      if (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) break;
      const nanoseconds = gl.getQueryParameter(query, gl.QUERY_RESULT) as number;
      const ms = nanoseconds / 1e6;
      this.gpuMs = this.gpuMs === null ? ms : this.gpuMs + (ms - this.gpuMs) * EMA_WEIGHT;
      gl.deleteQuery(query);
      this.pending.shift();
    }
  }

  snapshot(): TimingSnapshot {
    return {
      fps: this.frameMs > 0 ? 1000 / this.frameMs : 0,
      frameMs: this.frameMs,
      gpuMs: this.gpuMs,
      cpuMs: this.cpuMs,
    };
  }

  reset(): void {
    this.lastFrameAt = null;
    this.frameMs = 0;
    this.cpuMs = 0;
    this.gpuMs = null;
  }

  dispose(): void {
    const { gl } = this.ctx;
    for (const query of this.pending) gl.deleteQuery(query);
    this.pending.length = 0;
    if (this.active) {
      gl.endQuery(this.ext!.TIME_ELAPSED_EXT);
      gl.deleteQuery(this.active);
      this.active = null;
    }
  }
}
