/**
 * Per-device ceilings for the things here whose cost scales with resolution.
 *
 * Desktop browsers hand a tab several gigabytes without complaining. Mobile Safari does not: once a
 * page's footprint crosses a threshold well under 1 GB on most iPhones, iOS kills the WebContent
 * process outright. There is no catchable error and no warning — the tab reloads itself or shows
 * "A problem repeatedly occurred" — so the only defence is not to allocate that much in the first
 * place. Every limit here exists to keep a peak allocation off that cliff.
 *
 * (The same problem, and the same approach, as the dream project's `deviceLimits.ts`.)
 */

export interface DeviceLimits {
  /** True for phones and tablets, where the per-tab ceiling is low and enforced by process death. */
  memoryConstrained: boolean;
  /** Ceiling on pooled texture memory. */
  poolBudgetBytes: number;
  /** Largest capture size the control will offer. */
  maxCaptureSize: number;
  /** Where the capture size starts. */
  defaultCaptureSize: number;
  /**
   * Prefer the cheapest model on startup rather than the first listed.
   *
   * On a phone this is two separate wins: a 300 kB download instead of 4 MB, quite possibly over
   * cellular, and a network whose residual stack is a fraction of the arithmetic. Both matter more
   * than picking the best-looking model for someone who has not asked for one yet.
   */
  preferCheapestModel: boolean;
}

function isMobileBrowser(): boolean {
  if (typeof navigator === 'undefined') return false;

  // iPadOS reports itself as a Mac, and has been doing so since iPadOS 13. A touch-capable "Mac"
  // is an iPad, because no actual Mac reports maxTouchPoints above zero.
  const isIpad = /Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1;
  return isIpad || /Android|iPhone|iPad|iPod|Mobile|Silk/i.test(navigator.userAgent);
}

let cached: DeviceLimits | null = null;

export function getDeviceLimits(): DeviceLimits {
  if (cached) return cached;

  const constrained =
    isMobileBrowser() ||
    // Not implemented in Safari or Firefox, so this only ever adds detections, never removes the
    // user-agent one above.
    ((navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8) <= 4;

  cached = constrained
    ? {
        memoryConstrained: true,
        // A 512x384 feature map at 96 channels is ~18 MB, and a forward pass holds several at once.
        // 96 MB leaves room for the model, the feedback buffer, and the browser's own frame copies
        // inside what a phone tab survives.
        poolBudgetBytes: 96 * 1024 * 1024,
        maxCaptureSize: 512,
        defaultCaptureSize: 192,
        preferCheapestModel: true,
      }
    : {
        memoryConstrained: false,
        poolBudgetBytes: 384 * 1024 * 1024,
        maxCaptureSize: 1024,
        defaultCaptureSize: 256,
        preferCheapestModel: false,
      };

  return cached;
}
