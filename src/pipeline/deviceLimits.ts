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
   * Keyed on the connection rather than on being a phone. A modern phone runs the full-size model
   * comfortably, so choosing the small one for everyone on mobile gave up quality for nothing; what
   * actually hurts is pulling 4 MB down a metered or slow link before anything appears. So the rule
   * is about the pipe, not the processor.
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

/**
 * Whether the network is the thing to be careful about.
 *
 * `saveData` is an explicit request to not spend the user's data, and honouring it is not optional.
 * The effective-type check catches slow links that have not asked. Chrome-only, so this only ever
 * adds caution, never removes it.
 */
function onSlowConnection(): boolean {
  if (typeof navigator === 'undefined') return false;
  const connection = (navigator as Navigator & {
    connection?: { saveData?: boolean; effectiveType?: string };
  }).connection;
  if (!connection) return false;
  return connection.saveData === true || /^(slow-)?2g$|^3g$/.test(connection.effectiveType ?? '');
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
        // Raised from an initial guess of 96 MB after a full-size model turned out to run fine on a
        // current phone. This is only a churn ceiling, not a hard limit — the pool never evicts a
        // tensor a frame is using, so exceeding it costs re-allocation rather than correctness.
        poolBudgetBytes: 192 * 1024 * 1024,
        maxCaptureSize: 768,
        defaultCaptureSize: 256,
        preferCheapestModel: onSlowConnection(),
      }
    : {
        memoryConstrained: false,
        poolBudgetBytes: 384 * 1024 * 1024,
        maxCaptureSize: 1024,
        defaultCaptureSize: 256,
        preferCheapestModel: onSlowConnection(),
      };

  return cached;
}
