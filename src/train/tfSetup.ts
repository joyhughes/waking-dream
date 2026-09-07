import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

/**
 * TensorFlow.js setup, used only by the in-browser trainer.
 *
 * Training needs automatic differentiation, which the hand-written WebGL2 runtime in `src/gpu/`
 * deliberately does not have — it only ever runs forward, which is what makes it small and fast.
 * So the trainer brings its own framework. TFJS gets its own GPU context alongside the runtime's;
 * that is fine, but it is why the engine is paused during a training run rather than the two of
 * them competing for the same GPU sixty times a second.
 *
 * Everything here sits behind a dynamic import, so a visitor who never opens the Train panel never
 * downloads any of it.
 */

export interface TrainingBackend {
  name: string;
  webgpu: boolean;
}

let readyPromise: Promise<TrainingBackend> | null = null;

/**
 * Whether WebGPU is actually usable, not merely present.
 *
 * `navigator.gpu` existing says nothing about there being a device behind it — a headless browser,
 * a VM, or a machine whose GPU is blocklisted all expose the object and then hand back a null
 * adapter. TFJS's own initialization walks straight into that and throws from inside its backend
 * factory, which is a much harder failure to fall back from than asking first.
 */
async function webgpuUsable(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
  try {
    const adapter = await (navigator as Navigator & { gpu: GPU }).gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

async function selectBackend(): Promise<TrainingBackend> {
  if (await webgpuUsable()) {
    try {
      await tf.setBackend('webgpu');
      await tf.ready();
      if (tf.getBackend() === 'webgpu') return { name: tf.getBackend(), webgpu: true };
    } catch (error) {
      console.warn('WebGPU backend failed to initialize for training, falling back to WebGL.', error);
    }
  }

  await tf.setBackend('webgl');
  await tf.ready();
  return { name: tf.getBackend(), webgpu: false };
}

export function initializeTraining(): Promise<TrainingBackend> {
  readyPromise ??= selectBackend();
  return readyPromise;
}

export { tf };
