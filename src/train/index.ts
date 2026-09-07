/**
 * The entry point the app dynamically imports.
 *
 * Everything the trainer needs — TensorFlow.js, the WebGPU backend, MobileNet — is several
 * megabytes of JavaScript that the viewing path has no use for. Keeping the whole training module
 * behind one dynamic import means a visitor who only ever watches the filter never downloads any
 * of it, and Vite splits it into its own chunk on that basis.
 */

export { initializeTraining, type TrainingBackend } from './tfSetup';
export { FEATURE_NETWORKS, type FeatureNetworkId, type FeatureNetworkOption } from './featureNetworks';
export {
  trainStyleModel,
  DEFAULT_TRAINING_CONFIG,
  type TrainingConfig,
  type TrainingImage,
  type TrainingProgress,
  type TrainingResult,
  type StyleInput,
} from './trainer';
export { captureFromSource, loadImageFile, snapshot, DEFAULT_FRAME_COUNT } from './captureFrames';

/** Reached into only by the smoke test, which needs to build a network without training one. */
export * as __internals from './internals';

// Saved-model storage deliberately does NOT live behind this import. It has no TensorFlow
// dependency, and the model list has to be readable on a page load that never opens the trainer.
// It is in `src/model/storage.ts`.
