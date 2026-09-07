import type { ControlSpec } from '../model/format';
import { DreamNetTf } from './dreamNetTf';
import { exportDnw, styleControls } from './exportDnw';
import { loadFeatureNetwork, type FeatureNetworkId } from './featureNetworks';
import { gramMatrix, interior, mixture, shiftImage, styleLoss, totalVariation } from './losses';
import { initializeTraining, tf } from './tfSetup';

/**
 * Style transfer training, in the page.
 *
 * This is the same method as `train/style.py` — Johnson et al. 2016, fitting a feed-forward network
 * to a style image's Gram statistics rather than to a set of target pictures — with the budget of a
 * browser rather than of an afternoon on a GPU. The differences are all consequences of that:
 * a smaller network by default, smaller crops, and a step count in the hundreds rather than the
 * thousands. It converges to something recognisably the same effect, a bit rougher.
 *
 * What it produces is not a second-class artifact. The export goes through the same `.dnw` writer,
 * runs on the same WebGL kernels at the same speed, and can be downloaded, sent to someone else, or
 * dropped into `public/models/` and deployed.
 */

export interface TrainingConfig {
  featureNetwork: FeatureNetworkId;
  iterations: number;
  batch: number;
  /** Side of the square crops trained on. Smaller is faster and coarser. */
  cropSize: number;
  /**
   * Short side the style images are resized to before their statistics are taken.
   *
   * The most consequential control here, and not a quality setting: the network learns strokes at
   * the size they appear in pixels, so shrinking the style image gives finer, denser motifs and
   * enlarging it gives coarser, bolder ones.
   */
  styleSize: number;
  styleWeight: number;
  contentWeight: number;
  tvWeight: number;
  /** Weight on shift equivariance, which is what stops the effect boiling when the camera moves. */
  warpWeight: number;
  learningRate: number;
  width: number;
  blocks: number;
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  featureNetwork: 'mobilenet',
  iterations: 600,
  batch: 2,
  cropSize: 128,
  styleSize: 256,
  styleWeight: 12,
  contentWeight: 1,
  tvWeight: 2e-3,
  warpWeight: 0.3,
  learningRate: 2e-3,
  width: 8,
  blocks: 3,
};

export type TrainingPhase = 'loading-network' | 'reading-styles' | 'training' | 'exporting' | 'done' | 'stopped';

export interface TrainingProgress {
  phase: TrainingPhase;
  step: number;
  total: number;
  elapsedMs: number;
  losses: { content: number; style: number; tv: number; warp: number };
  /** A preview of the current network on a fixed frame, refreshed occasionally. */
  preview?: ImageData;
  message?: string;
}

export interface TrainingResult {
  buffer: ArrayBuffer;
  name: string;
  controls: ControlSpec[];
  bytes: number;
  steps: number;
  elapsedMs: number;
}

/**
 * What the trainer accepts as an image. Narrower than `TexImageSource` because TFJS cannot read an
 * `OffscreenCanvas`, and finding that out at runtime is worse than not offering it.
 */
export type TrainingImage = HTMLCanvasElement | HTMLImageElement | HTMLVideoElement | ImageBitmap | ImageData;

export interface StyleInput {
  name: string;
  image: TrainingImage;
}

export interface TrainingRequest {
  styles: StyleInput[];
  /** Frames to repaint. Their content barely matters; they only have to be things worth keeping recognisable. */
  contentFrames: TrainingImage[];
  modelName: string;
  config: TrainingConfig;
  onProgress: (progress: TrainingProgress) => void;
  signal: AbortSignal;
}

/** How often the loss terms are read back and a preview rendered. Each costs a GPU sync. */
const REPORT_EVERY = 10;
const PREVIEW_EVERY = 40;

/** Largest shift used by the equivariance term, and the margin excluded from its comparison. */
const MAX_SHIFT = 3;
const COMPARE_MARGIN = MAX_SHIFT + 2;

function toUnitTensor(source: TrainingImage): tf.Tensor3D {
  return tf.tidy(() => tf.browser.fromPixels(source).toFloat().div(255) as tf.Tensor3D);
}

/** Resizes so the short side is `shortSide`, which is how pattern scale is set. */
function resizeShortSide(image: tf.Tensor3D, shortSide: number): tf.Tensor3D {
  return tf.tidy(() => {
    const [height, width] = image.shape;
    const scale = shortSide / Math.min(height, width);
    return tf.image.resizeBilinear(image, [
      Math.max(1, Math.round(height * scale)),
      Math.max(1, Math.round(width * scale)),
    ]) as tf.Tensor3D;
  });
}

/** Scales up anything smaller than the crop, so every frame can supply one. */
function ensureAtLeast(image: tf.Tensor3D, size: number): tf.Tensor3D {
  const [height, width] = image.shape;
  if (height >= size && width >= size) return image;
  const scale = size / Math.min(height, width);
  const resized = tf.tidy(
    () => tf.image.resizeBilinear(image, [
      Math.max(size, Math.round(height * scale)),
      Math.max(size, Math.round(width * scale)),
    ]) as tf.Tensor3D,
  );
  image.dispose();
  return resized;
}

/**
 * Control vectors covering the corners, the pairs, and the interior of the slider space.
 *
 * Corners — one slider up, the rest down — are what a user reaches for most, so they take half the
 * samples. The rest matter because a network trained only on corners has never been asked what a
 * half-raised slider means, and will answer with an artifact.
 */
function sampleControls(batch: number, styles: number): tf.Tensor2D {
  const rows: number[][] = [];
  for (let row = 0; row < batch; row++) {
    const values = new Array<number>(styles).fill(0);
    const draw = Math.random();
    if (draw < 0.5 || styles === 1) {
      values[Math.floor(Math.random() * styles)] = 1;
    } else if (draw < 0.7) {
      const first = Math.floor(Math.random() * styles);
      let second = Math.floor(Math.random() * styles);
      if (second === first) second = (second + 1) % styles;
      values[first] = Math.random();
      values[second] = Math.random();
    } else {
      for (let column = 0; column < styles; column++) values[column] = Math.random();
    }
    rows.push(values);
  }
  return tf.tensor2d(rows, [batch, styles]);
}

export async function trainStyleModel(request: TrainingRequest): Promise<TrainingResult> {
  const { styles, contentFrames, config, onProgress, signal } = request;

  if (styles.length === 0) throw new Error('Pick at least one style image.');
  if (contentFrames.length === 0) throw new Error('Capture or add at least one content frame.');

  const began = performance.now();
  const report = (phase: TrainingPhase, step: number, extra: Partial<TrainingProgress> = {}) =>
    onProgress({
      phase,
      step,
      total: config.iterations,
      elapsedMs: performance.now() - began,
      losses: { content: 0, style: 0, tv: 0, warp: 0 },
      ...extra,
    });

  report('loading-network', 0, { message: 'Starting the training backend…' });
  const backend = await initializeTraining();

  report('loading-network', 0, { message: `Loading the feature network on ${backend.name}…` });
  const network = await loadFeatureNetwork(config.featureNetwork);

  report('reading-styles', 0, { message: 'Reading style statistics…' });

  // Style targets are computed once. After this the style images are never looked at again.
  const styleTargets = new Map<string, tf.Tensor3D>();
  for (const layer of network.styleLayers) {
    const perStyle: tf.Tensor3D[] = [];
    for (const style of styles) {
      const gram = tf.tidy(() => {
        const image = resizeShortSide(toUnitTensor(style.image), config.styleSize);
        const batched = network.preprocess(image.expandDims(0) as tf.Tensor4D);
        const activation = network.activations(batched, [layer])[0];
        return gramMatrix(activation);
      });
      perStyle.push(gram);
    }
    styleTargets.set(layer, tf.tidy(() => tf.concat(perStyle, 0) as tf.Tensor3D));
    for (const gram of perStyle) gram.dispose();
  }

  const frames = contentFrames.map((frame) => ensureAtLeast(toUnitTensor(frame), config.cropSize));
  const previewFrame = tf.tidy(() => {
    const source = frames[0];
    const [height, width] = source.shape;
    return source.slice(
      [Math.floor((height - config.cropSize) / 2), Math.floor((width - config.cropSize) / 2), 0],
      [config.cropSize, config.cropSize, 3],
    ).expandDims(0) as tf.Tensor4D;
  });

  const model = new DreamNetTf({
    width: config.width,
    blocks: config.blocks,
    filmHidden: Math.max(8, 8 * styles.length),
    condDims: styles.length,
  });
  const optimizer = tf.train.adam(config.learningRate);

  const terms = { content: 0, style: 0, tv: 0, warp: 0 };
  let capture = false;
  let stoppedAt = config.iterations;

  const cropBatch = (): tf.Tensor4D =>
    tf.tidy(() => {
      const crops: tf.Tensor3D[] = [];
      for (let i = 0; i < config.batch; i++) {
        const frame = frames[Math.floor(Math.random() * frames.length)];
        const [height, width] = frame.shape;
        const top = Math.floor(Math.random() * (height - config.cropSize + 1));
        const left = Math.floor(Math.random() * (width - config.cropSize + 1));
        const crop = frame.slice([top, left, 0], [config.cropSize, config.cropSize, 3]) as tf.Tensor3D;
        crops.push(Math.random() < 0.5 ? (tf.reverse(crop, 1) as tf.Tensor3D) : crop);
      }
      return tf.stack(crops) as tf.Tensor4D;
    });

  try {
    for (let step = 0; step < config.iterations; step++) {
      if (signal.aborted) {
        stoppedAt = step;
        break;
      }

      capture = step % REPORT_EVERY === 0;

      const content = cropBatch();
      const controls = sampleControls(config.batch, styles.length);
      const { direction, mass } = mixture(controls);
      const shiftY = Math.round((Math.random() * 2 - 1) * MAX_SHIFT);
      const shiftX = Math.round((Math.random() * 2 - 1) * MAX_SHIFT);

      optimizer.minimize(() => {
        const prediction = model.forward(content, controls);

        // One pass of the feature network covering both images: the prediction, whose gradient is
        // wanted, and the content frame, which is only a target.
        const both = network.preprocess(tf.concat([prediction, content], 0) as tf.Tensor4D);
        const wanted = [...new Set([...network.styleLayers, network.contentLayer])];
        const activations = network.activations(both, wanted);
        const byName = new Map(wanted.map((name, index) => [name, activations[index]]));

        const contentActivation = byName.get(network.contentLayer)!;
        const [predictedContent, targetContent] = tf.split(contentActivation, 2, 0) as tf.Tensor4D[];
        const contentTerm = predictedContent.sub(targetContent).square().mean() as tf.Scalar;

        let styleTerm = tf.scalar(0);
        for (const layer of network.styleLayers) {
          const [predicted] = tf.split(byName.get(layer)!, 2, 0) as tf.Tensor4D[];
          styleTerm = styleTerm.add(
            styleLoss(gramMatrix(predicted), styleTargets.get(layer)!, direction, mass),
          ) as tf.Scalar;
        }

        const tvTerm = totalVariation(prediction);

        let warpTerm = tf.scalar(0);
        if (config.warpWeight > 0) {
          // f(shift(x)) against shift(f(x)). A convolutional network is already equivariant to
          // translation except at its stride boundaries, and the stride-2 downsamples here are
          // exactly where that breaks — which is what makes detail crawl when the camera moves by
          // an odd number of pixels.
          const shiftedInput = shiftImage(content, shiftY, shiftX);
          const shiftedPrediction = shiftImage(prediction, shiftY, shiftX);
          const fromShifted = model.forward(shiftedInput, controls);
          warpTerm = interior(fromShifted, COMPARE_MARGIN)
            .sub(interior(shiftedPrediction, COMPARE_MARGIN))
            .square()
            .mean() as tf.Scalar;
        }

        const total = contentTerm
          .mul(config.contentWeight)
          .add(styleTerm.mul(config.styleWeight))
          .add(tvTerm.mul(config.tvWeight))
          .add(warpTerm.mul(config.warpWeight)) as tf.Scalar;

        if (capture) {
          // Reading a value does not detach anything from the tape; it just costs a GPU sync, which
          // is why it happens every tenth step rather than every step.
          terms.content = contentTerm.dataSync()[0];
          terms.style = styleTerm.dataSync()[0];
          terms.tv = tvTerm.dataSync()[0];
          terms.warp = warpTerm.dataSync()[0];
        }

        return total;
      }, false, model.trainableVariables);

      content.dispose();
      controls.dispose();
      direction.dispose();
      mass.dispose();

      if (step % REPORT_EVERY === 0 || step === config.iterations - 1) {
        const preview =
          step % PREVIEW_EVERY === 0 || step === config.iterations - 1
            ? await renderPreview(model, previewFrame, styles.length)
            : undefined;
        report('training', step + 1, { losses: { ...terms }, preview });
      }

      // Hands the frame back to the browser. Without this the page is frozen for the whole run and
      // the Stop button cannot be clicked.
      await tf.nextFrame();
    }

    report('exporting', stoppedAt, { losses: { ...terms }, message: 'Packing the weights…' });

    const buffer = exportDnw(model, {
      name: request.modelName,
      description: `Trained in the browser on ${styles.map((style) => style.name).join(', ')}.`,
      controls: styleControls(styles.map((style) => style.name)),
      trainedAt: config.cropSize,
      provenance: {
        featureNetwork: config.featureNetwork,
        iterations: stoppedAt,
        styleSize: config.styleSize,
        styleWeight: config.styleWeight,
      },
    });

    report(signal.aborted ? 'stopped' : 'done', stoppedAt, { losses: { ...terms } });

    return {
      buffer,
      name: request.modelName,
      controls: styleControls(styles.map((style) => style.name)),
      bytes: buffer.byteLength,
      steps: stoppedAt,
      elapsedMs: performance.now() - began,
    };
  } finally {
    optimizer.dispose();
    model.dispose();
    previewFrame.dispose();
    for (const frame of frames) frame.dispose();
    for (const target of styleTargets.values()) target.dispose();
  }
}

/** Renders the current network on the fixed preview frame, with every style raised. */
async function renderPreview(model: DreamNetTf, frame: tf.Tensor4D, styles: number): Promise<ImageData> {
  const pixels = tf.tidy(() => {
    const controls = tf.ones([1, styles]).div(styles) as tf.Tensor2D;
    return model.forward(frame, controls).squeeze([0]).clipByValue(0, 1) as tf.Tensor3D;
  });
  try {
    const [height, width] = pixels.shape;
    const bytes = await tf.browser.toPixels(pixels);
    return new ImageData(bytes, width, height);
  } finally {
    pixels.dispose();
  }
}
