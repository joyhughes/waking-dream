import { tf } from './tfSetup';

/**
 * The loss terms, all measured on a frozen network's features rather than on pixels.
 *
 * Comparing pixels asks for the wrong thing: two frames can differ everywhere in pixels and be the
 * same picture, and where many answers are equally right the safest way to minimize a pixel loss is
 * to average them, which is a blur. Features are invariant to exactly the differences that do not
 * matter.
 */

/**
 * Normalized Gram matrix of an `[N, H, W, C]` activation.
 *
 * The Gram matrix records which features co-occur, averaged over the frame with position discarded
 * entirely — the vocabulary of strokes, colours and textures, with no opinion about where any of it
 * goes. Dividing by `C * H * W` is what makes a style weight mean the same thing at one resolution
 * as at another; without it the loss silently changes magnitude whenever the crop size changes.
 */
export function gramMatrix(activation: tf.Tensor4D): tf.Tensor3D {
  return tf.tidy(() => {
    const [batch, height, width, channels] = activation.shape;
    const flat = activation.reshape([batch, height * width, channels]) as tf.Tensor3D;
    return tf.matMul(flat, flat, true, false).div(channels * height * width) as tf.Tensor3D;
  });
}

/** Mean squared gradient magnitude, penalizing the high-frequency noise a style loss will otherwise accept. */
export function totalVariation(images: tf.Tensor4D): tf.Scalar {
  return tf.tidy(() => {
    const [, height, width, channels] = images.shape;
    if (height < 2 || width < 2) return tf.scalar(0);
    const dy = images.slice([0, 1, 0, 0], [-1, height - 1, width, channels])
      .sub(images.slice([0, 0, 0, 0], [-1, height - 1, width, channels]));
    const dx = images.slice([0, 0, 1, 0], [-1, height, width - 1, channels])
      .sub(images.slice([0, 0, 0, 0], [-1, height, width - 1, channels]));
    return dy.square().mean().add(dx.square().mean()) as tf.Scalar;
  });
}

/**
 * Translates a batch by whole pixels, filling from the edge.
 *
 * The equivariance term needs the same picture in two places. A wrapping roll would bring the
 * opposite edge around, which is not a translation of the scene and would teach the network to
 * expect content that is not there; replicating the edge is what a camera pan actually looks like
 * at the frame boundary.
 */
export function shiftImage(images: tf.Tensor4D, dy: number, dx: number): tf.Tensor4D {
  if (dy === 0 && dx === 0) return images.clone();
  return tf.tidy(() => {
    const padY = Math.abs(dy);
    const padX = Math.abs(dx);
    const [batch, height, width, channels] = images.shape;

    const top = images.slice([0, 0, 0, 0], [batch, 1, width, channels]).tile([1, padY, 1, 1]);
    const bottom = images.slice([0, height - 1, 0, 0], [batch, 1, width, channels]).tile([1, padY, 1, 1]);
    const vertical = tf.concat([top, images, bottom], 1) as tf.Tensor4D;

    const tall = vertical.shape[1];
    const left = vertical.slice([0, 0, 0, 0], [batch, tall, 1, channels]).tile([1, 1, padX, 1]);
    const right = vertical.slice([0, 0, width - 1, 0], [batch, tall, 1, channels]).tile([1, 1, padX, 1]);
    const padded = tf.concat([left, vertical, right], 2) as tf.Tensor4D;

    return padded.slice([0, padY - dy, padX - dx, 0], [batch, height, width, channels]) as tf.Tensor4D;
  });
}

/**
 * Trims a margin off every side.
 *
 * The equivariance comparison is only meaningful away from the frame edge. Near it, the two images
 * being compared genuinely saw different things — one had real content where the other had
 * replicated edge — and demanding they agree there teaches the network to blur its borders.
 */
export function interior(images: tf.Tensor4D, margin: number): tf.Tensor4D {
  const [batch, height, width, channels] = images.shape;
  if (margin <= 0 || height <= 2 * margin || width <= 2 * margin) return images;
  return images.slice([0, margin, margin, 0], [batch, height - 2 * margin, width - 2 * margin, channels]) as tf.Tensor4D;
}

/**
 * Splits a control vector into a direction on the simplex and a total mass.
 *
 * The network is conditioned on the raw slider values, but the loss needs one target, so the vector
 * is read as "which mixture" and "how much of it". Two consequences, both of them things a slider
 * ought to do: every slider at zero is trained to reproduce the input, so zero is a real off
 * position rather than an untrained corner; and two sliders at 1 asks for the same blend as two at
 * 0.5, so raising everything does not walk off the end of what was trained.
 */
export function mixture(controls: tf.Tensor2D): { direction: tf.Tensor2D; mass: tf.Tensor1D } {
  return tf.tidy(() => {
    const mass = controls.sum(1) as tf.Tensor1D;
    const direction = controls.div(mass.expandDims(1).clipByValue(1e-6, Number.MAX_VALUE)) as tf.Tensor2D;
    return { direction: tf.keep(direction), mass: tf.keep(mass.clipByValue(0, 1) as tf.Tensor1D) };
  });
}

/**
 * Style loss against a per-example mixture of the style targets.
 *
 * `targets` holds one Gram matrix per style, stacked. The mixture each example asked for is formed
 * by a matrix multiply against its direction vector, so a batch can be training several different
 * blends at once rather than one style at a time.
 */
export function styleLoss(
  predictedGram: tf.Tensor3D,
  targets: tf.Tensor3D,
  direction: tf.Tensor2D,
  mass: tf.Tensor1D,
): tf.Scalar {
  return tf.tidy(() => {
    const [styles, channels] = [targets.shape[0], targets.shape[1]];
    const flat = targets.reshape([styles, channels * channels]) as tf.Tensor2D;
    const mixed = direction.matMul(flat).reshape([direction.shape[0], channels, channels]) as tf.Tensor3D;
    const perExample = predictedGram.sub(mixed).square().mean([1, 2]) as tf.Tensor1D;
    return perExample.mul(mass).mean() as tf.Scalar;
  });
}
