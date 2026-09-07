import * as mobilenetLib from '@tensorflow-models/mobilenet';
import { tf } from './tfSetup';

/**
 * The frozen classifier whose features define what "style" means.
 *
 * Style transfer never compares pixels. It compares the Gram matrix of some network's intermediate
 * activations — which features fire together, averaged over the frame with position discarded — so
 * the choice of network is a choice about what kind of thing the loss can see at all. These are the
 * same two the dream app offers, with the same tradeoff.
 */

export type FeatureNetworkId = 'mobilenet' | 'vgg19';

export interface FeatureNetworkOption {
  id: FeatureNetworkId;
  label: string;
  description: string;
  downloadLabel: string;
}

export const FEATURE_NETWORKS: FeatureNetworkOption[] = [
  {
    id: 'mobilenet',
    label: 'MobileNet V2',
    description: 'Fast to fetch and to train against. Style comes out softer and blotchier.',
    downloadLabel: '~14 MB',
  },
  {
    id: 'vgg19',
    label: 'VGG-19',
    description: 'The network classic style transfer was built on. Much stronger style, several times slower.',
    downloadLabel: '~80 MB',
  },
];

export interface FeatureNetwork {
  id: FeatureNetworkId;
  styleLayers: string[];
  contentLayer: string;
  /** Turns a batch of [0,1] NHWC images into what this network expects. Stays differentiable. */
  preprocess(images01: tf.Tensor4D): tf.Tensor4D;
  /** Runs the network and returns the requested activations, in the order asked for. */
  activations(input: tf.Tensor4D, names: string[]): tf.Tensor4D[];
}

/**
 * Keras' `VGG19(include_top=False)` converted to TF.js — the convolutional trunk only, so ~80 MB
 * rather than the ~500 MB the fully-connected classifier would add. Feature extraction never
 * touches those layers. Pinned to a commit so the weights cannot change underneath, and served
 * through jsDelivr, which fronts GitHub with CORS headers and a CDN cache.
 */
const VGG19_MODEL_URL =
  'https://cdn.jsdelivr.net/gh/paulsp94/tfjs_vgg19_imagenet@ea6bea2ac90e492592b552346e02abcb0eafa443/model/model.json';

/**
 * The layer set from Gatys et al.: one convolution from each block. Block 1 carries colour and fine
 * grain, block 5 carries large compositional structure, and the Gram matrices of all five together
 * are what make a style read as a coherent medium rather than a texture swatch.
 */
const VGG19_STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'];

/** Also from Gatys et al.: deep enough to pin down layout without dictating local texture. */
const VGG19_CONTENT_LAYER = 'block4_conv2';

/**
 * VGG's early blocks are enormous — a 320px input holds tens of megabytes in block1_conv1 alone,
 * and the gradient tape doubles it. Training crops are smaller than this; the cap is insurance.
 */
const VGG19_MAX_INPUT = 320;

/** Keras' "caffe" preprocessing: [0,1] RGB to [0,255] BGR, minus the ImageNet channel means. */
const VGG_MEAN_BGR = [103.939, 116.779, 123.68];

/**
 * This MobileNetV2 build declares a fixed 224x224 graph input, so unlike VGG it cannot take the
 * training crop at its own size — everything is resampled to 224 on the way in. That is real work
 * when training at 128, and part of why it is the faster-but-softer option rather than a free win.
 */
const MOBILENET_INPUT = 224;

/** Where in the layer stack to take statistics from, as fractions of network depth. */
const MOBILENET_STYLE_FRACTIONS = [0.05, 0.2, 0.4, 0.6, 0.8];
const MOBILENET_CONTENT_FRACTION = 0.5;

interface GraphNode {
  name: string;
  op: string;
}

interface IntrospectableGraphModel {
  executor?: { graph?: { nodes?: Record<string, GraphNode> } };
}

async function loadMobilenet(): Promise<FeatureNetwork> {
  const net = await mobilenetLib.load({ version: 2, alpha: 1.0 });
  const graphModel = (net as unknown as { model: tf.GraphModel }).model;

  const nodes = (graphModel as unknown as IntrospectableGraphModel).executor?.graph?.nodes ?? {};
  const activationNames = Object.keys(nodes).filter((name) => {
    const op = nodes[name]?.op ?? '';
    return /relu/i.test(name) || /relu/i.test(op);
  });

  if (activationNames.length === 0) {
    throw new Error('Could not find any activation layers in the MobileNet graph.');
  }

  const pick = (fraction: number) =>
    activationNames[Math.min(activationNames.length - 1, Math.round(fraction * (activationNames.length - 1)))];

  return {
    id: 'mobilenet',
    styleLayers: [...new Set(MOBILENET_STYLE_FRACTIONS.map(pick))],
    contentLayer: pick(MOBILENET_CONTENT_FRACTION),
    preprocess(images01) {
      // This build takes its input already in [0,1], so the only work is the resize.
      return tf.image.resizeBilinear(images01, [MOBILENET_INPUT, MOBILENET_INPUT]) as tf.Tensor4D;
    },
    activations(input, names) {
      const result = graphModel.execute(input, names);
      return (Array.isArray(result) ? result : [result]) as tf.Tensor4D[];
    },
  };
}

async function loadVgg19(): Promise<FeatureNetwork> {
  const model = await tf.loadLayersModel(VGG19_MODEL_URL);

  // VGG is a straight chain, so a sub-model that stops at the deepest layer asked for skips
  // everything past it. Caching by the requested name set builds each one once, not once per step.
  const subModels = new Map<string, tf.LayersModel>();
  const subModelFor = (names: string[]): tf.LayersModel => {
    const key = names.join('|');
    let sub = subModels.get(key);
    if (!sub) {
      sub = tf.model({
        inputs: model.inputs,
        outputs: names.map((name) => model.getLayer(name).output as tf.SymbolicTensor),
      });
      subModels.set(key, sub);
    }
    return sub;
  };

  return {
    id: 'vgg19',
    styleLayers: VGG19_STYLE_LAYERS,
    contentLayer: VGG19_CONTENT_LAYER,
    preprocess(images01) {
      const [, height, width] = images01.shape;
      const longest = Math.max(height, width);
      const scaled =
        longest > VGG19_MAX_INPUT
          ? (tf.image.resizeBilinear(images01, [
              Math.max(1, Math.round((height * VGG19_MAX_INPUT) / longest)),
              Math.max(1, Math.round((width * VGG19_MAX_INPUT) / longest)),
            ]) as tf.Tensor4D)
          : images01;
      return tf.reverse(scaled.mul(255), -1).sub(VGG_MEAN_BGR) as tf.Tensor4D;
    },
    activations(input, names) {
      const result = subModelFor(names).predict(input);
      return (Array.isArray(result) ? result : [result]) as tf.Tensor4D[];
    },
  };
}

const loaders = new Map<FeatureNetworkId, Promise<FeatureNetwork>>();

/**
 * Loads and thereafter reuses a feature network. A failed load drops the cached promise, so trying
 * again actually retries rather than replaying the same error forever.
 */
export function loadFeatureNetwork(id: FeatureNetworkId): Promise<FeatureNetwork> {
  let promise = loaders.get(id);
  if (!promise) {
    promise = (id === 'vgg19' ? loadVgg19() : loadMobilenet()).catch((error) => {
      loaders.delete(id);
      throw error;
    });
    loaders.set(id, promise);
  }
  return promise;
}
