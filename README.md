# DreamNet

Real-time DeepDream-style filtering in the browser: camera, video files, or a single still image fed
back into itself, running through hand-written WebGL2 convolution kernels at video rate.

Classic DeepDream is slow for a structural reason. It ascends the gradient of a deep network's
activations, and each ascent step needs a forward and a backward pass through that network — so a
single frame costs hundreds of passes. This repo takes the other route: train a small feed-forward
network **offline** to reproduce what the slow one produces, then run one pass per frame. Same
effect, no gradient ascent at runtime.

There are two ways to get a model, and two processors in the app.

| | Trained DreamNet | Shallow ascent |
|---|---|---|
| Needs a model file | yes, from `train/` | no, works immediately |
| What it does | one forward pass of a trained network | real gradient ascent on a one-layer filter bank |
| What it draws | a pattern you supplied, or a distilled DeepDream | oriented texture, ridges, cells |
| Why it exists | the point of the project | works on day one, and stays as the control |

A trained model comes from one of three paths:

- **The Train panel, in the app itself.** Drop in a style image, capture some frames from whatever
  source is running, and train — no Python, no clone. Uses MobileNet V2 or VGG-19 for its style
  loss, the same pair and the same tradeoff as the dream app. The whole trainer is behind a dynamic
  import, so nobody who does not open the panel downloads any of it.

The other two are in [`train/`](train/README.md), on your own machine with PyTorch:

- **[`style.py`](train/README.md#style-transfer-training-with-your-own-patterns)** — drop images in
  `train/styles/` and it learns to paint like them. No dataset build, because the target is a
  *statistic* of the style image rather than a set of pictures, so a run starts the moment you point
  it at a folder. Each style image becomes its own slider in the app, and the sliders blend.
- **[`train.py`](train/README.md#distilling-deepdream)** — distills your slow DeepDream. Costs hours
  of teacher runs up front, and is the only way to get the semantic hallucination that comes out of
  a deep network's own gradients.

All three write the same `.dnw` and run through the same WebGL kernels at the same speed. A model
trained in the browser is not a lesser artifact than one trained in PyTorch — just a rougher one,
for having had a few hundred steps instead of a few thousand.

## Saving, sharing, and getting back to a look

**Frames save as PNG with the whole parameter set written into them**, in a `tEXt` chunk that every
other decoder skips — so the file stays an ordinary image that Preview and Photos open, and is also
a way back to the settings that made it. Open one with **Settings from image…**, or just open it as
an image source and the app offers to apply what it finds. Send someone a frame and they land
exactly where you were.

**Recordings are H.264 MP4** wherever the browser can manage it, falling back to WebM where it
cannot. This matters on a Mac: almost nothing outside a browser opens WebM, so a WebM recording
arrives as a file the machine will not play, while MP4 opens in QuickTime, Photos and Final Cut.

**Models** can be downloaded as `.dnw` and sent to anyone — they load them with *Load .dnw model…*.
Models trained in the browser can also be saved to it, and then list alongside the shipped ones.
To ship one with a deployed build, drop it in `public/models/` and run `pnpm models:index`.

The shallow mode is not a placeholder effect. It is genuine activation maximization: the gradient of
`mean(relu(W * x))` with respect to `x` is exactly a convolution by the flipped, transposed kernel,
so with a one-layer network the backward pass is just another convolution and the whole ascent —
octaves, gradient normalization, feedback recursion — runs at frame rate with no training at all.
What it cannot do is hallucinate anything semantic, because a single layer of oriented filters has
no semantics in it. The difference between the two modes is exactly what the teacher's depth bought.

## Running it

```bash
pnpm install
pnpm dev            # http://localhost:5173
```

Pick **Camera**, **Video…**, **Image…**, or **Test pattern** and it starts. No model file is needed
for shallow mode.

Models in `public/models/` are listed in `index.json`, which the app reads at startup — it lists them
all and loads the first automatically, so a deployed build opens already running a real model rather
than asking whoever opened it to train one. `train/export.py` maintains that manifest as part of
every export, and the models are tracked in git, so a model is live on the next reload and ships
with `pnpm build` with no further step.

`/selftest.html` runs the numerical self-test: every GPU op against a CPU reference, a full network
forward pass against a CPU implementation of the same op list, the TFJS trainer's network against
the WebGL runtime, and a PNG parameter round trip. `/traintest.html` runs a short real training run
end to end.

## The size/speed dial

Capture size is the resolution the network actually sees, independent of the display. Cost is close
to quadratic in it, so this is the main lever. The panel has presets, a slider, and a **Sweep
capture sizes** button that measures the whole curve on your machine with the current configuration
and prints milliseconds and frames per second for each size.

The sweep serializes each frame with `glFinish` before reading the clock, so its numbers run a
little pessimistic against the live counter — the live loop overlaps CPU and GPU across frames and
the sweep deliberately does not, because otherwise the time cannot be attributed to one frame.

The live readout shows GPU milliseconds where the driver exposes `EXT_disjoint_timer_query_webgl2`,
and falls back to CPU submit time where it does not. Frame rate alone will not tell you where the
headroom went: a pipeline using 3ms and one using 15ms both read as 60fps.

## Feedback

`output(t) = DreamNet(α · frame(t) + β · warp(output(t − 1)))`

The warp — a slow zoom, a slight rotation, optional drift — is what makes this a dream rather than a
filter. With no warp, hallucinated detail lands on the same pixels every frame and simply saturates.
Push it outward a little each frame and the centre is continuously vacated for new detail to grow
into, which is the recursion the original DeepDream zoom videos were made of.

The **Fade** control damps the recursion by decaying the fed-back frame toward mid grey rather than
toward black, so it bleeds off accumulated contrast without darkening the picture. Below about 0.97
the loop settles into a steady state instead of running away; at 1.0 it will saturate.

Everything stays on the GPU. The previous frame's output is still a texture when it re-enters the
network, so feeding it back costs a texture sample rather than a round trip through CPU memory —
which is what makes the recursion affordable at all.

## How the runtime works

```
src/gpu/
  gl.ts        context, capability probe
  tensor.ts    NHWC feature maps as TEXTURE_2D_ARRAY, four channels per layer, plus a pool
  program.ts   shader cache, and binding array-texture layers as colour attachments
  shaders.ts   the GLSL generators — conv, instance norm, reduction, resize, warp, composite
  ops.ts       the ops themselves
  debug.ts     readback, self-test only
src/model/
  format.ts        the .dnw container
  dreamnet.ts      executes a trained model's op list
  conditioning.ts  the control MLP, evaluated on the CPU each frame
  shallowDream.ts  the training-free ascent mode
src/pipeline/
  sources.ts   camera, video file, still image, test pattern
  engine.ts    the frame loop, feedback, capture sizing, the benchmark
```

Three decisions carry most of the design:

**Channels live across array-texture layers, four to a layer.** A convolution reads a whole channel
group with one `texelFetch` and writes up to four groups per draw using multiple render targets, so
each input texel read serves sixteen multiply-adds instead of four. Activations are `RGBA16F`; the
precision is far below what the output tanh and the eventual 8-bit display would keep, and half the
bandwidth is what a fragment-shader convolution is actually limited by.

**Instance-norm statistics never touch the CPU.** A two-stage reduction folds the feature map down
to one texel per channel group, and the normalization shader samples that directly. A `readPixels`
here would stall the pipeline mid-forward-pass, once per normalization layer.

**Live controls come from conditional instance normalization, not from swapping weights.** The
convolution weights are fixed, but every normalization layer already applies a per-channel scale and
shift, and a small MLP on the CPU recomputes those from the control vector each frame. One trained
model therefore covers a continuum of dream configurations rather than a single point in it.
(Dumoulin et al. 2017 for conditional instance norm; Perez et al. 2018 for the FiLM framing.)

The runtime is WebGL2 because it works everywhere today, including Safari. `ops.ts` is the only
module that issues draws, so a WebGPU compute path can be added behind the same surface without
touching the model or pipeline layers.

## Training a model

Full detail in [`train/README.md`](train/README.md). To train on your own patterns:

```bash
cp ~/art/*.jpg train/styles/     # each image becomes its own slider
pnpm train -- --images ~/Pictures/photos --name patterns
```

That sets up the virtualenv on first run, trains, exports into `public/models/`, and rebuilds the
manifest. Reload the app; the model is in the list and already loaded.

`--style-size` is the control worth knowing about before any other: the network learns strokes at
the size they appear in pixels, so it sets how large the motifs come out, and it matters far more
than the loss weights do.

With `--reference` written, `/selftest.html` also runs the *exported* model against what PyTorch
computed for the same input. That is the only check that proves the exporter and the runtime agree
rather than merely both running — a weight packed in the wrong order produces a plausible picture,
not an error.

## Prior work this builds on

- Mordvintsev et al. 2015 — DeepDream itself, and the octave / Laplacian-normalization / jitter
  recipe `train/teacher.py` implements.
- Gatys et al. 2015 — that the Gram matrix of a network's features captures style.
- Johnson et al. 2016 — feed-forward image transformation with a perceptual loss, which is what makes
  style transfer a single pass. The architecture and both training losses come from here; the
  distillation path's change is that the target is a DeepDream frame rather than a style image.
- Odena et al. 2016 — why upsampling is nearest-then-convolve rather than a transposed convolution.
- Dumoulin et al. 2017, Perez et al. 2018 — conditioning a fixed network through its normalization
  layers, which is where the live sliders come from.
