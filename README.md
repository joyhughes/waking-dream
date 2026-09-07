# DreamNet

Real-time DeepDream-style filtering in the browser: camera, video files, or a single still image fed
back into itself, running through hand-written WebGL2 convolution kernels at video rate.

Classic DeepDream is slow for a structural reason. It ascends the gradient of a deep network's
activations, and each ascent step needs a forward and a backward pass through that network — so a
single frame costs hundreds of passes. This repo takes the other route: train a small feed-forward
network **offline** to reproduce what the slow one produces, then run one pass per frame. Same
effect, no gradient ascent at runtime.

There are two processors in the app, and they answer different questions.

| | Trained DreamNet | Shallow ascent |
|---|---|---|
| Needs a model file | yes, from `train/` | no, works immediately |
| What it does | one forward pass of a distilled network | real gradient ascent on a one-layer filter bank |
| What it draws | whatever the teacher drew — up to eyes and animals | oriented texture, ridges, cells |
| Why it exists | the point of the project | works on day one, and stays as the control |

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

`/selftest.html` runs the numerical self-test: every GPU op against a CPU reference, and a full
network forward pass against a CPU implementation of the same op list.

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

See [`train/README.md`](train/README.md). In short:

```bash
cd train
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python dataset.py --images ~/Pictures/photos --out data/pairs --count 2000
python train.py --data data/pairs --out runs/dreamnet --epochs 40
python export.py --checkpoint runs/dreamnet/checkpoint.pt \
    --out ../public/models/dreamnet.dnw --reference ../public/models/verify.json
```

Then **Load .dnw model…** in the app. With `--reference` written, `/selftest.html` also checks the
running WebGL model against what PyTorch computed for the same input — the only check that proves
the exporter and the runtime agree rather than merely both running.

## Prior work this builds on

- Mordvintsev et al. 2015 — DeepDream itself, and the octave / Laplacian-normalization / jitter
  recipe `train/teacher.py` implements.
- Johnson et al. 2016 — feed-forward image transformation with a perceptual loss. The student's
  architecture and the training loss both come from here; the change is that the target is a
  DeepDream frame rather than a style image.
- Odena et al. 2016 — why upsampling is nearest-then-convolve rather than a transposed convolution.
- Dumoulin et al. 2017, Perez et al. 2018 — conditioning a fixed network through its normalization
  layers, which is where the live sliders come from.
