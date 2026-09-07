# Training

Two paths produce the same kind of model file, run through the same runtime, and expose the same
kind of live sliders. They differ in what they teach the network to draw.

| | `style.py` | `train.py` |
|---|---|---|
| Teaches | a specific pattern you supply | DeepDream, distilled |
| Needs | style images + any photos | a generated dataset of dreamed pairs |
| Setup cost | none — starts immediately | hours of `dataset.py` first |
| Sliders it creates | one per pattern, and they blend | layer depth, strength, pattern scale |

If you want the network to paint like *this thing here*, use `style.py`. If you want the semantic
hallucination — the eyes and animals that only a deep network's own gradients produce — use the
distillation path.

## Setup

```bash
cd train
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Torch picks up Apple silicon's MPS backend on its own; `--device cpu` forces it off.

---

# Style transfer: training with your own patterns

Put images in [`styles/`](styles/README.md) — one file per pattern — then:

```bash
python style.py --images ~/Pictures/photos
```

That is the whole setup. There is no dataset build, because the target is not a set of images but a
*statistic* of your style images, and it can be evaluated on the fly against any photo. The photos
are not being learned from; they are only there to be things the network must keep recognisable
while it repaints them. A few hundred varied ones is plenty.

Expect a usable model in well under an hour on an M3, and something you can look at within minutes —
previews are written to `runs/<name>/previews/` throughout.

## What the loss is asking for

Both terms are measured on VGG-16 features rather than on pixels, because pixels ask for the wrong
thing: two frames can differ everywhere in pixels and be the same picture, and the safest way to
minimize a pixel loss over many equally-valid answers is to average them, which is a blur.

- **Style** — the Gram matrix at four layers: which features co-occur, with position thrown away
  entirely. The vocabulary of strokes, colours and textures, and no opinion at all about where any
  of it goes.
- **Content** — features at one layer, position by position. What is in the frame and where, with no
  opinion about the pixels that render it.
- **Total variation** — a small smoothness penalty. The style loss is perfectly happy to be
  satisfied by high-frequency noise with the right statistics, and on video that is different noise
  every frame, which reads as crawling grain.
- **Warp equivariance** — `f(warp(x))` should equal `warp(f(x))`. This is the temporal-stability
  term, and it is why the output does not boil when the camera moves. It needs neither optical flow
  nor video data. Set `--warp-weight 0` if you only ever process stills; it costs one extra forward
  pass per step.

## The three knobs that matter

**`--style-size` (default 384) is the most effective control and it is not a quality setting.** The
network learns strokes at the size they appear in pixels, so shrinking the style image makes it draw
finer, denser motifs and enlarging it makes them coarser and bolder — on the same content, at the
same capture size. If a style comes out as unreadable fine noise, this is almost always the cause
rather than the loss weights.

```bash
python style.py --images ~/Pictures --style-size 192   # finer, denser
python style.py --images ~/Pictures --style-size 768   # coarser, bolder
```

**`--style-weight` (default 12) is the balance.** Raise it if the output still looks like the
photograph; lower it if the photograph has vanished into wallpaper. The per-term losses are printed
every `--preview-every` steps so you can see which one is dominating.

**`--width` (default 16) is the size/speed tradeoff**, and it is the one that shows up in the app's
frame rate. 8 is fast and coarse, 16 is the default, 32 is slow and detailed. `--blocks` (default 5)
does the same in the other direction.

## Several patterns in one model

Every image in `styles/` becomes its own slider, and the sliders blend rather than select — push two
against each other and you get the mixture. That is one network, one download, and the same three
lines of CPU arithmetic per frame no matter how many styles are in it, because the conditioning
works entirely through the instance-norm affines (Dumoulin et al. 2017).

Sampling covers the corners, the pairs, and the interior of the slider space, so a half-raised
slider means something. All sliders at zero is trained to reproduce the input, which makes zero a
real off position rather than an untrained corner — and two sliders at 1 asks for the same blend as
two at 0.5, so raising everything does not walk off the end of what was trained.

Filenames become the slider labels. Name them how you want to read them.

---

# Distilling DeepDream

### 1. Build the dataset

```bash
python dataset.py --images ~/Pictures/photos --out data/pairs --count 2000 --size 256
```

This is the slow step, and it is meant to be — every pair costs a full octave sweep of gradient
ascent, which is exactly the cost the finished network exists to avoid. Start with `--count 200` to
see output the same hour. `--append` adds to an existing dataset, and the index is rewritten every
50 pairs so an interrupted build still leaves something trainable.

Each example gets its own uniformly sampled control vector, so the sliders end up trained across
their whole range rather than only where the settings happen to look good.

### 2. Train

```bash
python train.py --data data/pairs --out runs/dreamnet --epochs 40
```

Pixel loss against the teacher's frame, plus the same perceptual and warp-equivariance terms as the
style path. The perceptual term is what lets the network commit to *one* of the plausible dreams
instead of averaging them — it is the difference between output that looks dreamed and output that
looks smeared.

---

# Exporting, and shipping it with the app

```bash
python export.py --checkpoint runs/patterns/checkpoint.pt \
    --out ../public/models/patterns.dnw \
    --reference ../public/models/verify.json
```

This writes the model **and** adds it to `public/models/index.json`. The app reads that manifest at
startup, lists everything in it, and loads the first entry automatically — so an exported model is
live on the next reload and ships with `pnpm build` without any further step. Models are tracked in
git for exactly that reason: a deployed build should work for someone who opens it, not ask them to
train their own first.

`--reference` writes a deterministic input and PyTorch's answer to it. Load `/selftest.html` in the
running app and it will run the *exported* model on that same input and compare. This is the only
check that proves the exporter and the WebGL runtime agree rather than merely both running — a
weight packed in the wrong order or a transposed FiLM projection produces a plausible picture, not
an error. The probe control vector is deliberately spread across the slider range rather than left
at the defaults, so the conditioning path carries real values.

`python packing.py` checks the weight layout against the index formula the shader uses, with no
checkpoint needed.

## Files

| | |
|---|---|
| `common.py` | device selection, the control-vector types, image I/O |
| `perceptual.py` | VGG-16 features, Gram matrices, TV, the equivariance warp |
| `style.py` | style-transfer training — the "give it a pattern" path |
| `teacher.py` | DeepDream on GoogLeNet: octaves, Laplacian gradient normalization, jitter, TV |
| `dataset.py` | generates the distillation pairs |
| `train.py` | distillation training |
| `model.py` | the network — must stay in step with `src/gpu/ops.ts` |
| `packing.py` | weight layout, and a self-check of it |
| `export.py` | writes the `.dnw` and updates `index.json` |
