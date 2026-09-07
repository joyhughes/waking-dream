# Training a DreamNet

The idea: run slow, correct DeepDream over a pile of photographs, then fit a small feed-forward
network to reproduce what it produced. At runtime there are zero gradient-ascent iterations.

```
photographs  ──▶  teacher.py  ──▶  (frame, dreamed frame) pairs
                  (slow, offline)
                                        │
                                        ▼
                          train.py fits model.py to them
                                        │
                                        ▼
                          export.py  ──▶  ../public/models/*.dnw
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Torch picks up Apple silicon's MPS backend automatically; `--device cpu` forces it off.

## 1. Build the dataset

```bash
python dataset.py --images ~/Pictures/photos --out data/pairs --count 2000 --size 256
```

This is the slow step, and it is meant to be — every pair costs a full octave sweep of gradient
ascent, which is exactly the cost the finished network exists to avoid. Budget on the order of a
second or two per pair on an M-series Mac; start with `--count 200` to see output the same hour.

Crops are random and each example gets its own uniformly sampled control vector, so a few hundred
photographs go a long way, and the sliders end up trained across their whole range rather than only
where the settings happen to look good.

`--append` adds to an existing dataset. The index is rewritten every 50 pairs, so an interrupted
build still leaves something trainable.

## 2. Train

```bash
python train.py --data data/pairs --out runs/dreamnet --epochs 40
```

Three loss terms:

- **Pixel** against the teacher's frame. Alone, it produces a blur — where the teacher's choice of
  what to draw is arbitrary, the safest prediction is the average of every option.
- **Perceptual** (VGG-16, relu2_2 and relu3_3). This is what lets the network commit to *one* of the
  plausible dreams instead of averaging them, and it is the difference between output that looks
  dreamed and output that looks smeared.
- **Warp equivariance**: `f(warp(x))` should equal `warp(f(x))` for a small random similarity
  transform. The teacher runs with fresh random jitter per frame, so consecutive video frames get
  independently-drawn detail and the naive result boils. This asks the network to attach what it
  draws to the content rather than to the pixel grid, and needs neither optical flow nor video data.

Previews go to `runs/<name>/previews/` as input / prediction / teacher triptychs. Look at these. The
loss curve cannot tell "learning to draw the dream" from "learning to blur"; the triptychs can.

Size knobs: `--width` is the base channel count (8 fast and coarse, 16 default, 32 slow and
detailed) and `--blocks` the number of residual blocks.

## 3. Export

```bash
python export.py --checkpoint runs/dreamnet/checkpoint.pt \
    --out ../public/models/dreamnet.dnw \
    --reference ../public/models/verify.json
```

`--reference` writes a deterministic input and PyTorch's answer to it. Load `/selftest.html` in the
running app and it will run the exported model on that same input and compare — this is the check
that catches a weight packed in the wrong order or an op list that drifted from `model.py`, all of
which otherwise produce a plausible picture rather than an error.

Run `python packing.py` on its own to check the weight layout against the index formula the shader
uses, without needing a checkpoint.

## The control vector

Defined once, in `common.py`, and copied verbatim into the model file so the browser draws the right
sliders with the right labels. Currently three: layer depth, strength, pattern scale. Adding a
fourth means adding a row to `CONTROLS`, teaching `settings_from_controls` what it does to the
teacher, and rebuilding the dataset — the network side and the browser side both pick it up on
their own.

Conditioning works by predicting offsets to each instance-norm layer's per-channel scale and shift.
The projections are zero-initialized, so training starts from an unconditioned network and the
conditioning grows out of it rather than injecting noise into every layer before the controls mean
anything.

## Files

| | |
|---|---|
| `common.py` | device selection, the control vector, image I/O |
| `teacher.py` | DeepDream on GoogLeNet: octaves, Laplacian gradient normalization, jitter, TV |
| `dataset.py` | generates the pairs |
| `model.py` | the student — must stay in step with `src/gpu/ops.ts` |
| `train.py` | the training loop |
| `packing.py` | weight layout, and a self-check of it |
| `export.py` | writes the `.dnw` the browser loads |
