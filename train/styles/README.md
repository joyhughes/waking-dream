# Style images

Drop image files in this directory. Each one becomes a pattern the network learns, and each gets its
own slider in the app — raise one for that pattern alone, raise several to blend them.

```bash
cd train
source .venv/bin/activate
python style.py --images ~/Pictures/photos          # uses every image in this folder
```

## What makes a good style image

- **Texture over composition.** The network learns which strokes, colours and textures go together,
  with position thrown away entirely. A painting with a consistent surface all over teaches far more
  than one whose interest is in what is where.
- **Fill the frame with the pattern.** Large empty areas are learned as "large empty areas", and the
  network will happily paint them over your video.
- **Strong, saturated colour** survives the trip. Muted styles come out muted and then get further
  washed out by the feedback loop.
- **A few hundred pixels of detail is enough.** They are resized to `--style-size` (384 by default)
  before anything looks at them, so a 6000px scan buys nothing.

## Pattern scale

`--style-size` is the most effective control over how the result looks, and it is not a quality
setting. The network learns strokes at the size they appear *in pixels*, so:

```bash
python style.py --images ~/Pictures --style-size 192    # finer, denser, busier
python style.py --images ~/Pictures --style-size 384    # the default
python style.py --images ~/Pictures --style-size 768    # coarser, sparser, bolder
```

If a style comes out as unreadable fine noise, it is almost always this rather than the loss weights.

## How many at once

Each style adds a slider and costs almost nothing at runtime — the conditioning is the same three
lines of CPU arithmetic per frame whether there is one style or twelve. It does divide the training
budget, though: eight styles in one run each get an eighth of the gradient steps. Four to six in a
run of 8000+ iterations is a reasonable place to start.

Filenames become the slider labels, so name them how you want to see them: `great-wave.jpg` shows up
as "Style: great-wave".
