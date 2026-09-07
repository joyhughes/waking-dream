"""Builds the (frame, dreamed frame) pairs the student is fitted to.

This is the slow step and it is meant to be run once. Every pair costs a full octave sweep of
gradient ascent, which is exactly the cost the finished network exists to avoid -- so the whole
point is to pay it here, offline, in bulk, and never again at frame time.

    python dataset.py --images ~/Pictures/photos --out data/pairs --count 2000

Each example gets its own uniformly sampled control vector, so the dataset covers the whole range
the sliders can reach rather than one setting. Crops are random, so a modest photo directory yields
far more distinct examples than it has files.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from common import DREAM_CONTROLS, find_images, load_image, pick_device, random_controls, save_image, seed_everything, write_jsonl
from teacher import DeepDreamTeacher, settings_from_controls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeepDream training pairs.")
    parser.add_argument("--images", type=Path, required=True, help="Directory of source photographs, searched recursively.")
    parser.add_argument("--out", type=Path, default=Path("data/pairs"))
    parser.add_argument("--count", type=int, default=2000, help="Number of pairs to generate.")
    parser.add_argument("--size", type=int, default=256, help="Side of the square crops. Match the capture size you intend to run at.")
    parser.add_argument("--steps", type=int, default=10, help="Ascent steps per octave in the teacher.")
    parser.add_argument("--octaves", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--append", action="store_true", help="Add to an existing dataset instead of starting over.")
    return parser.parse_args()


def random_crop(image: torch.Tensor, size: int, rng: random.Random) -> torch.Tensor:
    """A random square crop, upscaling first if the image is smaller than the crop."""
    _, height, width = image.shape
    if height < size or width < size:
        scale = size / min(height, width)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(max(size, int(height * scale + 0.5)), max(size, int(width * scale + 0.5))),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        _, height, width = image.shape

    top = rng.randint(0, height - size)
    left = rng.randint(0, width - size)
    crop = image[:, top : top + size, left : left + size]

    if rng.random() < 0.5:
        crop = torch.flip(crop, dims=(2,))
    return crop.contiguous()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)

    sources = find_images(args.images)
    if not sources:
        raise SystemExit(f"No images found under {args.images}.")

    device = pick_device(args.device)
    print(f"{len(sources)} source images, generating {args.count} pairs at {args.size}px on {device}.")

    teacher = DeepDreamTeacher(device)

    inputs_dir = args.out / "inputs"
    targets_dir = args.out / "targets"
    index_path = args.out / "index.jsonl"

    rows: list[dict] = []
    start_at = 0
    if args.append and index_path.exists():
        from common import read_jsonl

        rows = read_jsonl(index_path)
        start_at = len(rows)
        print(f"Appending to {start_at} existing pairs.")

    began = time.time()
    for i in tqdm(range(args.count), desc="dreaming"):
        index = start_at + i
        # Loading fresh each time rather than caching decoded images: at 256px the crop is a
        # rounding error next to the ascent, and holding a photo directory in memory is not free.
        source = load_image(sources[rng.randrange(len(sources))])
        crop = random_crop(source, args.size, rng)

        controls = random_controls(rng, DREAM_CONTROLS)
        settings = settings_from_controls(controls, steps=args.steps, octaves=args.octaves)

        dreamed = teacher.dream(crop.unsqueeze(0).to(device), settings).squeeze(0).cpu()

        save_image(crop, inputs_dir / f"{index:06d}.png")
        save_image(dreamed, targets_dir / f"{index:06d}.png")
        rows.append({"index": index, "controls": controls, "layers": list(settings.layers)})

        # Rewritten periodically rather than only at the end, so an interrupted build still leaves a
        # dataset that train.py can read.
        if (i + 1) % 50 == 0 or i + 1 == args.count:
            write_jsonl(index_path, rows)

    elapsed = time.time() - began
    print(f"Wrote {len(rows)} pairs to {args.out} in {elapsed / 60:.1f} min ({elapsed / max(1, args.count):.2f} s/pair).")
    print("Controls sampled uniformly over: " + ", ".join(f"{c.name}[{c.minimum}, {c.maximum}]" for c in DREAM_CONTROLS))


if __name__ == "__main__":
    main()
