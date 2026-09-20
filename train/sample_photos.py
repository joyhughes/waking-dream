"""Builds a content set by sampling at random from the macOS Photos library.

    python3 sample_photos.py --count 400 --out data/content

Content photographs are not what a style model learns — the look comes entirely from the style
images. They are only things it must keep recognisable while repainting them, so what matters is
that they span the *kind* of picture the model will meet at run time. A random draw from a personal
library is a very good approximation of that, and a much better one than any stock photo set: it
already has the right faces, rooms, lighting and framing.

This reads the library's `originals` directory directly. That needs nothing installed — `sips`,
which does the HEIC conversion and the downscale, ships with macOS — but it also means it bypasses
the Photos database, with two consequences worth knowing before running it:

  - **It can see photos you deleted or hid.** Items sit in `originals` until Photos purges them, and
    hidden albums are not marked in any way visible from the file system. If that matters — and if a
    model trained on them might be shared, it does — use the osxphotos route described in the
    README instead, which asks the database and can exclude both.
  - It cannot filter by album, favourite, date or keyword, for the same reason.

Nothing in the library is modified. Files are read, and copies are written to `--out`.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_LIBRARY = Path.home() / "Pictures" / "Photos Library.photoslibrary" / "originals"

# PNG is left out deliberately: in a phone-backed library almost every PNG is a screenshot, and a
# screenshot teaches the network to repaint user interfaces. Movies are skipped outright.
STILL_SUFFIXES = {".jpg", ".jpeg", ".heic", ".heif"}

# Extremes at either end are not useful content. A near-square crop is taken from each image at
# training time, so a panorama contributes one narrow slice of itself, and anything tiny is a
# thumbnail or an avatar rather than a photograph.
MIN_PIXELS = 600
MAX_ASPECT = 2.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample random photos into a training content set.")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY,
                        help="Photos library originals directory, or any folder of images.")
    parser.add_argument("--out", type=Path, default=Path("data/content"))
    parser.add_argument("--count", type=int, default=400,
                        help="How many photos to end up with. 300-500 is the comfortable range.")
    parser.add_argument("--size", type=int, default=1024,
                        help="Longest side of the copies. Training crops at 256, so this leaves room to crop.")
    parser.add_argument("--seed", type=int, default=0, help="Change it to draw a different sample.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be taken, copy nothing.")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Add to whatever is already in --out instead of starting clean.")
    return parser.parse_args()


def dimensions(path: Path) -> tuple[int, int] | None:
    """Width and height via sips, or None if it cannot be read as an image."""
    try:
        result = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            capture_output=True, text=True, timeout=20,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    width = height = 0
    for line in result.stdout.splitlines():
        if "pixelWidth:" in line:
            width = int(line.split(":")[1])
        elif "pixelHeight:" in line:
            height = int(line.split(":")[1])
    return (width, height) if width and height else None


def usable(path: Path) -> bool:
    size = dimensions(path)
    if size is None:
        return False
    width, height = size
    if min(width, height) < MIN_PIXELS:
        return False
    return max(width, height) / min(width, height) <= MAX_ASPECT


def convert(source: Path, destination: Path, longest: int) -> bool:
    """Writes a downscaled JPEG copy. Returns False if sips could not read the source."""
    try:
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", "-Z", str(longest), str(source), "--out", str(destination)],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and destination.exists()


def main() -> None:
    args = parse_args()

    if not args.library.exists():
        raise SystemExit(f"No such directory: {args.library}")

    print(f"Scanning {args.library} …")
    candidates = [p for p in args.library.rglob("*") if p.suffix.lower() in STILL_SUFFIXES]
    if not candidates:
        raise SystemExit("Found no JPEG or HEIC files there.")
    print(f"{len(candidates)} still images found.")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    if args.dry_run:
        print(f"Would take {min(args.count, len(candidates))} of them, converting to JPEG at "
              f"{args.size}px into {args.out}.")
        for path in candidates[:5]:
            print(f"  e.g. {path.name}")
        return

    if args.out.exists() and not args.keep_existing:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    taken = skipped = failed = 0
    # Walked lazily rather than filtered up front: reading dimensions costs a subprocess per file,
    # and checking sixty thousand of them to choose four hundred would take far longer than the
    # training run it is preparing for.
    for path in candidates:
        if taken >= args.count:
            break
        if not usable(path):
            skipped += 1
            continue
        if convert(path, args.out / f"{taken:05d}.jpg", args.size):
            taken += 1
            if taken % 50 == 0:
                print(f"  {taken}/{args.count}")
        else:
            failed += 1

    print(f"\nWrote {taken} photos to {args.out}")
    print(f"  skipped {skipped} (too small, or too far from square)")
    if failed:
        print(f"  {failed} could not be converted")
    if taken < args.count:
        print(f"  ran out of candidates before reaching {args.count}")
    print(f"\nTrain with:\n  python style.py --images {args.out} --out runs/mine", file=sys.stderr)


if __name__ == "__main__":
    main()
