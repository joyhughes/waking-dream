"""Shared pieces: device selection, the control vector, and image plumbing.

The control vector is defined here and nowhere else. It is the one thing that has to mean exactly
the same in four places -- the teacher that is asked for a dream at these settings, the dataset that
records them, the network that is conditioned on them, and the sliders the browser draws -- so it
lives in a single table that the exporter copies verbatim into the model file.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class Control:
    name: str
    label: str
    description: str
    minimum: float
    maximum: float
    default: float


CONTROLS: tuple[Control, ...] = (
    Control(
        "layer",
        "Layer depth",
        "Which depth of the teacher network the dream was pulled from. Low is edges and texture; "
        "high is object-like, the eyes-and-animals end.",
        0.0, 1.0, 0.5,
    ),
    Control(
        "strength",
        "Strength",
        "How far the teacher's ascent was pushed before the frame was recorded.",
        0.0, 1.0, 0.6,
    ),
    Control(
        "scale",
        "Pattern scale",
        "How large the drawn motifs come out. The teacher gets fewer pixels to draw on, so each "
        "feature it draws covers more of the frame.",
        0.0, 1.0, 0.35,
    ),
)

CONTROL_DIMS = len(CONTROLS)


def pick_device(preferred: str | None = None) -> torch.device:
    """MPS where it exists, then CUDA, then CPU."""
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    """Every image under `root`, sorted so a run is reproducible from the same directory."""
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def load_image(path: Path, size: int | None = None) -> torch.Tensor:
    """Loads an image as a `[3, H, W]` float tensor in [0, 1], optionally centre-cropped to a square.

    Cropping rather than resizing to a square: the network is trained on undistorted photographs
    because that is what a camera hands it at runtime, and a squashed training set teaches it to
    draw squashed features.
    """
    image = Image.open(path).convert("RGB")

    if size is not None:
        shortest = min(image.size)
        scale = size / shortest
        resized = image.resize(
            (max(size, round(image.width * scale)), max(size, round(image.height * scale))),
            Image.LANCZOS,
        )
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        image = resized.crop((left, top, left + size, top + size))

    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Writes a `[3, H, W]` tensor in [0, 1] as a PNG.

    PNG rather than JPEG throughout: the targets are what the student is asked to reproduce exactly,
    and JPEG's ringing around the high-contrast edges a dream is full of would be baked into the
    thing it learns to draw.
    """
    array = (tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def random_controls(rng: random.Random) -> list[float]:
    """A uniformly sampled control vector.

    Uniform on purpose. The conditioning has to behave across the whole range a slider can reach,
    and a sampler concentrated on settings that look good would leave the ends of every slider
    trained on nothing.
    """
    return [rng.uniform(control.minimum, control.maximum) for control in CONTROLS]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
