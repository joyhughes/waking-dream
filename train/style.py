"""Trains the same network to paint a specific pattern instead of imitating DeepDream.

    python style.py --styles styles/*.jpg --images ~/Pictures/photos --out runs/patterns

This is fast neural style transfer (Johnson et al. 2016), and unlike `train.py` it needs no dataset
build at all -- there is no teacher to run. The target is not a set of precomputed images; it is a
*statistic* of the style image, and the loss can be evaluated on the fly against any content photo.
In practice that means the difference between a day of generating pairs and starting a run in the
time it takes to point at a folder.

Two losses define the pattern, both measured on VGG-16 features rather than on pixels:

  - the **Gram matrix** of the style image at four layers, which records which features co-occur with
    position thrown away -- the vocabulary of strokes, colours and textures, and nothing about where
    any of it goes;
  - the **content features** of the input frame at one layer, which pin what is in the picture and
    where, while staying indifferent to the pixels that render it.

Several style images can be trained into one network at once. Each gets a slider, and the sliders
blend: the conditioning does not select a style, it mixes them. That is the Dumoulin et al. 2017
result -- N styles in one network, addressed entirely through the instance-norm affines, which is
the same conditioning machinery the distilled models use and the same three lines of CPU arithmetic
per frame in the browser.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import controls_to_json, find_images, load_image, pick_device, save_image, seed_everything, style_controls
from model import DreamNet
from perceptual import (
    DEFAULT_CONTENT_LAYER,
    DEFAULT_STYLE_LAYERS,
    WARP_PADDING_MODE,
    VggFeatures,
    gram_matrix,
    random_warp_grid,
    total_variation,
)


class ContentDataset(Dataset):
    """Random square crops from a photo directory.

    The content images are not being learned from -- any reasonably varied set of photographs works,
    because their only job is to be things the network must keep recognisable while it repaints them.
    A few hundred is plenty; the crops make the effective set much larger.
    """

    def __init__(self, root: Path, size: int, length: int, seed: int):
        self.paths = find_images(root)
        if not self.paths:
            raise SystemExit(f"No images found under {root}.")
        self.size = size
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> torch.Tensor:
        # Seeded from the item index so a worker pool still produces a deterministic run.
        rng = random.Random(self.seed * 1_000_003 + item)
        image = load_image(self.paths[rng.randrange(len(self.paths))])

        _, height, width = image.shape
        if height < self.size or width < self.size:
            scale = self.size / min(height, width)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(max(self.size, int(height * scale + 0.5)), max(self.size, int(width * scale + 0.5))),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            _, height, width = image.shape

        top = rng.randint(0, height - self.size)
        left = rng.randint(0, width - self.size)
        crop = image[:, top : top + self.size, left : left + self.size]
        if rng.random() < 0.5:
            crop = torch.flip(crop, dims=(2,))
        return crop.contiguous()


def load_style_image(path: Path, short_side: int, device: torch.device) -> torch.Tensor:
    """Loads a style image with its short side set to `short_side`.

    This resize is the pattern-scale dial and it is worth understanding before turning anything else.
    The network learns strokes at the size they appear *in pixels*, so shrinking the style image
    makes it draw finer, denser motifs and enlarging it makes them coarser and sparser -- on the same
    content, at the same capture size. It is a far more effective control over how the result looks
    than the loss weights are.
    """
    image = Image.open(path).convert("RGB")
    scale = short_side / min(image.size)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)

    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def sample_controls(batch: int, styles: int, rng: random.Random, device: torch.device) -> torch.Tensor:
    """Control vectors covering the corners, the edges, and the interior of the slider space.

    Corners -- one slider up, the rest down -- are what the user reaches for most, so they get half
    the samples. The rest cover pairs and the full interior, because a network trained only on
    corners has never been asked what a half-raised slider means and will answer with an artifact.
    """
    controls = torch.zeros(batch, styles, device=device)
    for row in range(batch):
        draw = rng.random()
        if draw < 0.5 or styles == 1:
            controls[row, rng.randrange(styles)] = 1.0
        elif draw < 0.7:
            first, second = rng.sample(range(styles), 2)
            controls[row, first] = rng.random()
            controls[row, second] = rng.random()
        else:
            for column in range(styles):
                controls[row, column] = rng.random()
    return controls


def mixture(controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits a control vector into a direction on the simplex and a total mass.

    The network is conditioned on the raw slider values, but the loss needs a single target, so the
    vector is read as "which mixture" (direction) and "how much of it" (mass). Two consequences,
    both of them things a slider should do: all sliders down is trained to reproduce the input, so
    zero is a real off position rather than an untrained corner; and two sliders at 1 asks for the
    same blend as two at 0.5, so raising everything does not walk off the end of what was trained.
    """
    mass = controls.sum(dim=1, keepdim=True)
    direction = controls / mass.clamp(min=1e-6)
    return direction, mass.clamp(max=1.0)


def resolve_styles(entries: list[Path]) -> list[Path]:
    """Turns whatever was passed to --styles into a concrete, ordered list of image files.

    Accepts files, directories, and unexpanded globs, because all three are things people type. The
    order matters and is kept stable: it is the order of the sliders in the app, and it is baked
    into the exported model, so a run resumed after adding a style would otherwise silently
    renumber every existing one.
    """
    paths: list[Path] = []
    for entry in entries:
        if entry.is_dir():
            paths.extend(find_images(entry))
        elif entry.exists():
            paths.append(entry)
        else:
            matches = sorted(Path().glob(str(entry)))
            if not matches:
                raise SystemExit(f"No style images matched {entry}.")
            paths.extend(matches)

    unique = list(dict.fromkeys(paths))
    if not unique:
        raise SystemExit(
            "No style images found. Put a few in train/styles/ (see train/styles/README.md), "
            "or pass --styles path/to/image.jpg."
        )
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a real-time style transfer DreamNet.")
    parser.add_argument("--styles", type=Path, nargs="+", default=[Path("styles")],
                        help="Style images, or a directory of them. Each image becomes its own slider. Defaults to train/styles/.")
    parser.add_argument("--images", type=Path, required=True, help="Directory of content photographs.")
    parser.add_argument("--out", type=Path, default=Path("runs/style"))
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=256, help="Training crop size.")
    parser.add_argument("--style-size", type=int, default=384,
                        help="Short side the style images are resized to. The pattern-scale dial: smaller means finer, denser motifs.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--film-hidden", type=int, default=32)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=12.0,
                        help="The main dial. Raise it if the output still looks like the photo; lower it if the photo has vanished into wallpaper.")
    parser.add_argument("--tv-weight", type=float, default=2e-3)
    parser.add_argument("--warp-weight", type=float, default=0.3,
                        help="Temporal stability. Costs an extra forward pass per step; set to 0 for stills only.")
    parser.add_argument("--content-layer", type=str, default=DEFAULT_CONTENT_LAYER)
    parser.add_argument("--preview-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = pick_device(args.device)

    style_paths = resolve_styles(args.styles)
    style_names = [path.stem for path in style_paths]
    controls_spec = style_controls(style_names)
    style_count = len(style_paths)

    layers = tuple(dict.fromkeys(DEFAULT_STYLE_LAYERS + (args.content_layer,)))
    vgg = VggFeatures(device, layers)

    # The style targets are computed once. They are the only thing the style images are ever used
    # for -- after this they could be deleted and the run would be unaffected.
    style_grams: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        per_style = [vgg(load_style_image(path, args.style_size, device)) for path in style_paths]
        for name in DEFAULT_STYLE_LAYERS:
            style_grams[name] = torch.cat([gram_matrix(features[name]) for features in per_style], dim=0)

    print(f"{style_count} style(s): {', '.join(style_names)}")
    for name in DEFAULT_STYLE_LAYERS:
        print(f"  gram {name}: {tuple(style_grams[name].shape)}")

    dataset = ContentDataset(args.images, args.size, args.iterations * args.batch, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=2, drop_last=True, persistent_workers=True)

    model = DreamNet(
        width=args.width, blocks=args.blocks, film_hidden=args.film_hidden, cond_dims=style_count
    ).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, args.iterations), pct_start=0.1
    )

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{model.parameter_count() / 1000:.0f}k parameters, {args.iterations} iterations on {device}.")

    preview_content = torch.stack([dataset[i] for i in range(min(2, len(dataset)))]).to(device)

    running = {"content": 0.0, "style": 0.0, "tv": 0.0, "warp": 0.0}
    seen = 0
    began = time.time()

    for step, content in enumerate(tqdm(loader, total=args.iterations, desc="training")):
        if step >= args.iterations:
            break
        content = content.to(device, non_blocking=True)
        controls = sample_controls(content.shape[0], style_count, rng, device)
        direction, mass = mixture(controls)

        prediction = model(content, controls)

        # One VGG pass for both images. The content target is detached; it is a fixed thing to match,
        # not something the optimizer has any say in.
        features = vgg(torch.cat([prediction, content], dim=0))

        predicted_content, target_content = features[args.content_layer].chunk(2, dim=0)
        content_loss = F.mse_loss(predicted_content, target_content.detach())

        style_loss = prediction.new_zeros(())
        for name in DEFAULT_STYLE_LAYERS:
            predicted_gram = gram_matrix(features[name].chunk(2, dim=0)[0])
            # The per-example target is the mixture its control vector asks for.
            target_gram = torch.einsum("bs,scd->bcd", direction, style_grams[name])
            per_example = (predicted_gram - target_gram).pow(2).mean(dim=(1, 2))
            # Scaled by mass, so a control vector of all zeros carries no style loss at all and the
            # network is left with only the content term -- which is what makes zero mean "off".
            style_loss = style_loss + (per_example * mass.squeeze(1)).mean()

        tv_loss = total_variation(prediction)

        loss = args.content_weight * content_loss + args.style_weight * style_loss + args.tv_weight * tv_loss
        running["content"] += content_loss.item()
        running["style"] += style_loss.item()
        running["tv"] += tv_loss.item()

        if args.warp_weight > 0:
            grid = random_warp_grid(content.shape[0], content.shape[2], content.shape[3], device)
            warped_input = F.grid_sample(content, grid, align_corners=False, padding_mode=WARP_PADDING_MODE)
            warped_prediction = F.grid_sample(prediction, grid, align_corners=False, padding_mode=WARP_PADDING_MODE)
            # Only one side carries gradient. Letting both move makes the constraint symmetric and
            # noticeably less stable -- the pair can satisfy it by agreeing on something degenerate.
            warp_loss = F.mse_loss(model(warped_input, controls), warped_prediction.detach())
            loss = loss + args.warp_weight * warp_loss
            running["warp"] += warp_loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        seen += 1

        if (step + 1) % args.preview_every == 0 or step + 1 == args.iterations:
            means = {key: value / max(1, seen) for key, value in running.items()}
            elapsed = (time.time() - began) / 60
            print(
                f"\nstep {step + 1}: content {means['content']:.4f} style {means['style']:.4f} "
                f"tv {means['tv']:.5f} warp {means['warp']:.5f} · {elapsed:.1f} min"
            )
            running = {key: 0.0 for key in running}
            seen = 0

            write_previews(model, preview_content, style_names, device, args.out / "previews" / f"step{step + 1:06d}")
            torch.save(
                {
                    "model": model.state_dict(),
                    "width": args.width, "blocks": args.blocks, "film_hidden": args.film_hidden,
                    "controls": controls_to_json(controls_spec),
                    "description": f"Real-time style transfer: {', '.join(style_names)}.",
                    "provenance": {
                        "kind": "style",
                        "styles": [str(path) for path in style_paths],
                        "styleSize": args.style_size,
                        "styleWeight": args.style_weight,
                        "contentWeight": args.content_weight,
                    },
                    "trained_at": args.size,
                    "step": step + 1,
                },
                args.out / "checkpoint.pt",
            )

    print(f"Done in {(time.time() - began) / 60:.1f} min. Export with:")
    print(f"  python export.py --checkpoint {args.out / 'checkpoint.pt'} --out ../public/models/{args.out.name}.dnw")


@torch.no_grad()
def write_previews(
    model: DreamNet, content: torch.Tensor, style_names: list[str], device: torch.device, out: Path
) -> None:
    """One row per content image: the input, then the output at each style's own slider fully up.

    Worth actually looking at every time. The loss numbers cannot distinguish a network that has
    learned the pattern from one that has learned to paint the same texture over everything
    regardless of what is underneath, and side by side the two are obvious.
    """
    model.eval()
    style_count = len(style_names)
    for index in range(content.shape[0]):
        source = content[index : index + 1]
        panels = [source.squeeze(0).cpu()]
        for style in range(style_count):
            controls = torch.zeros(1, style_count, device=device)
            controls[0, style] = 1.0
            panels.append(model(source, controls).squeeze(0).cpu())
        save_image(torch.cat(panels, dim=2), out / f"{index:02d}.png")
    model.train()


if __name__ == "__main__":
    main()
