"""Fits the student to the teacher's output.

    python train.py --data data/pairs --out runs/eyes --epochs 40

Three loss terms, and each is doing something the others cannot:

  - **Pixel loss** against the teacher's frame. On its own it produces a blurry average of every
    plausible dream, because where the teacher's choice of what to draw is arbitrary the safe
    prediction is the mean of the options.
  - **Perceptual loss** against the same frame, on VGG features. This is what makes the output look
    dreamed rather than smeared: it asks for the same *kind* of structure in the same place instead
    of the same pixels, so the network is free to commit to one of the plausible dreams. Johnson et
    al. 2016 introduced it for exactly this failure.
  - **Warp equivariance**. The teacher is run per frame with fresh random jitter, so consecutive
    video frames get independently-drawn detail and the result flickers. Asking the network to
    commute with a small random warp -- f(warp(x)) should equal warp(f(x)) -- teaches it to attach
    what it draws to the content rather than to the pixel grid, without needing optical flow or
    video training data. It is the cheap stand-in for the temporal losses that video style-transfer
    work uses, and on a webcam it is the difference between a stable effect and a boiling one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import DREAM_CONTROLS, controls_to_json, load_image, pick_device, read_jsonl, save_image, seed_everything
from model import DreamNet
from perceptual import WARP_PADDING_MODE, VggFeatures, random_warp_grid


class PairDataset(Dataset):
    def __init__(self, root: Path):
        self.root = root
        self.rows = read_jsonl(root / "index.jsonl")
        if not self.rows:
            raise SystemExit(f"No pairs found in {root}. Run dataset.py first.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        row = self.rows[item]
        index = row["index"]
        source = load_image(self.root / "inputs" / f"{index:06d}.png")
        target = load_image(self.root / "targets" / f"{index:06d}.png")
        controls = torch.tensor(row["controls"], dtype=torch.float32)
        return source, target, controls


class PerceptualLoss(torch.nn.Module):
    """Feature matching at relu2_2 and relu3_3, the layers Johnson et al. matched content at."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.vgg = VggFeatures(device, ("relu2_2", "relu3_3"))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Both images go through in one batch, which halves the number of VGG passes per step.
        both = self.vgg(torch.cat([prediction, target], dim=0))
        total = prediction.new_zeros(())
        for activations in both.values():
            predicted, wanted = activations.chunk(2, dim=0)
            total = total + F.mse_loss(predicted, wanted.detach())
        return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DreamNet on DeepDream pairs.")
    parser.add_argument("--data", type=Path, default=Path("data/pairs"))
    parser.add_argument("--out", type=Path, default=Path("runs/dreamnet"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=16, help="Base channel width. 8 is fast and coarse, 32 is slow and detailed.")
    parser.add_argument("--blocks", type=int, default=5, help="Residual blocks.")
    parser.add_argument("--film-hidden", type=int, default=32)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--perceptual-weight", type=float, default=0.6)
    parser.add_argument("--warp-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)

    dataset = PairDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        # MPS has no separate worker device context worth spinning up for PNG decoding of this size;
        # two workers keeps the GPU fed without the startup cost of more.
        num_workers=2,
        drop_last=True,
        persistent_workers=True,
    )

    model = DreamNet(
        width=args.width, blocks=args.blocks, film_hidden=args.film_hidden, cond_dims=len(DREAM_CONTROLS)
    ).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"])

    perceptual = PerceptualLoss(device) if args.perceptual_weight > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = max(1, args.epochs * len(loader))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.15)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps(
            {"width": args.width, "blocks": args.blocks, "film_hidden": args.film_hidden,
             "data": str(args.data), "epochs": args.epochs, "lr": args.lr}, indent=2
        )
    )

    print(f"{len(dataset)} pairs, {model.parameter_count() / 1000:.0f}k parameters, {total_steps} steps on {device}.")

    step = 0
    began = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = {"pixel": 0.0, "perceptual": 0.0, "warp": 0.0}
        seen = 0

        for source, target, controls in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            controls = controls.to(device, non_blocking=True)

            prediction = model(source, controls)

            pixel = F.mse_loss(prediction, target)
            loss = args.pixel_weight * pixel
            running["pixel"] += pixel.item()

            if perceptual is not None:
                perceptual_loss = perceptual(prediction, target)
                loss = loss + args.perceptual_weight * perceptual_loss
                running["perceptual"] += perceptual_loss.item()

            if args.warp_weight > 0:
                grid = random_warp_grid(source.shape[0], source.shape[2], source.shape[3], device)
                warped_input = F.grid_sample(source, grid, align_corners=False, padding_mode=WARP_PADDING_MODE)
                # Warping the prediction rather than recomputing it: this side is the target, and
                # letting the gradient flow through both copies makes the constraint symmetric and
                # noticeably less stable to train.
                warped_prediction = F.grid_sample(prediction, grid, align_corners=False, padding_mode=WARP_PADDING_MODE)
                warp_loss = F.mse_loss(model(warped_input, controls), warped_prediction.detach())
                loss = loss + args.warp_weight * warp_loss
                running["warp"] += warp_loss.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()

            step += 1
            seen += 1

        means = {name: value / max(1, seen) for name, value in running.items()}
        elapsed = (time.time() - began) / 60
        print(
            f"epoch {epoch + 1}: pixel {means['pixel']:.5f} perceptual {means['perceptual']:.4f} "
            f"warp {means['warp']:.5f} · lr {schedule.get_last_lr()[0]:.2e} · {elapsed:.1f} min"
        )

        torch.save(
            {"model": model.state_dict(), "width": args.width, "blocks": args.blocks,
             "film_hidden": args.film_hidden, "epoch": epoch + 1,
             "controls": controls_to_json(DREAM_CONTROLS),
             "description": "Distilled DeepDream, GoogLeNet teacher.",
             "provenance": {"kind": "distilled", "backbone": "googlenet", "data": str(args.data)},
             "trained_at": 256},
            args.out / "checkpoint.pt",
        )
        write_previews(model, dataset, device, args.out / "previews" / f"epoch{epoch + 1:03d}")

    print(f"Done in {(time.time() - began) / 60:.1f} min. Export with:")
    print(f"  python export.py --checkpoint {args.out / 'checkpoint.pt'} --out ../public/models/dreamnet.dnw")


@torch.no_grad()
def write_previews(model: DreamNet, dataset: PairDataset, device: torch.device, out: Path, count: int = 3) -> None:
    """Writes input / prediction / teacher triptychs, which is the only way to see what is going wrong.

    The loss curve cannot distinguish "learning to draw the dream" from "learning to blur", and those
    two look completely different side by side.
    """
    model.eval()
    for i in range(min(count, len(dataset))):
        source, target, controls = dataset[i * max(1, len(dataset) // max(1, count))]
        prediction = model(source.unsqueeze(0).to(device), controls.unsqueeze(0).to(device)).squeeze(0).cpu()
        strip = torch.cat([source, prediction, target], dim=2)
        save_image(strip, out / f"{i:02d}.png")
    model.train()


if __name__ == "__main__":
    main()
