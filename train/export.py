"""Writes a trained checkpoint out as the `.dnw` file the browser runtime loads.

    python export.py --checkpoint runs/dreamnet/checkpoint.pt --out ../public/models/dreamnet.dnw

This script and `src/model/format.ts` are two halves of one format, and this script and
`src/gpu/shaders.ts` are two halves of one memory layout. The convolution shader indexes a weight
texel as `(tap * paddedIn + cin) * groupsOut + groupOut`, with four output channels to a texel and
both channel counts padded up to a multiple of four -- `pack_conv` below is that sentence in NumPy.

`--reference` additionally writes a small input/output pair straight from PyTorch. The browser
self-test picks it up and runs the exported model on the same input, which is the only check that
actually proves the two implementations agree rather than merely both running.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from common import Control, controls_from_json, controls_to_json, pick_device
from model import DreamNet
from packing import groups_for, pack_bias_array, pack_conv_array

DNW_MAGIC = b"DNW1"
DNW_FORMAT = "dreamnet-weights-1"


def pack_conv(weight: torch.Tensor) -> np.ndarray:
    return pack_conv_array(weight.detach().cpu().numpy())


def pack_bias(bias: torch.Tensor) -> np.ndarray:
    return pack_bias_array(bias.detach().cpu().numpy())


def build(
    model: DreamNet,
    controls: tuple[Control, ...],
    name: str,
    description: str,
    provenance: dict,
    trained_at: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    gpu_chunks: list[np.ndarray] = []
    offsets: list[dict[str, int]] = []
    cursor = 0

    for conv_layer in model.convs:
        weights = pack_conv(conv_layer.conv.weight)
        biases = pack_bias(conv_layer.conv.bias)
        offsets.append({"weight": cursor, "bias": cursor + len(weights)})
        gpu_chunks.extend([weights, biases])
        cursor += len(weights) + len(biases)

    gpu = np.concatenate(gpu_chunks, axis=0).astype(np.float32).reshape(-1)

    cpu_values: list[np.ndarray] = []
    cpu_tensors: dict[str, dict] = {}
    cpu_cursor = 0

    def add_cpu(key: str, array: np.ndarray) -> None:
        nonlocal cpu_cursor
        flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        cpu_tensors[key] = {"offset": cpu_cursor, "length": int(flat.size), "shape": list(array.shape)}
        cpu_values.append(flat)
        cpu_cursor += int(flat.size)

    norm_slots: list[int] = []
    for slot, norm in enumerate(model.norms):
        norm_slots.append(int(norm.channels))
        add_cpu(f"norm{slot}.gamma", norm.gamma.detach().cpu().numpy())
        add_cpu(f"norm{slot}.beta", norm.beta.detach().cpu().numpy())

    # nn.Linear stores [out, in]; the browser reads both of these row-major as [in, out], so they
    # are transposed on the way out rather than indexed differently on the way in.
    add_cpu("film.w1", model.film.weight.detach().cpu().numpy().T)
    add_cpu("film.b1", model.film.bias.detach().cpu().numpy())
    for slot, norm in enumerate(model.norms):
        add_cpu(f"film.gammaW{slot}", norm.gamma_projection.weight.detach().cpu().numpy().T)
        add_cpu(f"film.betaW{slot}", norm.beta_projection.weight.detach().cpu().numpy().T)

    cpu = np.concatenate(cpu_values, axis=0).astype(np.float32) if cpu_values else np.zeros(0, dtype=np.float32)

    ops: list[dict] = []

    def conv_op(index: int, source: str, target: str, activation: str = "none") -> None:
        layer = model.convs[index]
        ops.append({
            "type": "conv", "in": source, "out": target,
            "kernel": int(layer.conv.kernel_size[0]),
            "stride": int(layer.conv.stride[0]),
            "inChannels": int(layer.conv.in_channels),
            "outChannels": int(layer.conv.out_channels),
            "weightOffset": offsets[index]["weight"],
            "biasOffset": offsets[index]["bias"],
            "activation": activation,
        })

    def norm_op(slot: int, source: str, target: str, relu: bool, skip: str | None = None) -> None:
        ops.append({
            "type": "norm", "in": source, "out": target,
            "channels": norm_slots[slot], "slot": slot, "relu": relu, "skip": skip,
        })

    conv_op(0, "x", "c0")
    norm_op(0, "c0", "h0", relu=True)
    conv_op(1, "h0", "c1")
    norm_op(1, "c1", "h1", relu=True)
    conv_op(2, "h1", "c2")
    norm_op(2, "c2", "h2", relu=True)

    current = "h2"
    for block in range(model.blocks):
        first, second = 3 + 2 * block, 4 + 2 * block
        conv_op(first, current, f"b{block}c0")
        norm_op(first, f"b{block}c0", f"b{block}h0", relu=True)
        conv_op(second, f"b{block}h0", f"b{block}c1")
        # The skip is folded into the normalization's write, exactly as the residual add in model.py.
        norm_op(second, f"b{block}c1", f"b{block}out", relu=False, skip=current)
        current = f"b{block}out"

    upsample_first = 3 + 2 * model.blocks
    ops.append({"type": "resize", "in": current, "out": "u0", "scale": 2})
    conv_op(upsample_first, "u0", "uc0")
    norm_op(upsample_first, "uc0", "uh0", relu=True)
    ops.append({"type": "resize", "in": "uh0", "out": "u1", "scale": 2})
    conv_op(upsample_first + 1, "u1", "uc1")
    norm_op(upsample_first + 1, "uc1", "uh1", relu=True)
    conv_op(model.output_conv_index, "uh1", "y", activation="tanh01")

    header = {
        "format": DNW_FORMAT,
        "name": name,
        "description": description,
        "teacher": provenance,
        "trainedAt": trained_at,
        "inputName": "x",
        "outputName": "y",
        "ops": ops,
        "normSlots": norm_slots,
        "gpuTexels": int(gpu.size // 4),
        "cpuFloats": int(cpu.size),
        "cpuTensors": cpu_tensors,
        "conditioning": {
            "dims": int(model.cond_dims),
            "hidden": int(model.film_hidden),
            "controls": controls_to_json(controls),
        },
    }

    return header, gpu, cpu


def write_dnw(path: Path, header: dict, gpu: np.ndarray, cpu: np.ndarray) -> None:
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The float payload has to start 4-byte aligned; padding with spaces keeps the header valid JSON.
    padding = (-(8 + len(body))) % 4
    body = body + b" " * padding

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(DNW_MAGIC)
        handle.write(struct.pack("<I", len(body)))
        handle.write(body)
        handle.write(gpu.astype("<f4").tobytes())
        handle.write(cpu.astype("<f4").tobytes())


def update_index(model_path: Path, header: dict, byte_size: int) -> Path:
    """Adds this model to the `index.json` the app reads to populate its model list.

    The app cannot enumerate a directory over HTTP, so a served build needs a manifest of what is
    there. Keeping it here rather than as a separate build step means an exported model is available
    in the app immediately, and the same file is what gets deployed.
    """
    index_path = model_path.parent / "index.json"
    entry = {
        "file": model_path.name,
        "name": header["name"],
        "description": header["description"],
        "bytes": byte_size,
        "trainedAt": header.get("trainedAt"),
        "kind": header.get("teacher", {}).get("kind", "unknown"),
        "controls": [control["label"] for control in header["conditioning"]["controls"]],
    }

    models: list[dict] = []
    if index_path.exists():
        try:
            models = json.loads(index_path.read_text()).get("models", [])
        except json.JSONDecodeError:
            print(f"Warning: {index_path} was not readable JSON; rewriting it.")

    models = [existing for existing in models if existing.get("file") != entry["file"]]
    models.append(entry)
    models.sort(key=lambda row: row["name"])

    index_path.write_text(json.dumps({"models": models}, indent=2) + "\n")
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a DreamNet checkpoint to .dnw.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("../public/models/dreamnet.dnw"))
    parser.add_argument("--name", type=str, default=None, help="Defaults to the output filename.")
    parser.add_argument("--description", type=str, default=None, help="Defaults to what the trainer recorded.")
    parser.add_argument("--trained-at", type=int, default=0, help="0 uses what the trainer recorded.")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Also write a PyTorch input/output pair here for the browser self-test.")
    parser.add_argument("--reference-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    controls = controls_from_json(checkpoint["controls"])

    model = DreamNet(
        width=checkpoint["width"],
        blocks=checkpoint["blocks"],
        film_hidden=checkpoint["film_hidden"],
        cond_dims=len(controls),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    name = args.name or args.out.stem
    # Whatever the trainer recorded about where this model came from — the teacher's backbone for a
    # distilled one, the style filenames for a style-transfer one. Carried through so a model file
    # on its own can still say what produced it.
    header, gpu, cpu = build(
        model, controls, name,
        args.description or checkpoint.get("description", ""),
        provenance=checkpoint.get("provenance", {}),
        trained_at=args.trained_at or checkpoint.get("trained_at", 256),
    )
    write_dnw(args.out, header, gpu, cpu)

    total = int((gpu.size + cpu.size) * 4)
    print(f"Wrote {args.out} - {len(header['ops'])} ops, {model.parameter_count() / 1000:.0f}k parameters, {total / 1024:.0f} kB.")

    index_path = update_index(args.out, header, total)
    print(f"Listed it in {index_path}; the app will offer it in its model list.")

    if args.reference:
        write_reference(model, controls, args.out, args.reference, args.reference_size)


@torch.no_grad()
def write_reference(
    model: DreamNet, controls: tuple[Control, ...], model_path: Path, reference_path: Path, size: int
) -> None:
    """A deterministic input and PyTorch's answer to it, for the browser to check itself against."""
    device = pick_device("cpu")
    generator = torch.Generator(device="cpu").manual_seed(1234)
    source = torch.rand(1, 3, size, size, generator=generator)

    # Deliberately not the defaults. A default vector is often a corner -- one slider up, the rest at
    # zero -- and a corner can leave most of the conditioning MLP multiplying by nothing, so a
    # transposed FiLM projection would sail through the comparison. Spreading the probe across the
    # range makes every path carry a value.
    control_vector = torch.tensor(
        [[c.minimum + (c.maximum - c.minimum) * (0.25 + 0.5 * ((index * 7) % 5) / 4.0)
          for index, c in enumerate(controls)]],
        dtype=torch.float32,
    )

    output = model.to(device)(source, control_vector)

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps({
        "model": model_path.name,
        "width": size,
        "height": size,
        "controls": control_vector.squeeze(0).tolist(),
        # Channel-last flat order, which is how the runtime's readback returns a tensor.
        "input": source.squeeze(0).permute(1, 2, 0).reshape(-1).tolist(),
        "output": output.squeeze(0).permute(1, 2, 0).reshape(-1).tolist(),
    }))
    print(f"Wrote {reference_path} — load /selftest.html in the app to check the runtime against it.")


if __name__ == "__main__":
    main()
