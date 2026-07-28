"""Run HistoSet inference on prepared NPZ tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from histoset.torch_engine import build_model
from histoset.torch_losses import remap_explanation_to_pattern


def iter_npz_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.npz") if p.is_file())


def predict_one(model: torch.nn.Module, npz_path: Path, save_dir: Path) -> dict[str, object]:
    data = np.load(npz_path)
    image = torch.from_numpy(data["image"]).permute(2, 0, 1).float()[None] / 255.0
    with torch.no_grad():
        output = model(image)
        explanation_probability = F.softmax(output["explanation_logits"], dim=1)
        pattern_probability = remap_explanation_to_pattern(explanation_probability)
        direct_pattern_probability = (
            F.softmax(output["pattern_logits"], dim=1) if "pattern_logits" in output else pattern_probability
        )
    output_path = save_dir / f"{npz_path.stem}_histoset_prediction.npz"
    np.savez_compressed(
        output_path,
        explanation_probability=explanation_probability[0].cpu().numpy().astype(np.float32),
        pattern_probability=pattern_probability[0].cpu().numpy().astype(np.float32),
        direct_pattern_probability=direct_pattern_probability[0].cpu().numpy().astype(np.float32),
    )
    return {"input": str(npz_path), "prediction": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--save-path", required=True, type=Path)
    parser.add_argument("--mode", default="histoset_hierarchy")
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Image path does not exist: {args.images}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model = build_model(args.mode, base_channels=args.base_channels)
    model.load_state_dict(state_dict)
    model.eval()

    args.save_path.mkdir(parents=True, exist_ok=True)
    records = [predict_one(model, npz_path, args.save_path) for npz_path in iter_npz_inputs(args.images)]
    manifest = {"checkpoint": str(args.checkpoint), "mode": args.mode, "predictions": records}
    manifest_path = args.save_path / "histoset_prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"predictions={len(records)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
