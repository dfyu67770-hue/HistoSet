"""HistoSet training entry point."""

from __future__ import annotations

import argparse

from histoset.torch_engine import TrainingConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Prepared HistoSet manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument(
        "--mode",
        default="histoset_hierarchy",
        choices=[
            "histoset_hierarchy",
            "soft_explanation_ce",
            "hard_explanation_ce",
            "soft_pattern_ce",
            "hard_pattern_ce",
        ],
    )
    parser.add_argument("--seed", type=int, default=409)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hierarchy-weight", type=float, default=0.25)
    parser.add_argument("--pattern-weight", type=float, default=0.50)
    parser.add_argument("--dice-weight", type=float, default=0.85)
    parser.add_argument("--tumor-margin-weight", type=float, default=0.20)
    parser.add_argument("--tumor-margin", type=float, default=0.20)
    args = parser.parse_args()

    outputs = train_model(
        TrainingConfig(
            manifest=args.manifest,
            output_dir=args.output_dir,
            mode=args.mode,
            seed=args.seed,
            epochs=args.epochs,
            patch_size=args.patch_size,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            base_channels=args.base_channels,
            device=args.device,
            hierarchy_weight=args.hierarchy_weight,
            pattern_weight=args.pattern_weight,
            dice_weight=args.dice_weight,
            tumor_margin_weight=args.tumor_margin_weight,
            tumor_margin=args.tumor_margin,
        )
    )
    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
