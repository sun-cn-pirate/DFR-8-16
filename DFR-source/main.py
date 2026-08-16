"""Command-line entry point for the DFR MVTec AD reproduction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from anoseg_dfr import AnoSegDFR


TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
OBJECTS = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)
MVTEC_CATEGORIES = OBJECTS + TEXTURES
PAPER_FEATURE_LAYERS = (
    "relu1_1",
    "relu1_2",
    "relu2_1",
    "relu2_2",
    "relu3_1",
    "relu3_2",
    "relu3_3",
    "relu3_4",
    "relu4_1",
    "relu4_2",
    "relu4_3",
    "relu4_4",
)
METRIC_FIELDS = (
    "category",
    "epochs",
    "latent_dim",
    "train_seconds",
    "eval_seconds",
    "det_pr",
    "det_auc",
    "seg_pr",
    "seg_auc",
    "seg_pro",
    "seg_iou",
)
METRIC_VALUE_FIELDS = (
    "det_pr",
    "det_auc",
    "seg_pr",
    "seg_auc",
    "seg_pro",
    "seg_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce DFR on MVTec AD")
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "evaluation", "all"),
        default="all",
        help="Run training, evaluation, or both",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the 15 MVTec AD category folders",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="Category names, or 'all'",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-name", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--img-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--backbone", default="vgg19", choices=("vgg19",))
    parser.add_argument("--cnn-layers", nargs="+", default=PAPER_FEATURE_LAYERS)
    parser.add_argument("--upsample", choices=("nearest", "bilinear"), default="nearest")
    parser.add_argument("--aggregate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--featmap-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--kernel-size", type=int, nargs=2, default=(4, 4))
    parser.add_argument("--stride", type=int, nargs=2, default=(4, 4))
    parser.add_argument("--dilation", type=int, default=1)

    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--batch-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--checkpoint-every", type=int, default=10)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--expected-fpr", type=float, default=0.3)
    parser.add_argument("--metric-steps", type=int, default=5000)
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def select_categories(requested: list[str]) -> tuple[str, ...]:
    if requested == ["all"]:
        return MVTEC_CATEGORIES
    unknown = sorted(set(requested) - set(MVTEC_CATEGORIES))
    if unknown:
        raise ValueError(
            f"Unknown MVTec categories: {', '.join(unknown)}. "
            f"Choose from: {', '.join(MVTEC_CATEGORIES)}"
        )
    return tuple(dict.fromkeys(requested))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_runtime(args: argparse.Namespace, categories: tuple[str, ...]) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({args.device}), but CUDA is unavailable")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    for category in categories:
        category_dir = args.data_root / category
        required = (category_dir / "train" / "good", category_dir / "test")
        missing = [str(path) for path in required if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete MVTec category '{category}'; missing: {', '.join(missing)}"
            )


def build_category_config(args: argparse.Namespace, category: str) -> argparse.Namespace:
    cfg = argparse.Namespace(**vars(args))
    category_dir = args.data_root / category
    cfg.data_name = category
    cfg.train_data_path = str(category_dir / "train" / "good")
    cfg.test_data_path = str(category_dir / "test")
    cfg.save_path = str(args.output_dir)
    cfg.model_name = args.model_name
    cfg.img_size = tuple(args.img_size)
    cfg.cnn_layers = tuple(args.cnn_layers)
    cfg.is_agg = args.aggregate
    cfg.featmap_size = tuple(args.featmap_size)
    cfg.kernel_size = tuple(args.kernel_size)
    cfg.stride = tuple(args.stride)
    cfg.is_bn = args.batch_norm
    cfg.thred = args.threshold
    cfg.except_fpr = args.expected_fpr
    return cfg


def write_summary(rows: list[dict[str, float | str]], report_dir: Path, args: argparse.Namespace) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized.setdefault("epochs", 0)
        normalized.setdefault("latent_dim", 0)
        normalized.setdefault("train_seconds", 0.0)
        normalized.setdefault("eval_seconds", 0.0)
        normalized_rows.append(normalized)
    rows = normalized_rows
    csv_path = report_dir / "dfr_mvtec_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=METRIC_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    numeric_fields = METRIC_VALUE_FIELDS
    means = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in numeric_fields
    } if rows else {}
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "target_epochs": args.epochs,
        "categories_completed": len(rows),
        "categories_at_target": sum(
            int(row["epochs"]) >= args.epochs for row in rows
        ),
        "legacy_timing_rows": sum(
            int(row["epochs"]) > 0 and float(row["train_seconds"]) == 0.0
            for row in rows
        ),
        "means": means,
    }
    (report_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# DFR MVTec AD reproduction results",
        "",
        f"Reported categories: {len(rows)}/15; categories at the current "
        f"{args.epochs}-epoch target: "
        f"{sum(int(row['epochs']) >= args.epochs for row in rows)}/15.",
        "",
        "| Category | Epochs | PCA dim | Train (h) | Eval (s) | Det AP | Det AUC | Seg AP | Seg AUC | PRO-AUC | Best IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {epochs:d} | {latent_dim:d} | {train_hours:.3f} | "
            "{eval_seconds:.1f} | {det_pr:.5f} | {det_auc:.5f} | {seg_pr:.5f} | "
            "{seg_auc:.5f} | {seg_pro:.5f} | {seg_iou:.5f} |".format(
                **row,
                train_hours=float(row["train_seconds"]) / 3600.0,
            )
        )
    if rows:
        lines.append(
            "| **Mean metrics** |  |  |  |  | "
            + " | ".join(f"**{means[key]:.5f}**" for key in numeric_fields)
            + " |"
        )
    lines.extend([
        "",
        "A zero timing value marks a smoke result produced before cumulative timing metadata was introduced.",
        "",
        "The upstream project does not publish complete package versions or all random-state details; "
        "differences from the paper are reported without hidden tuning.",
        "",
    ])
    (report_dir / "dfr_mvtec_summary.md").write_text("\n".join(lines), encoding="utf-8")


def load_summary(report_dir: Path) -> list[dict[str, float | str]]:
    csv_path = report_dir / "dfr_mvtec_summary.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "category": row["category"],
            "epochs": int(row.get("epochs") or 0),
            "latent_dim": int(row.get("latent_dim") or 0),
            "train_seconds": float(row.get("train_seconds") or 0.0),
            "eval_seconds": float(row.get("eval_seconds") or 0.0),
            **{key: float(row[key]) for key in METRIC_VALUE_FIELDS},
        }
        for row in rows
        if row.get("category") in MVTEC_CATEGORIES
    ]


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.report_dir = args.report_dir.expanduser().resolve()
    categories = select_categories(args.categories)
    validate_runtime(args, categories)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode = "evaluate" if args.mode == "evaluation" else args.mode
    rows = load_summary(args.report_dir)
    for category in categories:
        print(f"\n{'=' * 72}\nDFR category: {category}\n{'=' * 72}", flush=True)
        cfg = build_category_config(args, category)
        dfr = AnoSegDFR(cfg)
        if mode in ("train", "all"):
            dfr.train(resume=args.resume)
        if mode in ("evaluate", "all"):
            evaluation_started = time.monotonic()
            metrics = dfr.metrics_evaluation(
                expect_fpr=args.expected_fpr,
                max_step=args.metric_steps,
                save_visualizations=args.save_visualizations,
            )
            if metrics is None:
                raise RuntimeError(f"Evaluation failed for category '{category}'")
            evaluation_seconds = time.monotonic() - evaluation_started
            rows = [row for row in rows if row["category"] != category]
            rows.append({
                "category": category,
                "epochs": args.epochs,
                "latent_dim": int(dfr.n_dim),
                "train_seconds": float(dfr.training_elapsed_seconds),
                "eval_seconds": evaluation_seconds,
                **metrics,
            })
            category_order = {name: index for index, name in enumerate(MVTEC_CATEGORIES)}
            rows.sort(key=lambda row: category_order[str(row["category"])])
            write_summary(rows, args.report_dir, args)
        del dfr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
