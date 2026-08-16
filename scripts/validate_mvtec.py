#!/usr/bin/env python3
"""Validate the original MVTec AD directory structure and public image counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Counts from the public MVTec AD dataset paper: train, test normal, test anomalous.
EXPECTED_COUNTS = {
    "bottle": (209, 20, 63),
    "cable": (224, 58, 92),
    "capsule": (219, 23, 109),
    "carpet": (280, 28, 89),
    "grid": (264, 21, 57),
    "hazelnut": (391, 40, 70),
    "leather": (245, 32, 92),
    "metal_nut": (220, 22, 93),
    "pill": (267, 26, 141),
    "screw": (320, 41, 119),
    "tile": (230, 33, 84),
    "toothbrush": (60, 12, 30),
    "transistor": (213, 60, 40),
    "wood": (247, 19, 60),
    "zipper": (240, 32, 119),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_category(root: Path, category: str) -> dict[str, int]:
    category_dir = root / category
    train = image_files(category_dir / "train" / "good")
    test_good = image_files(category_dir / "test" / "good")
    test_anomalous = [
        path for path in image_files(category_dir / "test")
        if path.parent.name != "good"
    ]
    masks = image_files(category_dir / "ground_truth")
    actual = (len(train), len(test_good), len(test_anomalous))
    expected = EXPECTED_COUNTS[category]
    if actual != expected:
        raise ValueError(
            f"{category}: expected train/good/anomalous {expected}, found {actual}"
        )
    if len(masks) != len(test_anomalous):
        raise ValueError(
            f"{category}: found {len(test_anomalous)} anomalous tests but {len(masks)} masks"
        )
    missing_masks = []
    for image in test_anomalous:
        expected_mask = (
            category_dir
            / "ground_truth"
            / image.parent.name
            / f"{image.stem}_mask.png"
        )
        if not expected_mask.is_file():
            missing_masks.append(str(expected_mask))
    if missing_masks:
        raise ValueError(
            f"{category}: {len(missing_masks)} masks are missing; first: {missing_masks[0]}"
        )
    return {
        "train": len(train),
        "test_good": len(test_good),
        "test_anomalous": len(test_anomalous),
        "masks": len(masks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    report = {
        category: validate_category(root, category)
        for category in EXPECTED_COUNTS
    }
    totals = {
        key: sum(row[key] for row in report.values())
        for key in ("train", "test_good", "test_anomalous", "masks")
    }
    if totals["train"] + totals["test_good"] + totals["test_anomalous"] != 5354:
        raise ValueError(f"Expected 5,354 MVTec images, found totals: {totals}")
    payload = {"categories": report, "totals": totals}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
