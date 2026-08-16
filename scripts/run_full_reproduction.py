#!/usr/bin/env python3
"""Run and publish the 15-category DFR experiment one category at a time."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path


CATEGORIES = (
    "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill",
    "screw", "toothbrush", "transistor", "zipper", "carpet", "grid",
    "leather", "tile", "wood",
)
METRICS = ("det_pr", "det_auc", "seg_pr", "seg_auc", "seg_pro", "seg_iou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable 700-epoch DFR experiments and publish reports"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--metric-steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--branch", default="reproduction")
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--push", action=argparse.BooleanOptionalAction, default=True,
        help="Commit and push each validated category report",
    )
    return parser.parse_args()


def run_streamed(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def validate_report(report_dir: Path, category: str, epochs: int) -> None:
    report_path = report_dir / "dfr_mvtec_summary.csv"
    with report_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["category"]: row for row in csv.DictReader(handle)}
    if category not in rows:
        raise RuntimeError(f"Report does not contain completed category: {category}")
    row = rows[category]
    if int(row["epochs"]) != epochs:
        raise RuntimeError(
            f"{category}: expected report epoch {epochs}, found {row['epochs']}"
        )
    invalid = [name for name in METRICS if not math.isfinite(float(row[name]))]
    if invalid:
        raise RuntimeError(f"{category}: non-finite metrics: {', '.join(invalid)}")


def git_publish(root: Path, args: argparse.Namespace, category: str) -> None:
    paths = [
        str(args.report_dir / "dfr_mvtec_summary.csv"),
        str(args.report_dir / "dfr_mvtec_summary.md"),
        str(args.report_dir / "environment.json"),
        str(args.report_dir / "mvtec_validation.json"),
    ]
    subprocess.run(["git", "add", "--", *paths], cwd=root, check=True)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=root, check=False
    ).returncode != 0
    if not changed:
        print(f"No report changes to publish for {category}.", flush=True)
        return
    subprocess.run(
        ["git", "commit", "-m", f"Record {category} {args.epochs}-epoch DFR result"],
        cwd=root,
        check=True,
    )
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = (
        "ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 "
        "-o ServerAliveCountMax=3"
    )
    for attempt in range(1, 4):
        try:
            subprocess.run(
                ["git", "push", args.remote, args.branch],
                cwd=root,
                env=env,
                check=True,
                timeout=120,
            )
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 3:
                raise
            print(f"Push attempt {attempt}/3 failed; retrying.", flush=True)
            time.sleep(10)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).name != "dfr":
        raise RuntimeError(
            f"This runner must use the dfr Conda environment; got {sys.executable}"
        )
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_branch != args.branch:
        raise RuntimeError(
            f"Expected Git branch {args.branch!r}, found {current_branch!r}"
        )
    unknown = sorted(set(args.categories) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown categories: {', '.join(unknown)}")

    for name in ("data_root", "output_dir", "report_dir", "log_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (root / value).resolve())

    validation_path = args.report_dir / "mvtec_validation.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "validate_mvtec.py"),
            str(args.data_root),
            "--json",
            str(validation_path),
        ],
        cwd=root,
        check=True,
    )

    for category in args.categories:
        command = [
            sys.executable,
            str(root / "DFR-source" / "main.py"),
            "--mode", "all",
            "--data-root", str(args.data_root),
            "--categories", category,
            "--epochs", str(args.epochs),
            "--checkpoint-every", str(args.checkpoint_every),
            "--metric-steps", str(args.metric_steps),
            "--workers", str(args.workers),
            "--output-dir", str(args.output_dir),
            "--report-dir", str(args.report_dir),
            "--resume",
        ]
        run_streamed(command, args.log_dir / f"{category}-{args.epochs}.log")
        validate_report(args.report_dir, category, args.epochs)
        if args.push:
            git_publish(root, args, category)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
