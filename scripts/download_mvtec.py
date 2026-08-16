#!/usr/bin/env python3
"""Download the approved MVTec AD research mirror from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/mvtec_ad"))
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="foersben/mvtec-ad",
        repo_type="dataset",
        local_dir=output,
        allow_patterns=[
            "*/train/**",
            "*/test/**",
            "*/ground_truth/**",
            "*/license.txt",
            "*/readme.txt",
        ],
    )
    print(output)


if __name__ == "__main__":
    main()
