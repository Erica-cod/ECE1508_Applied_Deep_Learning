#!/usr/bin/env python3
"""
Utility for downloading one or more models from the Hugging Face Hub.

Usage example:
    python scripts/download_models.py \
        --model bert-base-uncased \
        --model google/flan-t5-base \
        --output-dir model
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def ensure_hfh_available() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub package is required. Install with "
            "`pip install huggingface_hub`."
        ) from exc


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models into a local directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Model identifier on the Hugging Face Hub (repeatable).",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        help="Optional newline-separated file with model ids to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model"),
        help="Directory where downloaded models will be stored.",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token; defaults to the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--revision",
        help="Optional specific revision (branch, tag, or commit) to download for all models.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        help="Restrict downloads to files matching these glob patterns (repeatable).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        help="Skip files matching these glob patterns (repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download models even if the target directory already exists.",
    )

    args = parser.parse_args(argv)
    models: List[str] = []

    if args.models_file:
        if not args.models_file.exists():
            parser.error(f"Models file not found: {args.models_file}")
        with args.models_file.open("r", encoding="utf-8") as fh:
            models.extend(
                line.strip()
                for line in fh
                if line.strip() and not line.strip().startswith("#")
            )

    if args.models:
        models.extend(args.models)

    if not models:
        parser.error("No models specified. Use --model or --models-file.")

    args.models = models
    return args


def download_models(
    models: Iterable[str],
    output_dir: Path,
    token: Optional[str],
    revision: Optional[str],
    allow_patterns: Optional[List[str]],
    ignore_patterns: Optional[List[str]],
    force: bool,
) -> None:
    from huggingface_hub import HfApi, snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)

    failures: List[str] = []

    for model_id in models:
        safe_name = model_id.replace("/", "__")
        target_dir = output_dir / safe_name

        if target_dir.exists() and not force:
            print(f"[skip] {model_id} already present at {target_dir}")
            continue

        if target_dir.exists() and force:
            print(f"[clean] removing existing directory {target_dir}")
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"[download] {model_id} -> {target_dir}")
        try:
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            info = api.model_info(model_id, revision=revision)
            print(
                f"[done] {model_id} | files: {len(info.siblings)} |"
                f" sha: {info.sha[:8]} | stored at: {target_dir}"
            )
        except Exception as err:  # noqa: BLE001
            failures.append(model_id)
            print(f"[error] {model_id}: {err}", file=sys.stderr)

    if failures:
        raise SystemExit(
            f"Downloads completed with errors. Failed models: {', '.join(failures)}"
        )


def main(argv: Optional[List[str]] = None) -> None:
    ensure_hfh_available()
    args = parse_args(argv)

    from os import getenv

    token = args.token or getenv("HF_TOKEN")

    download_models(
        models=args.models,
        output_dir=args.output_dir,
        token=token,
        revision=args.revision,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
        force=args.force,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
