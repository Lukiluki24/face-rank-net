"""
upload_cache.py — push local caches, CSVs, and checkpoint into Modal Volumes.

Typical first-time setup (after Colab Cell 4+5):
    python modal/upload_cache.py --cache-dir path/to/cache --data-dir path/to/data

Upload / overwrite a checkpoint only (resume from Colab run):
    python modal/upload_cache.py --checkpoint path/to/checkpoint_best.pt

Upload everything including a checkpoint:
    python modal/upload_cache.py --cache-dir path/to/cache --data-dir path/to/data \\
                                  --checkpoint path/to/checkpoint_best.pt

Expected cache layout
---------------------
cache_dir/
    train_landmarks.pkl
    test_landmarks.pkl
    pseudo_labels.pkl
    avg_face.npy

data_dir/
    train_labels.csv
    test_labels.csv

After uploading, run training:
    modal run modal/app.py                         # resume if checkpoint exists
    modal run modal/app.py --no-resume             # force fresh start
    modal run --detach modal/app.py                # detached (survives disconnect)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

CACHE_FILES = (
    "train_landmarks.pkl",
    "test_landmarks.pkl",
    "pseudo_labels.pkl",
    "avg_face.npy",
)
DATA_FILES = (
    "train_labels.csv",
    "test_labels.csv",
)


def _upload(volume_name: str, files: list[tuple[Path, str]]) -> None:
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)
    print(f"\n→ Uploading to volume '{volume_name}':")
    with vol.batch_upload(force=True) as batch:
        for local, remote in files:
            print(f"   {local}  →  {remote}")
            batch.put_file(local, remote)
    print(f"  done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory containing landmark .pkl + pseudo_labels + avg_face files.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing train_labels.csv and test_labels.csv.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        metavar="PATH",
        help="Local checkpoint_best.pt to upload to frn-checkpoints volume "
             "(enables resume in the next modal run modal/app.py).",
    )
    args = parser.parse_args()

    if args.cache_dir is None and args.data_dir is None and args.checkpoint is None:
        parser.print_help()
        raise SystemExit("\nError: provide at least one of --cache-dir, --data-dir, --checkpoint")

    # ── Cache files ────────────────────────────────────────────────────────────
    if args.cache_dir is not None:
        cache_files = []
        for name in CACHE_FILES:
            path = args.cache_dir / name
            if not path.exists():
                raise SystemExit(f"missing required cache file: {path}")
            cache_files.append((path, f"/{name}"))
        _upload("frn-cache", cache_files)

    # ── Data / CSV files ───────────────────────────────────────────────────────
    if args.data_dir is not None:
        data_files = []
        for name in DATA_FILES:
            path = args.data_dir / name
            if not path.exists():
                raise SystemExit(f"missing required data file: {path}")
            data_files.append((path, f"/{name}"))
        _upload("frn-data", data_files)

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    if args.checkpoint is not None:
        ckpt = args.checkpoint.resolve()
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        _upload("frn-checkpoints", [(ckpt, "/checkpoint_best.pt")])
        print("  Checkpoint uploaded. Next run will resume from this point.")

    print("\nAll uploads complete. You can now run:")
    print("    modal run modal/app.py             # resume from checkpoint")
    print("    modal run --detach modal/app.py    # detached (long runs)")


if __name__ == "__main__":
    main()
