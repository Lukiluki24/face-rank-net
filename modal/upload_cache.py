"""
upload_cache.py — push local caches and CSVs into Modal Volumes.

After you finish Cells 4 + 5 in Colab (landmark extraction + pseudo-labels),
download these files from Drive to your local machine, then run:

    python modal/upload_cache.py --cache-dir path/to/cache --data-dir path/to/data

Expected layout
---------------
cache_dir/
    train_landmarks.pkl
    test_landmarks.pkl
    pseudo_labels.pkl
    avg_face.npy

data_dir/
    train_labels.csv
    test_labels.csv

You can also use the Modal CLI directly:

    modal volume put frn-data  path/to/train_labels.csv
    modal volume put frn-cache path/to/avg_face.npy
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    cache_files = []
    for name in CACHE_FILES:
        path = args.cache_dir / name
        if not path.exists():
            raise SystemExit(f"missing required cache file: {path}")
        cache_files.append((path, f"/{name}"))

    data_files = []
    for name in DATA_FILES:
        path = args.data_dir / name
        if not path.exists():
            raise SystemExit(f"missing required data file: {path}")
        data_files.append((path, f"/{name}"))

    _upload("frn-cache", cache_files)
    _upload("frn-data", data_files)
    print("\nAll uploads complete. You can now run:")
    print("    modal run modal/app.py")


if __name__ == "__main__":
    main()
