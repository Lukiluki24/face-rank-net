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
    aug_train_landmarks.pkl   (OPTIONAL — pseudo-label-time MixUp / synthaug)

data_dir/
    train_labels.csv
    test_labels.csv
    aug_train_labels.csv      (OPTIONAL — synthaug rows)

Optional aug_* files are uploaded only if present. With them in the volume
and ``use_augmented_train=True`` (default) in app.py, Modal training uses
the augmented set; without them, it falls back to the original cache.

After uploading, run training:
    modal run modal/app.py                                  # resume + synthaug
    modal run modal/app.py --no-use-augmented-train         # original cache
    modal run modal/app.py --no-resume                      # fresh start
    modal run --detach modal/app.py                         # long runs
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
# Optional cache files — uploaded only if present in --cache-dir.
# aug_train_landmarks.pkl holds real + synth coords from Cell 5d (synthaug).
OPTIONAL_CACHE_FILES = (
    "aug_train_landmarks.pkl",
)
DATA_FILES = (
    "train_labels.csv",
    "test_labels.csv",
)
# Optional data files — same rule. aug_train_labels.csv carries the synth rows.
OPTIONAL_DATA_FILES = (
    "aug_train_labels.csv",
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
        # Augmented landmark cache is optional; include if user generated it.
        for name in OPTIONAL_CACHE_FILES:
            path = args.cache_dir / name
            if path.exists():
                cache_files.append((path, f"/{name}"))
                print(f"   (optional) found {name}")
        _upload("frn-cache", cache_files)

    # ── Data / CSV files ───────────────────────────────────────────────────────
    if args.data_dir is not None:
        data_files = []
        for name in DATA_FILES:
            path = args.data_dir / name
            if not path.exists():
                raise SystemExit(f"missing required data file: {path}")
            data_files.append((path, f"/{name}"))
        # Augmented CSV is optional; include if user generated it.
        for name in OPTIONAL_DATA_FILES:
            path = args.data_dir / name
            if path.exists():
                data_files.append((path, f"/{name}"))
                print(f"   (optional) found {name}")
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
