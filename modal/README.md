# FaceRankNet on Modal

Run the FaceRankNet training loop on a Modal A100 instead of Colab T4.

## Scope

This folder migrates **training only** (Colab Cell 8). Landmark extraction
(Cell 4) and pseudo-label generation (Cell 5) still run in Colab — they're
cheap enough on CPU/T4 and the outputs are cached.

## Layout

```
modal/
├── app.py            # Modal app — image, volumes, train_remote function
├── upload_cache.py   # one-shot uploader for caches + CSVs
├── requirements.txt  # local deps (just `modal`)
└── README.md
```

## Volumes

| Name              | Mount         | Contents                                                                |
| ----------------- | ------------- | ----------------------------------------------------------------------- |
| `frn-data`        | `/data`       | `train_labels.csv`, `test_labels.csv`                                   |
| `frn-cache`       | `/cache`      | `train_landmarks.pkl`, `test_landmarks.pkl`, `pseudo_labels.pkl`, `avg_face.npy` |
| `frn-checkpoints` | `/checkpoints`| `checkpoint_best.pt`                                                    |

All three are created automatically on first use.

## Setup (one time)

```powershell
pip install -r modal/requirements.txt
modal token new        # opens browser; logs you in
```

## Populate volumes (one time)

After Colab Cells 4 + 5 finish, download these files from Drive to your
machine, then:

```powershell
python modal/upload_cache.py `
    --cache-dir "C:\path\to\cache" `
    --data-dir  "C:\path\to\data"
```

This pushes the 6 files (4 cache + 2 CSV) into the `frn-cache` and
`frn-data` volumes.

## Train

```powershell
# default — 50 epochs, batch 32, resume if checkpoint exists
modal run modal/app.py

# custom hyperparams
modal run modal/app.py --epochs 80 --batch-size 64 --lr 5e-4

# start fresh, ignore any existing checkpoint
modal run modal/app.py --no-resume


# Download training curves (they're in the frn-checkpoints volume)
modal volume get frn-checkpoints results.csv .
modal volume get frn-checkpoints results_simplified.csv .

modal volume get frn-checkpoints training_curves.png .
modal volume get frn-checkpoints training_curves_simplified.png .

modal volume get frn-checkpoints checkpoint_best_simplified.pt .
modal volume get frn-checkpoints checkpoint_best.pt .

# simplified gradnorm
modal run --detach modal/app.py::main_simplified

```

For long jobs use `--detach` so the run survives client disconnect:

```powershell
modal run --detach modal/app.py --epochs 100
```

Track detached runs at https://modal.com/apps.

## Fetching the checkpoint later

```powershell
modal volume get frn-checkpoints /checkpoint_best.pt ./checkpoint_best.pt
```

## Cost notes

A100-40GB on Modal: ~$1.32/hr. A full 50-epoch run on 5500 images / batch 32
typically lands well under an hour, so a single training run is ~$0.50–$1.

## Gotchas

- `train.py` reads several `config.*` paths at call time — `app.py` patches
  these to the mounted volume paths before importing `train`.
- DGL is installed with `--no-deps` (same as Colab) to keep torch pinned.
- `NUM_WORKERS=0` is enforced by `config.py`; do not change it on Modal —
  DGL graphs still can't be pickled across worker processes.
