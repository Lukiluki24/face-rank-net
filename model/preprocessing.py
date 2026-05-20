"""
preprocessing.py — FaceRankNet
================================
Handles:
  1. MediaPipe landmark extraction from raw face images.
  2. Centroid + inter-ocular normalisation of 468 3-D coordinates.
  3. Building fully-connected DGL sub-graphs per anatomical organ.
  4. Dataset-level preprocessing with result caching to a .pkl file.

Coding conventions (from FaceRankNet AI Instruction):
  - Python 3.10, type hints on all function signatures.
  - No pixel data is kept beyond this module; coords only.
  - Coordinate tensor shape: (N_nodes, 3) — columns are normalised X, Y, Z.
  - DGL graphs: fully connected + self-loops via dgl.add_self_loop().
"""

from __future__ import annotations

import logging
import os
import pickle
from itertools import product
from pathlib import Path

import cv2
import dgl
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from organ_indices import ORGAN_INDICES

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh singleton
# Supports both APIs:
#   Legacy  (mediapipe <= 0.10.14): mp.solutions.face_mesh
#   New     (mediapipe >= 0.10.15): mp.tasks / FaceLandmarker
# ---------------------------------------------------------------------------
_USE_LEGACY_MP: bool = (
    hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
)

if _USE_LEGACY_MP:
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
else:
    import urllib.request
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision

    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    _model_path = "/tmp/face_landmarker.task"
    if not os.path.exists(_model_path):
        logger.info("Downloading MediaPipe FaceLandmarker model (~30 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _model_path)
        logger.info("Model saved to %s", _model_path)

    _face_mesh = _mp_vision.FaceLandmarker.create_from_options(
        _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
    )

# Inter-ocular landmark indices (outer eye corners)
_LEFT_OUTER_CORNER: int = 33
_RIGHT_OUTER_CORNER: int = 263


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_landmarks(image_path: str) -> np.ndarray | None:
    """
    Run MediaPipe Face Mesh on a single image.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the face image (JPEG/PNG).

    Returns
    -------
    np.ndarray | None
        Float32 array of shape (468, 3) with normalised [0, 1] MediaPipe
        coordinates (x, y, z), or ``None`` if no face was detected.

    Notes
    -----
    Pixels are only used here and immediately discarded after landmark
    extraction — the model pipeline is purely geometric.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        logger.warning("Cannot read image: %s", image_path)
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if _USE_LEGACY_MP:
        result = _face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            logger.debug("No face detected in: %s", image_path)
            return None
        lm = result.multi_face_landmarks[0].landmark
        coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _face_mesh.detect(mp_image)
        if not result.face_landmarks:
            logger.debug("No face detected in: %s", image_path)
            return None
        lm = result.face_landmarks[0]
        coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    # The new Tasks API returns 478 landmarks (468 face mesh + 10 iris).
    # Organ indices only reference the canonical 468 FaceMesh nodes,
    # so we always truncate to the first 468.
    return coords[:468]  # shape (468, 3), in [0,1] MediaPipe space


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    Apply centroid-centering and inter-ocular scale normalisation.

    Steps
    -----
    1. Centroid centering  : subtract mean of all 468 nodes (per axis).
    2. Inter-ocular scale  : divide by Euclidean distance between the outer
                             eye corners (left idx 33, right idx 263).

    Parameters
    ----------
    coords : np.ndarray
        Shape (468, 3) — raw MediaPipe landmark coordinates.

    Returns
    -------
    np.ndarray
        Shape (468, 3), float32, centred and scale-normalised.
    """
    # Silently truncate — the Tasks API returns 478 (468 face + 10 iris).
    # We only need the canonical 468 FaceMesh landmarks.
    if coords.shape[0] > 468:
        coords = coords[:468]

    if coords.shape != (468, 3):
        raise ValueError(
            f"Expected coords shape (468, 3), got {coords.shape}"
        )

    # 1. Centroid centering
    centroid = coords.mean(axis=0)          # (3,)
    coords_c = coords - centroid            # (468, 3)

    # 2. Inter-ocular distance (using centred coords)
    iod = float(
        np.linalg.norm(
            coords_c[_LEFT_OUTER_CORNER] - coords_c[_RIGHT_OUTER_CORNER]
        )
    )
    if iod < 1e-8:
        logger.warning(
            "Inter-ocular distance is near zero (%.2e); skipping scale.",
            iod,
        )
        return coords_c.astype(np.float32)

    return (coords_c / iod).astype(np.float32)


def build_subgraph(
    coords: np.ndarray,
    indices: list[int],
    avg_face: np.ndarray | None = None,
) -> dgl.DGLGraph:
    """
    Build a fully-connected DGL graph for one anatomical organ sub-graph.

    Parameters
    ----------
    coords : np.ndarray
        Shape (468, 3) — **normalised** landmark coordinates for the whole face.
    indices : list[int]
        MediaPipe landmark indices belonging to this organ.
    avg_face : np.ndarray | None
        Shape (468, 3) — Universal Average Face (training-set mean).
        When provided, node features are extended with per-landmark deviation
        (Δx, Δy, Δz) from the average face, making the Averageness signal
        explicit at the input level.  Results in NODE_FEAT_DIM = 6.

    Returns
    -------
    dgl.DGLGraph
        Graph with ``len(indices)`` nodes.
        Node feature ``'feat'`` : float32 tensor of shape (N_nodes, 3 or 6).
        Self-loops are included (via ``dgl.add_self_loop``).
    """
    n = len(indices)

    if n == 0:
        raise ValueError("indices must be a non-empty list.")

    # Fully-connected edges (all ordered pairs, including i→i for self-loops)
    # We build non-self edges first, then add_self_loop handles self-edges.
    src_list, dst_list = zip(
        *[(i, j) for i, j in product(range(n), range(n)) if i != j]
    ) if n > 1 else ([], [])

    g = dgl.graph((list(src_list), list(dst_list)), num_nodes=n)
    g = dgl.add_self_loop(g)

    # Base node features: normalised 3-D coordinates
    organ_coords = coords[indices]          # (N_nodes, 3)

    if avg_face is not None:
        # Augment with per-node deviation from the Universal Average Face.
        # This makes the Averageness signal explicit rather than requiring
        # the GAT to infer it from absolute coordinates alone.
        deviation = organ_coords - avg_face[indices]   # (N_nodes, 3)
        node_feats = np.concatenate([organ_coords, deviation], axis=1)  # (N_nodes, 6)
    else:
        node_feats = organ_coords           # (N_nodes, 3)

    g.ndata["feat"] = torch.tensor(node_feats, dtype=torch.float32)

    return g


def build_all_subgraphs(
    coords: np.ndarray,
    avg_face: np.ndarray | None = None,
) -> dict[str, dgl.DGLGraph]:
    """
    Convenience wrapper: build all 5 organ sub-graphs from a single face.

    Parameters
    ----------
    coords : np.ndarray
        Shape (468, 3) — normalised landmark coordinates.
    avg_face : np.ndarray | None
        Shape (468, 3) — Universal Average Face. Passed through to
        ``build_subgraph`` to enable 6-dim node features.

    Returns
    -------
    dict[str, dgl.DGLGraph]
        Keys match ``ORGAN_INDICES``: ``left_eye``, ``right_eye``,
        ``nose``, ``mouth``, ``jawline``.
    """
    return {
        organ: build_subgraph(coords, idxs, avg_face=avg_face)
        for organ, idxs in ORGAN_INDICES.items()
    }


def preprocess_dataset(
    image_dir: str,
    csv_path: str,
    cache_path: str,
) -> dict[str, np.ndarray]:
    """
    Run landmark extraction + normalisation on every image in the dataset
    and persist the result as a pickle cache.

    Parameters
    ----------
    image_dir : str
        Directory containing the raw SCUT-FBP5500 face images.
    csv_path : str
        CSV file with at least a column ``Filename`` (and optionally
        ``Rating`` for labels).  Only the ``Filename`` column is used here.
    cache_path : str
        Output ``.pkl`` file path.  Will be overwritten if it already exists.

    Returns
    -------
    dict[str, np.ndarray]
        Maps each successfully processed filename → shape (468, 3) float32
        normalised coordinate array.

    Side-effects
    ------------
    Saves the returned dict to ``cache_path`` via ``pickle``.
    Logs the total number of images skipped (no face detected or unreadable).
    """
    image_dir_p = Path(image_dir)
    csv_path_p = Path(csv_path)
    cache_path_p = Path(cache_path)

    if not image_dir_p.is_dir():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not csv_path_p.is_file():
        raise FileNotFoundError(f"csv_path not found: {csv_path}")

    df = pd.read_csv(csv_path_p)
    if "Filename" not in df.columns:
        raise ValueError("CSV must contain a 'Filename' column.")

    filenames: list[str] = df["Filename"].tolist()
    logger.info(
        "Preprocessing %d images from '%s' …", len(filenames), image_dir
    )

    coords_cache: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    for fname in tqdm(filenames, desc="Extracting landmarks", unit="img"):
        img_path = str(image_dir_p / fname)

        raw = extract_landmarks(img_path)
        if raw is None:
            skipped.append(fname)
            continue

        try:
            normed = normalize_coords(raw)
        except ValueError as exc:
            logger.warning(
                "Normalisation failed for '%s': %s — skipping.", fname, exc
            )
            skipped.append(fname)
            continue

        coords_cache[fname] = normed

    # Persist cache
    cache_path_p.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path_p, "wb") as f:
        pickle.dump(coords_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        "Done. Saved %d entries to '%s'. Skipped %d image(s).",
        len(coords_cache),
        cache_path_p,
        len(skipped),
    )
    if skipped:
        logger.info("Skipped filenames: %s", skipped)

    return coords_cache


def load_coords_cache(cache_path: str) -> dict[str, np.ndarray]:
    """
    Load a previously saved landmark cache from disk.

    Parameters
    ----------
    cache_path : str
        Path to the ``.pkl`` file generated by :func:`preprocess_dataset`.

    Returns
    -------
    dict[str, np.ndarray]
        Maps filename → (468, 3) float32 normalised coords.
    """
    with open(cache_path, "rb") as f:
        cache: dict[str, np.ndarray] = pickle.load(f)
    logger.info("Loaded %d cached face entries from '%s'.", len(cache), cache_path)
    return cache


# ---------------------------------------------------------------------------
# CLI entry-point (for use in Colab via %run preprocessing.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FaceRankNet — preprocess SCUT-FBP5500 dataset"
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Path to SCUT-FBP5500 image directory",
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to CSV file containing 'Filename' column",
    )
    parser.add_argument(
        "--cache_path",
        default="landmark_cache.pkl",
        help="Output pickle file path (default: landmark_cache.pkl)",
    )
    args = parser.parse_args()

    preprocess_dataset(
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        cache_path=args.cache_path,
    )
