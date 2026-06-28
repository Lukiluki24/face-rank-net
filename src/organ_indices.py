"""
organ_indices.py — FaceRankNet
===============================
Canonical MediaPipe FaceMesh 468-landmark index mapping per anatomical organ.

Rules (from FaceRankNet AI Instruction):
  - Eyebrow nodes are merged into the corresponding eye sub-graph.
  - Keys: left_eye, right_eye, nose, mouth, jawline.
  - All indices are from the standard 468-point FaceMesh topology.

Reference: mediapipe/python/solutions/face_mesh_connections.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Left Eye landmarks (person's left — appears on RIGHT side of image)
# Includes left eyebrow merged in.
# ---------------------------------------------------------------------------
_LEFT_EYE: list[int] = [
    # eye contour
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]
_LEFT_EYEBROW: list[int] = [
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
]

# ---------------------------------------------------------------------------
# Right Eye landmarks (person's right — appears on LEFT side of image)
# Includes right eyebrow merged in.
# ---------------------------------------------------------------------------
_RIGHT_EYE: list[int] = [
    # eye contour
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]
_RIGHT_EYEBROW: list[int] = [
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
]

# ---------------------------------------------------------------------------
# Nose landmarks
# ---------------------------------------------------------------------------
_NOSE: list[int] = [
    1, 2, 5, 4, 6, 19, 94,
    125, 354,               # alar base corners
    129, 358,               # lateral nose
    168, 197, 195,          # bridge / tip
    45, 220, 115, 48,       # left side
    294, 440, 344, 278,     # right side
    64, 98, 97,             # left nostril
    326, 327, 294,          # right nostril
    141, 370,               # nasal wings
    44, 275,                # upper nose
]

# ---------------------------------------------------------------------------
# Mouth / Lips landmarks
# ---------------------------------------------------------------------------
_MOUTH: list[int] = [
    # outer lip
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
    # inner lip
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
]

# ---------------------------------------------------------------------------
# Jawline / Face Shape landmarks (face oval)
# ---------------------------------------------------------------------------
_JAWLINE: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]


def _dedupe_sorted(lst: list[int]) -> list[int]:
    """Remove duplicates while preserving a deterministic (sorted) order."""
    return sorted(set(lst))


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------
ORGAN_INDICES: dict[str, list[int]] = {
    "left_eye":  _dedupe_sorted(_LEFT_EYE + _LEFT_EYEBROW),
    "right_eye": _dedupe_sorted(_RIGHT_EYE + _RIGHT_EYEBROW),
    "nose":      _dedupe_sorted(_NOSE),
    "mouth":     _dedupe_sorted(_MOUTH),
    "jawline":   _dedupe_sorted(_JAWLINE),
}

# Sanity check: all indices must be within [0, 467]
for _organ, _idxs in ORGAN_INDICES.items():
    assert all(0 <= i <= 467 for i in _idxs), (
        f"Out-of-range index in organ '{_organ}'"
    )

if __name__ == "__main__":
    for organ, idxs in ORGAN_INDICES.items():
        print(f"{organ:12s}: {len(idxs):3d} nodes  — {idxs[:6]} …")
