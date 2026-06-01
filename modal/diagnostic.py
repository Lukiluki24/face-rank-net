"""
diagnostic.py — Graph topology + edge feature visualiser.

Usage:
    modal run modal/diagnostic.py                      # stats only
    modal run modal/diagnostic.py --download           # + download plot PNG
    modal run modal/diagnostic.py --download-to out.png
"""

from __future__ import annotations
from pathlib import Path
import modal

# Same image as app_dgf + matplotlib
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libgles2-mesa",
                 "libegl1", "libopengl0")
    .pip_install(
        "torch==2.3.0", "numpy>=2.0", "pandas", "scipy",
        "scikit-learn", "tqdm", "psutil", "networkx",
        "requests", "pyyaml", "pydantic",
        "mediapipe", "opencv-python-headless", "Pillow",
        "matplotlib",
    )
    .pip_install(
        "dgl",
        find_links="https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html",
        extra_options="--no-deps",
    )
    .add_local_dir(
        local_path=str(Path(__file__).parent.parent / "model"),
        remote_path="/root/face_rank_net",
        ignore=["*.ipynb", "__pycache__", "*.pyc"],
    )
    .add_local_dir(
        local_path=str(Path(__file__).parent.parent / "model_dgf"),
        remote_path="/root/face_rank_net_dgf",
        ignore=["__pycache__", "*.pyc"],
    )
)

cache_vol = modal.Volume.from_name("frn-cache",           create_if_missing=False)
ckpt_vol  = modal.Volume.from_name("frn-checkpoints-dgf", create_if_missing=True)

CACHE_DIR = "/cache"
CKPT_DIR  = "/checkpoints"

app = modal.App("frn-diagnostic", image=image)


@app.function(
    volumes={CACHE_DIR: cache_vol, CKPT_DIR: ckpt_vol},
    timeout=300,
)
def run_diagnostic() -> dict:
    import sys, pickle
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay

    sys.path.insert(0, "/root/face_rank_net_dgf")
    sys.path.insert(1, "/root/face_rank_net")

    import config
    config.AVG_FACE_CACHE = Path(CACHE_DIR) / "avg_face.npy"
    config.LANDMARK_CACHE_TRAIN = Path(CACHE_DIR) / "train_landmarks.pkl"

    from organ_indices import ORGAN_INDICES
    import preprocessing as pre   # resolves to model_dgf/preprocessing.py

    # ---- Load one face ----
    with open(config.LANDMARK_CACHE_TRAIN, "rb") as f:
        coords_cache = pickle.load(f)
    avg_face = np.load(str(config.AVG_FACE_CACHE))

    fname  = list(coords_cache.keys())[0]
    coords = coords_cache[fname]

    # ---- Edge builders ----
    def edges_baseline(indices):
        n = len(indices)
        return [(i, j) for i in range(n) for j in range(n) if i != j]

    def edges_delaunay(indices):
        pts  = coords[indices][:, :2]
        tri  = Delaunay(pts)
        seen = set()
        out  = []
        for a, b, c in tri.simplices:
            for u, v in [(a,b),(b,c),(a,c)]:
                k = (min(u,v), max(u,v))
                if k not in seen:
                    seen.add(k)
                    out += [(u,v),(v,u)]
        return out

    organs = list(ORGAN_INDICES.keys())

    # ---- Plot ----
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for col, organ in enumerate(organs):
        idxs = ORGAN_INDICES[organ]
        pts  = coords[idxs][:, :2]

        for row, (label, get_edges) in enumerate([
            ("Baseline\n(fully-connected)", edges_baseline),
            ("DGF\n(Delaunay)",             edges_delaunay),
        ]):
            ax  = axes[row][col]
            eds = get_edges(idxs)
            for u, v in eds:
                ax.plot([pts[u,0], pts[v,0]], [pts[u,1], pts[v,1]],
                        'steelblue', alpha=0.2, lw=0.6)
            ax.scatter(pts[:,0], pts[:,1], c='crimson', s=20, zorder=3)
            ax.set_title(f"{organ}\n({len(eds)} edges)", fontsize=8)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=9, labelpad=4)
                ax.axis('on')
                ax.set_yticks([])
                ax.set_xticks([])

    plt.suptitle(f"Graph topology comparison — {fname}", fontsize=11, y=1.01)
    plt.tight_layout()

    plot_path = Path(CKPT_DIR) / "graph_topology.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    ckpt_vol.commit()

    # ---- Edge feature stats per organ ----
    stats = {}
    feat_names = ["θ_ij", "y_ij", "γ_ij", "α_ij", "β_ij"]
    for organ, idxs in ORGAN_INDICES.items():
        g  = pre.build_subgraph(coords, idxs, avg_face=avg_face)
        ef = g.edata["efeat"].numpy()
        stats[organ] = {
            "n_edges": ef.shape[0],
            "has_nan": bool(np.isnan(ef).any()),
            "features": {
                name: {"mean": round(float(ef[:,i].mean()), 4),
                       "std":  round(float(ef[:,i].std()),  4)}
                for i, name in enumerate(feat_names)
            }
        }

    return {"plot_saved_to": str(plot_path), "edge_stats": stats}


@app.function(volumes={CKPT_DIR: ckpt_vol})
def fetch_plot() -> bytes:
    return (Path(CKPT_DIR) / "graph_topology.png").read_bytes()


@app.local_entrypoint()
def main(download: bool = False, download_to: str = "graph_topology.png"):
    result = run_diagnostic.remote()

    print("\n=== Edge Feature Stats ===")
    for organ, s in result["edge_stats"].items():
        print(f"\n{organ}  ({s['n_edges']} edges)  NaN={s['has_nan']}")
        for feat, vals in s["features"].items():
            print(f"  {feat}: mean={vals['mean']:+.4f}  std={vals['std']:.4f}")

    print(f"\nPlot saved → {result['plot_saved_to']}")

    if download:
        blob = fetch_plot.remote()
        Path(download_to).write_bytes(blob)
        print(f"Downloaded → {download_to}  ({len(blob)/1e3:.1f} KB)")
    else:
        print("Add --download to get the PNG locally.")
