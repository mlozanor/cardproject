"""Generate single-card synthetic queries from the reference database.

Usage:
    python -m cardrec.make_queries --n 200 --severity 3
Writes <data>/queries/s<severity>/q_XXXX.jpg and <data>/queries/s<severity>/queries.csv
(the card images are known, so ground truth is exact; true_quad = the card's 4 corners in the image,
used to measure rectification quality).
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd

from . import config
from .manifest import load_manifest
from .synth import render_scene


def make_queries(manifest, n, severity, seed=config.SEED, out_dir=None, canvas=(960, 1280)):
    """canvas = scene size (h, w). The default puts a card ~530 px tall in the scene; a 12 MP phone
    photo has ~2000 px, so the language and condition tests use a larger canvas."""
    rng = np.random.default_rng(seed + severity)
    out_dir = out_dir or (config.QUERIES_DIR / f"s{severity}")
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = rng.choice(len(manifest), size=min(n, len(manifest)), replace=False)
    rows = []
    for j, idx in enumerate(picks):
        card = cv2.imread(manifest.iloc[idx]["abs_path"])
        scene, placed = render_scene([card], rng, severity, canvas=canvas)
        if not placed:
            continue
        path = out_dir / f"q_{j:04d}.jpg"                 # JPEG like a phone photo (and ~6x smaller)
        cv2.imwrite(str(path), scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"query_id": f"s{severity}_{j:04d}",
                     "image_path": str(path.resolve().relative_to(config.DATA_DIR)).replace("\\", "/"),
                     "true_id": manifest.iloc[idx]["id"], "severity": severity,
                     "true_quad": json.dumps(np.asarray(placed[0]["quad"]).round(1).tolist())})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "queries.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    df = make_queries(load_manifest(), a.n, a.severity)
    print(f"{len(df)} queries written to {config.QUERIES_DIR / f's{a.severity}'}")
