"""Smoke test that needs NO internet and NO downloaded data.

Draws fake "cards", then runs the real pipeline end to end:
synthetic distortion -> rectification -> retrieval -> metrics, for several methods and severities.
Use it to check your setup and to see the logging/metrics work before the real database exists.

    python -m cardrec.demo_offline
"""
import shutil

import cv2
import numpy as np
import pandas as pd

from . import config
from .evaluate import evaluate_method
from .make_queries import make_queries
from .methods import get_method

DEMO = config.DATA_DIR / "demo"


def make_fake_card(rng, idx, w=config.CANON_W, h=config.CANON_H):
    card = np.zeros((h, w, 3), np.uint8)
    card[:] = (30, 200, 240)                                  # yellow-ish border (BGR)
    cv2.rectangle(card, (25, 25), (w - 25, h - 25), (20, 20, 20), 3)
    art = np.zeros((300, w - 80, 3), np.uint8)
    art[:] = rng.integers(0, 255, size=3)
    for _ in range(30):
        col = tuple(int(c) for c in rng.integers(0, 255, size=3))
        p1 = (int(rng.integers(0, art.shape[1])), int(rng.integers(0, art.shape[0])))
        if rng.random() < 0.5:
            cv2.circle(art, p1, int(rng.integers(8, 60)), col, -1)
        else:
            p2 = (int(rng.integers(0, art.shape[1])), int(rng.integers(0, art.shape[0])))
            cv2.rectangle(art, p1, p2, col, -1)
    art = np.clip(art.astype(np.int16) + rng.normal(0, 12, art.shape), 0, 255).astype(np.uint8)
    card[90:390, 40:w - 40] = art
    cv2.putText(card, f"CARD {idx:03d}", (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 3)
    for r in range(6):
        y = 430 + r * 40
        cv2.line(card, (45, y), (int(rng.integers(150, w - 60)), y), (40, 40, 40), 3)
        cv2.putText(card, f"{int(rng.integers(0, 99999)):05d}", (w - 130, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    return card


def main(n_cards=30, n_queries=20, severities=(1, 3, 5)):
    rng = np.random.default_rng(config.SEED)
    shutil.rmtree(DEMO, ignore_errors=True)
    (DEMO / "cards").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_cards):
        p = DEMO / "cards" / f"fake_{i:03d}.png"
        cv2.imwrite(str(p), make_fake_card(rng, i))
        rows.append({"id": f"fake_{i:03d}", "abs_path": str(p)})
    manifest = pd.DataFrame(rows)
    paths, ids = manifest["abs_path"].tolist(), manifest["id"].tolist()

    methods = [get_method(m, k=10) for m in ("sift", "orb", "pixel", "hybrid-pixel")]
    for m in methods:
        m.build_index(paths, ids)

    results = []
    for sev in severities:
        q = make_queries(manifest, n_queries, sev, out_dir=DEMO / "queries" / f"s{sev}")
        for m in methods:
            summary, _ = evaluate_method(m, q, rectify_first=True)
            summary["severity"] = sev
            results.append(summary)
    df = pd.DataFrame(results)[["severity", "method", "n_queries", "top1", "top5", "mAP",
                                "ms_retrieval", "index_MB"]]
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
