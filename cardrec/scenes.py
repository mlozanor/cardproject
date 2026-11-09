"""Multi-card scenes: generation with exact ground truth, then detection + end-to-end evaluation.

    python -m cardrec.scenes make --n 100 --severity 2 --overlap 0.15
    python -m cardrec.scenes eval --method hybrid-clip --scenes scenes/s2_o15/scenes.json

Ground truth per scene: card ids, corner quads, visible fraction (1.0 = not covered by other cards).

Detection metrics (a detection is a true positive if it overlaps an unmatched true card with
IoU >= IOU_MIN; IoU = area(intersection) / area(union) of the two card polygons, matched greedily
by highest IoU):   precision = TP / detections,  recall = TP / true cards,  F1 = 2PR / (P + R).
End-to-end accuracy = true cards that were detected AND identified correctly (top-1) / all true cards.
One row per true card is logged, so results can be split by visible fraction (occlusion analysis).
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .evaluate import bootstrap_ci, rank_row
from .manifest import load_manifest
from .methods import get_method
from .methods.cache import build_or_load
from .rectify import find_card_quads, polygon_iou, warp_to_canonical
from .synth import render_scene

IOU_MIN = 0.5


def make_scenes(manifest, n, severity, max_overlap, n_cards=(2, 6), seed=config.SEED, out_dir=None):
    rng = np.random.default_rng(seed + 100 * severity + int(100 * max_overlap))
    out_dir = out_dir or config.DATA_DIR / "scenes" / f"s{severity}_o{int(100 * max_overlap)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = []
    for j in tqdm(range(n), desc="scenes"):
        k = int(rng.integers(n_cards[0], n_cards[1] + 1))
        picks = rng.choice(len(manifest), size=k, replace=False)
        cards = [cv2.imread(manifest.iloc[i]["abs_path"]) for i in picks]
        img, placed = render_scene(cards, rng, severity, max_overlap=max_overlap)
        path = out_dir / f"scene_{j:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        scenes.append({"scene_id": f"{out_dir.name}_{j:04d}", "severity": severity, "max_overlap": max_overlap,
                       "image_path": str(path.relative_to(config.DATA_DIR)).replace("\\", "/"),
                       "cards": [{"true_id": manifest.iloc[picks[p["index"]]]["id"],
                                  "quad": np.asarray(p["quad"]).round(1).tolist(),
                                  "visible_frac": round(p["visible_frac"], 3)} for p in placed]})
    (out_dir / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
    return out_dir / "scenes.json"


def match(dets, gts, shape):
    """Greedy IoU matching. Returns {gt_index: (det_index, iou)}."""
    pairs = sorted(((polygon_iou(d, g, shape), di, gi) for di, d in enumerate(dets) for gi, g in enumerate(gts)),
                   reverse=True)
    used_d, out = set(), {}
    for iou, di, gi in pairs:
        if iou < IOU_MIN:
            break
        if di not in used_d and gi not in out:
            used_d.add(di)
            out[gi] = (di, iou)
    return out


def evaluate_scenes(retriever, scenes, art_group=None, log_path=None):
    rows, n_det = [], 0
    for s in tqdm(scenes, desc=f"scenes {retriever.name}"):
        img = cv2.imread(str(config.DATA_DIR / s["image_path"]))
        dets = find_card_quads(img, max_cards=10)
        n_det += len(dets)
        gts = [c["quad"] for c in s["cards"]]
        m = match(dets, gts, img.shape)
        for gi, c in enumerate(s["cards"]):
            row = {"scene_id": s["scene_id"], "true_id": c["true_id"], "visible_frac": c["visible_frac"],
                   "n_cards": len(gts), "detected": gi in m, "iou": m[gi][1] if gi in m else 0.0}
            if gi in m:
                ranked = retriever.query(warp_to_canonical(img, dets[m[gi][0]]))
                row.update(rank_row(ranked, c["true_id"], art_group))
            else:
                row.update(top1=False, top1_lenient=False)
            rows.append(row)
    log = pd.DataFrame(rows)
    if log_path is not None:
        log.to_csv(log_path, index=False)
    tp, n_gt = int(log["detected"].sum()), len(log)
    p = tp / n_det if n_det else 0.0
    r = tp / n_gt if n_gt else 0.0
    e2e = log["top1"].astype(float)
    lo, hi = bootstrap_ci(e2e)
    return {"method": retriever.name, "n_scenes": len(scenes), "n_true_cards": n_gt, "n_detections": n_det,
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "mean_iou_matched": float(log.loc[log["detected"], "iou"].mean()) if tp else 0.0,
            "e2e_top1": float(e2e.mean()), "e2e_top1_lo": lo, "e2e_top1_hi": hi,
            "e2e_top1_len": float(log["top1_lenient"].astype(float).mean()),
            "id_acc_given_detected": float(log.loc[log["detected"], "top1"].astype(float).mean()) if tp else 0.0}, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "eval"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--severity", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--method", default="hybrid-clip")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--scenes", help="scenes.json, relative to the data folder")
    a = ap.parse_args()
    manifest = load_manifest()
    if a.cmd == "make":
        print("written:", make_scenes(manifest, a.n, a.severity, a.overlap))
    else:
        scenes = json.loads((config.DATA_DIR / a.scenes).read_text(encoding="utf-8"))
        method = build_or_load(get_method(a.method, k=a.k), manifest)
        config.RESULTS_DIR.mkdir(exist_ok=True)
        out = config.RESULTS_DIR / f"scenes_{method.name}_{scenes[0]['scene_id'].rsplit('_', 1)[0]}.csv"
        summary, _ = evaluate_scenes(method, scenes, dict(zip(manifest["id"], manifest["art_group"])), out)
        print(pd.Series(summary).to_string())
