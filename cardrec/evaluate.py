"""Step 3: run a method over a query set, log EVERY query, and compute the metrics.

Usage:
    python -m cardrec.evaluate --method sift --queries queries/s3/queries.csv
    python -m cardrec.evaluate --method hybrid-clip --k 20 --queries ... --no-rectify

queries.csv columns: query_id, image_path (relative to config.DATA_DIR), true_id, severity
Per-query log goes to results/<method>_<tag>.csv; the summary is printed.

Metrics (each query has exactly one correct card):
  strict   correct = the exact card id.
  lenient  correct = any card with the same art_group (same artwork, e.g. a reprint in another set).
  top1 / top5 = fraction of queries whose correct card is ranked 1st / in the first 5.
  mAP: with one relevant card, average precision = 1/rank, so mAP == mean reciprocal rank (MRR).
  *_lo / *_hi = 95% bootstrap confidence interval: resample the queries with replacement
  N_BOOT times, recompute the mean each time, keep the 2.5% and 97.5% percentiles.
"""
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd

from . import config
from .manifest import load_manifest
from .methods import get_method
from .methods.cache import build_or_load
from .rectify import polygon_iou, rectify

N_BOOT = 1000


def bootstrap_ci(values, n_boot=N_BOOT, seed=config.SEED):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def first_rank(ids, ok):
    """1-based rank of the first id for which ok(id) is True (len+1 if none)."""
    return next((r for r, cid in enumerate(ids, 1) if ok(cid)), len(ids) + 1)


def rank_row(ranked, true_id, art_group=None):
    """Strict + lenient ranks and hit flags for one query."""
    ids = [cid for cid, _ in ranked]
    rank = first_rank(ids, lambda c: c == true_id)
    g = (art_group or {}).get(true_id, true_id)
    rank_len = first_rank(ids, lambda c: (art_group or {}).get(c, c) == g)
    return {"pred_id": ids[0] if ids else "", "pred_score": ranked[0][1] if ranked else None,
            "second_score": ranked[1][1] if len(ranked) > 1 else None,
            "rank_of_true": rank, "top1": rank == 1, "top5": rank <= 5, "rr": 1.0 / rank,
            "top1_lenient": rank_len == 1, "top5_lenient": rank_len <= 5, "rr_lenient": 1.0 / rank_len}


def summarize(log, name, index_bytes=0):
    s = {"method": name, "n_queries": len(log)}
    for col, out in [("top1", "top1"), ("top5", "top5"), ("rr", "mAP"),
                     ("top1_lenient", "top1_len"), ("rr_lenient", "mAP_len")]:
        s[out] = float(log[col].mean())
        s[out + "_lo"], s[out + "_hi"] = bootstrap_ci(log[col])
    total = log["ms_rectify"] + log["ms_retrieval"]
    s.update(ms_retrieval=float(log["ms_retrieval"].mean()), ms_rectify=float(log["ms_rectify"].mean()),
             ms_total_median=float(total.median()), ms_total_p95=float(total.quantile(0.95)),
             rectify_found=float(log["rectify_found"].mean()) if log["rectify_found"].notna().all() else None,
             rect_ok=float((log["rect_iou"] >= 0.8).mean()) if "rect_iou" in log and log["rect_iou"].notna().all() else None,
             index_MB=index_bytes / 1e6)
    return s


def evaluate_method(retriever, queries, rectify_first=True, log_path=None, art_group=None):
    rows = []
    for q in queries.itertuples():
        img = cv2.imread(str(config.DATA_DIR / q.image_path))
        t0 = time.perf_counter()
        if rectify_first:
            card, quad, found = rectify(img)
        else:
            card, quad, found = cv2.resize(img, (config.CANON_W, config.CANON_H)), None, None
        t1 = time.perf_counter()
        ranked = retriever.query(card)
        t2 = time.perf_counter()
        # how well the found outline matches the true one (synthetic queries only; 0 = nothing found)
        true_quad = getattr(q, "true_quad", None)
        rect_iou = None
        if rectify_first and isinstance(true_quad, str) and true_quad:
            rect_iou = polygon_iou(quad, json.loads(true_quad), img.shape) if quad is not None else 0.0
        rows.append({"query_id": q.query_id, "true_id": q.true_id, **rank_row(ranked, q.true_id, art_group),
                     "rectify_found": found, "rect_iou": rect_iou,
                     "ms_rectify": 1000 * (t1 - t0), "ms_retrieval": 1000 * (t2 - t1),
                     "severity": getattr(q, "severity", "")})
    log = pd.DataFrame(rows)
    if log_path is not None:
        log.to_csv(log_path, index=False)
    return summarize(log, retriever.name, retriever.index_size_bytes()), log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--queries", required=True, help="path relative to the data folder, or absolute")
    ap.add_argument("--k", type=int, default=20, help="shortlist size for hybrid-* methods")
    ap.add_argument("--no-rectify", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    manifest = load_manifest()
    queries = pd.read_csv(config.DATA_DIR / a.queries, dtype=str)
    if a.limit:
        queries = queries.head(a.limit)
    method = build_or_load(get_method(a.method, k=a.k), manifest)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"{method.name}_{a.tag}.csv"
    summary, _ = evaluate_method(method, queries, not a.no_rectify, out,
                                 dict(zip(manifest["id"], manifest["art_group"])))
    print(pd.Series(summary).to_string())
    print("per-query log:", out)
