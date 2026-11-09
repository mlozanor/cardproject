"""Step 9: failure analysis - give every failed query ONE cause, with fixed rules, and count them.

    python -m cardrec.failure_analysis            # after cardrec.experiments (and real_photos eval)

Rules, checked in this order (the first that applies is the cause):
  localization        rect_iou < 0.5: the card outline was not found, or something else was found
  rectification       0.5 <= rect_iou < 0.8: the card was found but the outline is inaccurate
  upside down         (only for rotation-sensitive methods: not SIFT/ORB, not "+rot")
                      the warp cannot tell up from down. Measured (every 5 degrees, no perspective):
                      a card rotated by -45..+135 degrees comes out upright, outside that window it
                      comes out rotated by 180 degrees (the window is set by which corner has the
                      smallest x+y). The true rotation comes from the true corner order.
  same artwork        the prediction shares the true card's art_group (a reprint, e.g. JTG / ASC)
  lookalike           the prediction has the same NAME as the true card (alternate art, other version)
  heavy distortion    severity 4-5 (strong glare, blur, noise, lighting) and none of the above
  other               none of the above
Multi-card scenes (per true card): not detected -> "occluded" if less than 90% of it was visible,
else "missed (no overlap)"; detected but wrong -> same rules as above (from "same artwork" on).
Real photos: failure rate split by every condition column of labels.csv (glare, sleeve, ...).
Outputs: results/failures.csv (one row per failure), results/summary_failures.csv (counts per
method and cause), results/failures/<cause>_<n>.jpg (a few example photos per cause).
"""
import json

import cv2
import numpy as np
import pandas as pd

from . import config
from .manifest import load_manifest

CAUSES = ["localization", "rectification", "upside down", "same artwork", "lookalike", "heavy distortion", "other"]
EXAMPLES_PER_CAUSE = 4
UPRIGHT_WINDOW = (-45, 135)         # degrees: rotations that the warp returns upright (measured)


def rotation_deg(quad):
    """In-plane angle of the card's top edge (corner order = the card's own top-left, top-right, ...)."""
    q = np.asarray(quad, float)
    v = q[1] - q[0]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def rotation_sensitive(method):
    """SIFT/ORB are rotation invariant and '+rot' tries both orientations: they cannot fail 'upside down'."""
    return not (method.startswith(("sift", "orb")) or "+rot" in method)


def cause(row, art, name, method):
    iou = row.get("rect_iou")
    if iou is not None and not pd.isna(iou):
        if iou < 0.5:
            return "localization"
        if iou < 0.8:
            return "rectification"
    if rotation_sensitive(method) and isinstance(row.get("true_quad"), str) and row["true_quad"]:
        a = rotation_deg(json.loads(row["true_quad"]))
        if not UPRIGHT_WINDOW[0] < a < UPRIGHT_WINDOW[1]:
            return "upside down"
    return identity_cause(row, art, name)


def identity_cause(row, art, name):
    t, p = row["true_id"], row["pred_id"]
    if art.get(t, t) == art.get(p, p):
        return "same artwork"
    if name.get(t) and name.get(t) == name.get(p):
        return "lookalike"
    if str(row.get("severity", "")) in ("4", "5"):
        return "heavy distortion"
    return "other"


def single_card(manifest, art, name):
    rows = []
    for log_path in sorted((config.RESULTS_DIR / "main").glob("*.csv")):
        method, sev = log_path.stem.rsplit("_s", 1)
        log = pd.read_csv(log_path, dtype={"query_id": str, "true_id": str, "pred_id": str})
        q = pd.read_csv(config.QUERIES_DIR / f"s{sev}" / "queries.csv", dtype=str)
        log = log.merge(q[["query_id", "image_path", "true_quad"]], on="query_id", how="left")
        for r in log[~log["top1"].astype(bool)].to_dict("records"):
            rows.append({"set": "synthetic", "method": method, "severity": sev, "query_id": r["query_id"],
                         "true_id": r["true_id"], "pred_id": r["pred_id"], "rect_iou": r.get("rect_iou"),
                         "image_path": r["image_path"], "cause": cause(r, art, name, method)})
    return rows


def scenes(art, name):
    rows = []
    for log_path in sorted((config.RESULTS_DIR / "scenes").glob("*.csv")):
        method, ov = log_path.stem.rsplit("_o", 1)
        log = pd.read_csv(log_path, dtype={"true_id": str, "pred_id": str})
        for r in log[~log["top1"].astype(bool)].to_dict("records"):
            if not r["detected"]:
                c = "occluded" if r["visible_frac"] < 0.9 else "missed (no overlap)"
            else:
                c = identity_cause(r, art, name)
            rows.append({"set": f"scenes overlap {int(ov)}%", "method": method, "severity": "2",
                         "query_id": r["scene_id"], "true_id": r["true_id"], "pred_id": r.get("pred_id", ""),
                         "rect_iou": r["iou"], "image_path": "", "cause": c})
    return rows


def real_conditions():
    """Failure rate by condition column for the real single-card photos (if evaluated)."""
    out = []
    for log_path in sorted((config.RESULTS_DIR / "real").glob("*_single.csv")):
        log = pd.read_csv(log_path, dtype=str).fillna("")
        log["failed"] = log["top1"] != "True"
        for col in ["glare", "sleeve", "blur", "background", "lighting", "language", "overlap"]:
            if col in log and log[col].ne("").any():
                g = log.groupby(col)["failed"].agg(["mean", "size"]).reset_index()
                out += [{"method": log_path.stem.replace("_single", ""), "condition": col, "value": v,
                         "n_photos": int(n), "failure_rate": round(float(f), 3)} for v, f, n in g.itertuples(index=False)]
    return out


def save_examples(df):
    out = config.RESULTS_DIR / "failures"
    out.mkdir(parents=True, exist_ok=True)
    for c in CAUSES:
        sub = df[(df["cause"] == c) & (df["image_path"] != "")].drop_duplicates("query_id").head(EXAMPLES_PER_CAUSE)
        for i, r in enumerate(sub.itertuples()):
            img = cv2.imread(str(config.DATA_DIR / r.image_path))
            if img is not None:
                small = cv2.resize(img, None, fx=0.4, fy=0.4)
                cv2.putText(small, f"{r.method} true {r.true_id} pred {r.pred_id}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                cv2.imwrite(str(out / f"{c.replace(' ', '_')}_{i}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    m = load_manifest()
    art, name = dict(zip(m["id"], m["art_group"])), dict(zip(m["id"], m["name"]))
    df = pd.DataFrame(single_card(m, art, name) + scenes(art, name))
    if df.empty:
        raise SystemExit("no failures found - run cardrec.experiments first")
    df.to_csv(config.RESULTS_DIR / "failures.csv", index=False)
    summary = (df.groupby(["set", "method", "cause"]).size().unstack(fill_value=0).reset_index())
    summary.to_csv(config.RESULTS_DIR / "summary_failures.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    save_examples(df)
    real = real_conditions()
    if real:
        pd.DataFrame(real).to_csv(config.RESULTS_DIR / "summary_real_conditions.csv", index=False)
        print(pd.DataFrame(real).to_string(index=False))


if __name__ == "__main__":
    main()
