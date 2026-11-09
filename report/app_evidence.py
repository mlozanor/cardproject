"""Evidence from the CardDex application for the report: saved collection + re-identification of its photos.

    (venv active, from the project root)
    python report/app_evidence.py            # then: python report/make_figures.py

What it does
  1. Reads the app's collection (<data>/collection.db, opened READ-ONLY; the app may keep running).
  2. Writes results/app/collection_snapshot.csv: one row per saved card (what the app stored: card id,
     price, method and score at save time, printed language, condition estimate per side).
  3. Re-identifies every saved photo (<data>/uploads/<photo>) with each method in METHODS, through the
     same path the app uses (rectify -> method.query), and writes results/app/collection_reid.csv.
     Ground truth = the card the user SAVED (the user can correct the app's suggestion before saving),
     so this is a small user-labelled real-photo check, not a benchmark.
  4. Crops the app screenshots in "app images/" to the app area (no browser tabs / bookmarks) and
     writes them to report/figures/app_*.png.
"""
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
sys.path.insert(0, str(ROOT))

from cardrec import config  # noqa: E402
from cardrec.evaluate import rank_row  # noqa: E402
from cardrec.manifest import load_app_manifest  # noqa: E402
from cardrec.methods import get_method  # noqa: E402
from cardrec.methods.cache import build_or_load  # noqa: E402
from cardrec.rectify import rectify  # noqa: E402

METHODS = ["pixel", "sift", "orb", "resnet", "clip", "hybrid-clip+rot", "hybrid-clip@art+rot"]
MIN_INLIERS, MIN_MARGIN = 15, 1.4          # the app's confidence rule for inlier scores (app/engine.py)
OUT = config.RESULTS_DIR / "app"
SCREENSHOTS = ROOT / "app images"
# (source file, output name, crop y0:y1, x0:x1 in pixels of a 1920x1020 screenshot: the two app panels)
CROPS = [("princiapal intercafe.png", "app_collection.png", (120, 1000, 200, 1720)),
         ("interface.png", "app_analysis.png", (120, 1000, 200, 1720)),
         ("top matches.png", "app_matches_tags.png", (120, 1000, 940, 1720)),
         ("surface details.png", "app_surface.png", (120, 1000, 940, 1720))]


def read_collection():
    uri = (config.DATA_DIR / "collection.db").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("SELECT * FROM collection ORDER BY id")]


def snapshot(items):
    rows = []
    for it in items:
        cond = json.loads(it["condition"]) if it.get("condition") else {}
        row = {k: it.get(k) for k in ["id", "card_id", "name", "set_name", "number", "quantity", "price",
                                      "price_source", "price_updated", "method", "score", "language", "photo",
                                      "added_at"]}
        row["cond_overall"] = cond.get("overall")
        row["cond_problems"] = " | ".join(cond.get("problems") or [])
        for side in ["front", "back"]:
            s = (cond.get("sides") or {}).get(side)
            if not s:
                continue
            for k, v in (s.get("subtotals") or {}).items():
                row[f"cond_{side}_{k}"] = v
            row[f"cond_{side}_overall"] = s.get("overall")
            row[f"cond_{side}_aligned"] = s.get("reference_aligned")
            row[f"cond_{side}_inliers"] = s.get("reference_inliers")
            row[f"cond_{side}_n_defects"] = len((s.get("surface") or {}).get("defects") or [])
            row[f"cond_{side}_centering_ratio"] = f"{(s.get('centering') or {}).get('left_right', '')} L/R, " \
                                                  f"{(s.get('centering') or {}).get('top_bottom', '')} T/B"
        rows.append(row)
    return pd.DataFrame(rows)


def reidentify(items, manifest):
    groups = dict(zip(manifest["id"], manifest["art_group"]))
    photos = [(it["card_id"], it["photo"]) for it in items if it.get("photo")]
    rows = []
    for name in METHODS:
        m = build_or_load(get_method(name, k=20), manifest)
        for true_id, photo in photos:
            img = cv2.imread(str(config.DATA_DIR / "uploads" / photo))
            t0 = time.perf_counter()
            card, quad, found = rectify(img)
            ranked = m.query(card)
            ms = 1000 * (time.perf_counter() - t0)
            r = rank_row(ranked, true_id, groups)
            inliers = m.name.startswith(("sift", "orb", "hybrid"))
            s1, s2 = r["pred_score"], r["second_score"] or 0
            rows.append({"method": m.name, "true_id": true_id, "photo": photo, "rectify_found": found,
                         "ms_total": ms, "confident": (s1 >= MIN_INLIERS and s1 >= MIN_MARGIN * max(s2, 1))
                         if inliers else None, **r})
            print(f"{m.name:28s} {true_id:12s} pred {r['pred_id']:12s} rank {r['rank_of_true']}  {ms:,.0f} ms")
    return pd.DataFrame(rows)


def crop_screenshots():
    for src, dst, (y0, y1, x0, x1) in CROPS:
        img = cv2.imread(str(SCREENSHOTS / src))
        if img is None:
            print("missing screenshot:", src)
            continue
        cv2.imwrite(str(REPORT / "figures" / dst), img[y0:y1, x0:x1])
        print("written", REPORT / "figures" / dst)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(exist_ok=True)
    items = read_collection()
    snap = snapshot(items)
    snap["snapshot_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snap.to_csv(OUT / "collection_snapshot.csv", index=False)
    print(f"{len(snap)} saved cards -> {OUT / 'collection_snapshot.csv'}")
    reidentify(items, load_app_manifest()).to_csv(OUT / "collection_reid.csv", index=False)
    print("re-identification log ->", OUT / "collection_reid.csv")
    crop_screenshots()
