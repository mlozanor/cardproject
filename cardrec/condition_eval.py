"""Step 8b: validate the condition estimate on synthetically DAMAGED cards with known severity.

    python -m cardrec.condition_eval --n 80 --severity 1 --split dev     # while tuning
    python -m cardrec.condition_eval --n 80 --severity 1 --split test    # final numbers (other cards)

For each card (official scan at condition resolution) a random set of damages with known strength
is applied, then the card is "photographed" with the scene generator at phone-like resolution
(rounded corners and cuts show the background) and analysed exactly like a real photo: rectify ->
refine_quad -> warp -> condition.assess(photo, official scan). Damages (strength s in {0, 0.25, 0.5, 0.75, 1}, 0 = none, chosen per corner/edge):
  corner whitening : white specks on the border within s * 80 px of the corner
  corner damage    : a missing circle of radius s * 28 px at the corner (chipped / bent corner)
  edge whitening   : white segments on the outer 0-6 px over a fraction s of the edge length
  edge fray        : ragged outline, random depth up to s * 7 px
  surface          : 0-3 scratches (thin bright lines) and 0-2 stains (dark blobs) at known places
  centering        : the print shifted by up to 12 px (about 1 mm) horizontally and vertically
Ground truth for centering: border widths of the CLEAN scan (where the frame edge is sharp) + the
known shift. Results: results/condition/validation.csv (one row per card) and summary printed:
Spearman rank correlation between true strength and estimated sub-score, centering error in
percentage points, surface defect recall/precision (match = within 0.04 of the card size), and the
score of undamaged cards (false alarms).
"""
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import condition as C
from . import config
from .manifest import load_manifest
from .rectify import find_card_quads, refine_quad, warp_to_canonical
from .synth import render_scene

LEVELS = [0, 0.25, 0.5, 0.75, 1.0]
CORNER_PTS = {"top_left": (0, 0), "top_right": (1, 0), "bottom_right": (1, 1), "bottom_left": (0, 1)}


def spearman(a, b):
    """Rank correlation; pairs with a missing value (e.g. surface not assessed) are left out."""
    ok = pd.Series(a).notna().values & pd.Series(b).notna().values
    a, b = pd.Series(a)[ok].rank(), pd.Series(b)[ok].rank()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def damage(scan, rng):
    """Returns (bgra image, truth dict)."""
    w, h = C.COND_W, C.COND_H
    im = cv2.resize(scan, (w, h), interpolation=cv2.INTER_AREA)
    clean_b = C.border_widths(im)
    pick = lambda: float(rng.choice(LEVELS)) if rng.random() < 0.5 else 0.0
    truth = {}
    # centering: shift the print, the exposed side repeats the border colour
    dx, dy = (int(rng.integers(-12, 13)), int(rng.integers(-12, 13))) if rng.random() < 0.6 else (0, 0)
    im = cv2.warpAffine(im, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    truth["left_pct"] = 100 * (clean_b["left"] + dx) / (clean_b["left"] + clean_b["right"])
    truth["top_pct"] = 100 * (clean_b["top"] + dy) / (clean_b["top"] + clean_b["bottom"])
    alpha = C.card_template()
    yy, xx = np.mgrid[0:h, 0:w]
    dist_out = C.inside_distance(alpha)                                   # px to the outline
    white = lambda m: im.__setitem__(m, rng.integers(225, 256, size=(int(m.sum()), 1)).repeat(3, 1))
    for name, (cx, cy) in CORNER_PTS.items():
        px, py = cx * (w - 1), cy * (h - 1)
        s = pick()
        truth[f"corner_{name}_white"] = s
        if s:
            near = (np.hypot(xx - px, yy - py) < s * C.CORNER_ZONE) & (dist_out > 0) & (dist_out < 10)
            white(near & (rng.random((h, w)) < 0.7))
        s = pick()
        truth[f"corner_{name}_cut"] = s
        if s:
            cv2.circle(alpha, (int(px), int(py)), int(C.CORNER_R + s * 28), 0, -1)
    for name in ("top", "right", "bottom", "left"):
        horiz = name in ("top", "bottom")
        n = w if horiz else h
        s = pick()
        truth[f"edge_{name}_white"] = s
        if s:
            on = np.zeros(n, bool)
            for start in rng.integers(C.CORNER_ZONE, n - C.CORNER_ZONE, size=int(1 + 6 * s)):
                on[start:start + int(s * (n - 2 * C.CORNER_ZONE) / (1 + 6 * s))] = True
            line = on[None, :] if horiz else on[:, None]
            band = (dist_out > 0) & (dist_out < 6) & np.broadcast_to(line, (h, w))
            band &= (yy < h / 2) if name == "top" else (yy > h / 2) if name == "bottom" else \
                    (xx < w / 2) if name == "left" else (xx > w / 2)
            white(band)
        s = pick()
        truth[f"edge_{name}_fray"] = s
        if s:
            depth = np.clip(np.cumsum(rng.normal(0, 1.5, n)), 0, None)
            depth = (depth / max(depth.max(), 1e-6) * s * 7).round().astype(int)
            for i in range(C.CORNER_ZONE, n - C.CORNER_ZONE):
                if depth[i]:
                    if name == "top":
                        alpha[:depth[i], i] = 0
                    elif name == "bottom":
                        alpha[h - depth[i]:, i] = 0
                    elif name == "left":
                        alpha[i, :depth[i]] = 0
                    else:
                        alpha[i, w - depth[i]:] = 0
    defects = []
    m = C.SURFACE_MARGIN + 20
    for _ in range(int(rng.integers(0, 4)) if rng.random() < 0.6 else 0):
        x0, y0 = int(rng.integers(m, w - m)), int(rng.integers(m, h - m))
        ang, L = rng.uniform(0, np.pi), int(rng.integers(40, 150))
        x1 = int(np.clip(x0 + L * np.cos(ang), m, w - m))
        y1 = int(np.clip(y0 + L * np.sin(ang), m, h - m))
        cv2.line(im, (x0, y0), (x1, y1), (235, 235, 235), int(rng.integers(1, 3)), cv2.LINE_AA)
        defects.append({"type": "scratch", "x": (x0 + x1) / 2 / w, "y": (y0 + y1) / 2 / h})
    for _ in range(int(rng.integers(0, 3)) if rng.random() < 0.5 else 0):
        x0, y0, r = int(rng.integers(m, w - m)), int(rng.integers(m, h - m)), int(rng.integers(6, 16))
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (x0, y0), r, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r / 3)[..., None]
        im = (im * (1 - 0.6 * blob)).astype(np.uint8)
        defects.append({"type": "stain / dent", "x": x0 / w, "y": y0 / h})
    truth["defects"] = defects
    return np.dstack([im, alpha]), truth


SEEDS = {"dev": config.SEED + 500, "test": config.SEED + 900}   # tune on dev, report test
CANVAS = (1920, 2560)        # card ~1050 px tall like a 12 MP phone photo (the default 960x1280 halves it)


def run(n, severity, split="dev"):
    seed = SEEDS[split]
    rng = np.random.default_rng(seed)
    man = load_manifest().sample(n, random_state=seed)
    out_dir = config.RESULTS_DIR / "condition"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, examples = [], []
    for r in tqdm(list(man.itertuples()), desc="condition"):
        scan = cv2.imread(r.abs_path)
        card, truth = damage(scan, rng)
        scene, placed = render_scene([card], rng, severity, canvas=CANVAS)
        quads = find_card_quads(scene, max_cards=1)
        if not quads:
            rows.append({"card": r.id, "rectified": False})
            continue
        photo = warp_to_canonical(scene, refine_quad(scene, quads[0]), C.COND_W, C.COND_H)
        res = C.assess(photo, scan)
        row = {"card": r.id, "rectified": True, "aligned": res["reference_aligned"], "overall": res["overall"],
               **{f"sub_{k}": v for k, v in res["subtotals"].items()},
               "true_left_pct": truth["left_pct"], "est_left_pct": res["centering"]["left_pct"],
               "true_top_pct": truth["top_pct"], "est_top_pct": res["centering"]["top_pct"]}
        for name in CORNER_PTS:
            z = res["corners"][name]
            row.update({f"corner_{name}_white": truth[f"corner_{name}_white"], f"corner_{name}_cut": truth[f"corner_{name}_cut"],
                        f"est_corner_{name}_whitening": z["whitening"], f"est_corner_{name}_fill": z["fill"],
                        f"est_corner_{name}_score": z["score"]})
        for name in ("top", "right", "bottom", "left"):
            z = res["edges"][name]
            row.update({f"edge_{name}_white": truth[f"edge_{name}_white"], f"edge_{name}_fray": truth[f"edge_{name}_fray"],
                        f"est_edge_{name}_whitening": z["whitening"], f"est_edge_{name}_fray": z["fray"],
                        f"est_edge_{name}_score": z["score"]})
        est = res["surface"]["defects"]
        hit = [any(np.hypot(d["x"] - e["x"], d["y"] - e["y"]) < 0.04 for e in est) for d in truth["defects"]]
        typed = [any(np.hypot(d["x"] - e["x"], d["y"] - e["y"]) < 0.04 and e["type"] == d["type"] for e in est)
                 for d in truth["defects"]]
        real = [any(np.hypot(d["x"] - e["x"], d["y"] - e["y"]) < 0.04 for d in truth["defects"]) for e in est]
        row.update(n_true_defects=len(truth["defects"]), n_found=sum(hit), n_typed=sum(typed),
                   n_est_defects=len(est), n_est_real=sum(real),
                   undamaged=all(v == 0 for k, v in truth.items() if k.startswith(("corner_", "edge_")))
                   and not truth["defects"])
        rows.append(row)
        if len(examples) < 4 and truth["defects"]:
            examples.append(np.hstack([photo, C.draw_map(photo, res)]))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"validation_{split}_s{severity}.csv", index=False)
    for i, ex in enumerate(examples):
        cv2.imwrite(str(out_dir / f"example_{split}_{i}.jpg"), cv2.resize(ex, None, fx=0.5, fy=0.5))
    summarize(df)


def summarize(df):
    d = df[df["rectified"] == True]                                        # noqa: E712
    print(f"cards: {len(df)}, rectified: {len(d)}, reference aligned: {int(d['aligned'].sum())}")
    print(f"centering abs error (percentage points): left/right median {np.median(abs(d.true_left_pct - d.est_left_pct)):.2f}"
          f" p90 {np.percentile(abs(d.true_left_pct - d.est_left_pct), 90):.2f} | top/bottom median "
          f"{np.median(abs(d.true_top_pct - d.est_top_pct)):.2f} p90 {np.percentile(abs(d.true_top_pct - d.est_top_pct), 90):.2f}")
    long = []
    for grp, names, pairs in [("corner", CORNER_PTS, [("white", "whitening"), ("cut", "fill")]),
                              ("edge", ["top", "right", "bottom", "left"], [("white", "whitening"), ("fray", "fray")])]:
        for n in names:
            for t, e in pairs:
                long += [{"what": f"{grp} {t} -> est {e}", "true": a, "est": b, "score": c}
                         for a, b, c in zip(d[f"{grp}_{n}_{t}"], d[f"est_{grp}_{n}_{e}"], d[f"est_{grp}_{n}_score"])]
    L = pd.DataFrame(long)
    print("Spearman rank correlation, true damage strength vs estimate (fill: expect negative):")
    for what, g in L.groupby("what"):
        print(f"  {what:32s} rho(sub-score) {spearman(g['true'], g['est']):+.2f}   rho(zone score) "
              f"{spearman(g['true'], g['score']):+.2f}   n={len(g)}")
    nt, nf = d.n_true_defects.sum(), d.n_found.sum()
    ne, nr = d.n_est_defects.sum(), d.n_est_real.sum()
    print(f"surface defects: recall {nf}/{nt} = {nf / max(nt, 1):.2f}, correct type {d.n_typed.sum()}/{nt}, "
          f"precision {nr}/{ne} = {nr / max(ne, 1):.2f}")
    u = d[d.undamaged == True]                                             # noqa: E712
    if len(u):
        print(f"undamaged cards: {len(u)}, overall score median {u.overall.median():.0f}, min {u.overall.min():.0f}")
    dmg = {"corners": d.filter(regex="^corner_").sum(axis=1), "edges": d.filter(regex="^edge_").sum(axis=1),
           "surface": d.n_true_defects, "centering": np.maximum(abs(d.true_left_pct - 50), abs(d.true_top_pct - 50))}
    total = dmg["corners"] + dmg["edges"] + d.n_true_defects + dmg["centering"] / 10
    print("part score vs that part's true damage (expect negative): " + ", ".join(
        f"{k} {spearman(v, d[f'sub_{k}']):+.2f}" for k, v in dmg.items() if f"sub_{k}" in d))
    print(f"overall score vs total damage (corners + edges + defects + centering/10): rho {spearman(total, d.overall):+.2f}"
          " (expect negative)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--severity", type=int, default=1, help="photo conditions of the scene generator (1-5)")
    ap.add_argument("--split", choices=["dev", "test"], default="dev", help="dev = for tuning, test = final numbers")
    a = ap.parse_args()
    run(a.n, a.severity, a.split)
