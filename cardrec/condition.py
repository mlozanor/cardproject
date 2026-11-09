"""Step 8: VISUAL CONDITION ESTIMATE of a photographed card (NOT an official grade).

Idea: once the card is identified we have its official scan = a picture of the perfect card.
Condition is then measured by comparing the photo with that reference, region by region.

    photo (original) --quad--> warp at COND_W x COND_H (~12 px per mm)
    reference scan  --SIFT+RANSAC homography--> aligned to the photo (sub-pixel), colours matched
    centering : border widths L, R, T, B from the photo alone (first strong brightness change inside
                each edge = start of the inner frame) -> left/right and top/bottom percentages
    corners(4), edges(4): three sub-scores each, in [0, 1]
        whitening  share of the zone's border pixels that look white/washed-out where the reference is not
        fill       share of the expected card shape (rounded-corner template) that looks like card
                   (low = bent, chipped or missing corner)
        fray       share of a thin ring just inside the outline that does NOT look like card (ragged edge)
    surface   : remaining differences inside the card (outside the 4 corner zones, which are scored
                separately) -> blobs -> type from shape and brightness
                (scratch = long and thin, stain/dent = darker, spot/print line = brighter) and position
    scores    : zone score = 1000 * (1 - min(1, W_WHITE*whitening + W_FILL*(1-fill) + W_FRAY*fray))
                overall = 0.20 centering + 0.30 corners + 0.25 edges + 0.25 surface,
                where corners/edges = 0.5 * mean + 0.5 * worst (one bad corner matters).
                The weights are hand-set and documented; they are NOT fitted to any grading company.
Back side: there is no official back scan. Pass a photo of one clean card back as `reference`
(e.g. <data>/condition/back_reference.jpg); without it only centering, corners and edges are
measured, with each border's own median colour as the reference, and surface is skipped.
"""
import cv2
import numpy as np

COND_W, COND_H = 750, 1050          # 63 x 88 mm card -> ~11.9 px per mm
CORNER_R = 37                       # rounded-corner radius in px (~3.1 mm), measured on 40 official scans
CORNER_ZONE = 80                    # corner zone: 80 x 80 px square
EDGE_DEPTH = 16                     # edge zone: strip this deep along each side (between corners)
RING = (2, 6)                       # fray ring: 2..6 px inside the outline (0..2 px = rectification error)
BORDER_SEARCH = (0.015, 0.09)       # inner-frame search, as a fraction of the card width/height
W_WHITE, W_FILL, W_FRAY = 2.0, 2.5, 1.5
SURFACE_MARGIN = 24                 # px kept away from the outline for the surface test
SURFACE_THRESH = 0.16               # mean abs colour difference (0..1) that counts as a defect
MIN_DEFECT_PX = 25
MAX_CORNER_SHIFT = 0.06             # alignment accepted only if it moves the card corners < 6% of the width
LABEL = "Visual condition ESTIMATE from a photo - not an official grade"
LIMITS = [
    "Exact dimensions, thickness and warping cannot be measured: one photo has no metric scale or depth.",
    "Centering is limited by the photo's perspective correction (about +/-1-2 percentage points).",
    "Whitening or fraying smaller than about 1 px (~0.1 mm) at this resolution is invisible.",
    "Glare, holo/foil reflections and sleeves can look like surface defects (and hide real ones); "
    "light scratches are often only visible under grazing light.",
    "The card back has no official scan: a clean back photo must be given as reference for the surface test.",
    "Scores use hand-set weights on our own 0-1000 scale; they are not PSA/BGS/CGC grades.",
]


def card_template(w=COND_W, h=COND_H, r=CORNER_R):
    """255 inside the ideal card (a rectangle with rounded corners), 0 outside."""
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in [(r, r), (w - 1 - r, r), (w - 1 - r, h - 1 - r), (r, h - 1 - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def inside_distance(mask):
    """Distance (px) from each pixel inside `mask` to the card outline (0 outside). The mask is padded
    with a 1-px empty frame first: the card touches the image border, and without the frame the
    image border would not count as 'outside' (the same problem makes cv2.erode useless here)."""
    padded = cv2.copyMakeBorder((mask > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]


def align_reference(photo, ref, min_inliers=15):
    """Warp the reference onto the photo with a SIFT + RANSAC homography.
    Returns (aligned reference, n_inliers, H or None)."""
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, d1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    kp2, d2 = sift.detectAndCompute(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        return ref, 0, None
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < min_inliers:
        return ref, len(good), None
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    n = int(mask.sum()) if mask is not None else 0
    if H is None or n < min_inliers:
        return ref, n, None
    # sanity check: both images show the whole card, so H must send the reference's corners close to
    # the photo's corners (a wrong H can still have many inliers on repeated patterns)
    h, w = photo.shape[:2]
    c = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    if np.abs(cv2.perspectiveTransform(c, H) - c).max() > MAX_CORNER_SHIFT * w:
        return ref, n, None
    return cv2.warpPerspective(ref, H, (photo.shape[1], photo.shape[0]), borderMode=cv2.BORDER_REPLICATE), n, H


def match_colors(ref, photo, mask):
    """Per colour channel, fit photo ~ a * ref + b on the masked pixels (least squares, refit after
    dropping the 10% worst pixels) and apply it to ref, so lighting differences are not 'damage'."""
    out = np.empty(ref.shape, np.float32)
    sel = mask > 0
    for c in range(3):
        x, y = ref[..., c][sel].astype(np.float32), photo[..., c][sel].astype(np.float32)
        a, b = np.polyfit(x, y, 1)
        keep = np.abs(a * x + b - y) <= np.percentile(np.abs(a * x + b - y), 90)
        a, b = np.polyfit(x[keep], y[keep], 1)
        out[..., c] = a * ref[..., c].astype(np.float32) + b
    return np.clip(out, 0, 255).astype(np.uint8)


def border_widths(card):
    """Distance (px) from each outer edge to the inner frame: the first strong brightness change.
    Measured on the middle 60% of each side, averaged over all lines there."""
    g = cv2.GaussianBlur(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    h, w = g.shape
    out = {}
    for side in ("left", "right", "top", "bottom"):
        if side in ("left", "right"):
            band = g[int(0.2 * h):int(0.8 * h), :]
            prof = np.abs(np.diff(band, axis=1)).mean(axis=0)            # one value per column
            if side == "right":
                prof = prof[::-1]
            size = w
        else:
            band = g[:, int(0.2 * w):int(0.8 * w)]
            prof = np.abs(np.diff(band, axis=0)).mean(axis=1)
            if side == "bottom":
                prof = prof[::-1]
            size = h
        lo, hi = int(BORDER_SEARCH[0] * size), int(BORDER_SEARCH[1] * size)
        seg = prof[lo:hi]
        peaks = np.flatnonzero(seg >= 0.5 * seg.max())                   # first strong change, not the biggest
        out[side] = float(lo + (peaks[0] if len(peaks) else int(np.argmax(seg))) + 1)
    return out


def borders_from_reference(ref, H, w=COND_W, h=COND_H):
    """Border widths in the PHOTO via the reference: measure the inner frame on the clean scan (sharp),
    map those frame lines into the photo with the alignment homography H, and measure their distance
    to the photo's card edges (= the image frame after rectification)."""
    b = border_widths(ref)
    ys, xs = np.linspace(0.3 * h, 0.7 * h, 5), np.linspace(0.3 * w, 0.7 * w, 5)
    pts = {"left": [(b["left"], y) for y in ys], "right": [(w - b["right"], y) for y in ys],
           "top": [(x, b["top"]) for x in xs], "bottom": [(x, h - b["bottom"]) for x in xs]}
    out = {}
    for side, p in pts.items():
        m = cv2.perspectiveTransform(np.float32(p).reshape(-1, 1, 2), H).reshape(-1, 2)
        out[side] = float({"left": m[:, 0].mean(), "right": w - m[:, 0].mean(),
                           "top": m[:, 1].mean(), "bottom": h - m[:, 1].mean()}[side])
    return out


def centering(card, ref=None, H=None):
    """Left/right and top/bottom centering. With an aligned reference the frame position comes from
    the clean scan (more reliable, e.g. on full-art cards); otherwise it is detected in the photo."""
    b, method = (borders_from_reference(ref, H), "reference") if H is not None else (border_widths(card), "photo")
    h, w = card.shape[:2]
    plausible = lambda bw: all(0 < bw[s] < 0.15 * (w if s in ("left", "right") else h) for s in bw)
    if not plausible(b) and method == "reference":
        b, method = border_widths(card), "photo"
    lr = 100 * b["left"] / (b["left"] + b["right"])
    tb = 100 * b["top"] / (b["top"] + b["bottom"])
    worst = max(abs(lr - 50), abs(tb - 50))
    # 50/50 -> 1000, 55/45 -> 900, 60/40 -> 750, 65/35 -> 550, 70/30 -> 350, 75/25 or worse -> 150
    score = float(np.interp(worst, [0, 5, 10, 15, 20, 25], [1000, 900, 750, 550, 350, 150]))
    return {"left_right": f"{lr:.0f}/{100 - lr:.0f}", "top_bottom": f"{tb:.0f}/{100 - tb:.0f}",
            "left_pct": round(lr, 1), "top_pct": round(tb, 1), "method": method, "plausible": plausible(b),
            "border_px": {k: round(v, 1) for k, v in b.items()}, "score": round(score)}


def _zones(w=COND_W, h=COND_H):
    """name -> (x0, y0, x1, y1) of the 4 corner squares and the 4 edge strips."""
    z, d = CORNER_ZONE, EDGE_DEPTH
    return {
        "corners": {"top_left": (0, 0, z, z), "top_right": (w - z, 0, w, z),
                    "bottom_right": (w - z, h - z, w, h), "bottom_left": (0, h - z, z, h)},
        "edges": {"top": (z, 0, w - z, d), "right": (w - d, z, w, h - z),
                  "bottom": (z, h - d, w - z, h), "left": (0, z, d, h - z)},
    }


def _side_medians(photo, template):
    """No reference: each border's own median colour (borders are one uniform colour)."""
    ref = np.empty_like(photo)
    ref[:] = np.median(photo[template > 0], axis=0)
    d = 3 * EDGE_DEPTH
    h, w = template.shape
    for (x0, y0, x1, y1) in [(0, 0, w, d), (0, h - d, w, h), (0, 0, d, h), (w - d, 0, w, h)]:
        sel = template[y0:y1, x0:x1] > 0
        ref[y0:y1, x0:x1] = np.median(photo[y0:y1, x0:x1][sel], axis=0)
    return ref


def zone_scores(photo, ref, template, border_only=False):
    """border_only=True (no aligned reference): only the border strip is compared, because only the
    border has one uniform colour that the per-side median can stand in for."""
    hsv_p = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
    hsv_r = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
    diff = np.abs(photo.astype(np.float32) - ref.astype(np.float32)).mean(axis=2) / 255
    white = (hsv_p[..., 2] - hsv_r[..., 2] > 0.18) & (hsv_p[..., 1] < 0.25) & (hsv_p[..., 2] > 0.7)
    cardlike = (diff < 0.15) | white
    dist = inside_distance(template)
    inside = dist > RING[0]                 # skip the outermost RING[0] px: rectification error band
    ring = (dist > RING[0]) & (dist <= RING[1])
    border = inside & (dist <= EDGE_DEPTH)
    if border_only:
        inside = border
    out = {}
    for group, zones in _zones(*photo.shape[1::-1]).items():
        out[group] = {}
        for name, (x0, y0, x1, y1) in zones.items():
            s = np.s_[y0:y1, x0:x1]
            f = lambda m: float(m.mean()) if m.size else 0.0
            whitening = f(white[s][border[s]])
            fill = f(cardlike[s][inside[s]])
            fray = 1 - f(cardlike[s][ring[s]])
            score = 1000 * (1 - min(1.0, W_WHITE * whitening + W_FILL * (1 - fill) + W_FRAY * fray))
            out[group][name] = {"whitening": round(whitening, 3), "fill": round(fill, 3), "fray": round(fray, 3),
                                "score": round(score)}
    return out


def surface_defects(photo, ref, template):
    inner = inside_distance(template) > SURFACE_MARGIN
    for (x0, y0, x1, y1) in _zones(*photo.shape[1::-1])["corners"].values():
        inner[y0:y1, x0:x1] = False          # corners are assessed by zone_scores: do not count them twice
    p = cv2.GaussianBlur(photo, (0, 0), 1.5).astype(np.float32)
    r = cv2.GaussianBlur(ref, (0, 0), 1.5).astype(np.float32)
    res = np.abs(p - r).mean(axis=2) / 255
    sign = (p.mean(axis=2) - r.mean(axis=2)) / 255                       # + brighter, - darker than ref
    m = ((res > SURFACE_THRESH) & inner).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    card_area = float(inner.sum())
    defects, warnings = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_DEFECT_PX:
            continue
        pts = np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(np.float32)
        (_, _), (a, b), _ = cv2.minAreaRect(pts)
        long_side, short_side = max(a, b), max(min(a, b), 1.0)
        bright = float(sign[lab == i].mean()) > 0
        frac = area / card_area
        if bright and frac > 0.01 and float(cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)[..., 1][lab == i].mean()) < 70:
            warnings.append(f"large bright area at ({cents[i][0] / w:.2f}, {cents[i][1] / h:.2f}) looks like glare - "
                            "not scored; retake the photo without reflections")
            continue
        if long_side / short_side >= 4 and short_side <= 8:
            kind = "scratch"
        elif frac > 0.015:
            kind = "large area (crease, stain or print difference)"
        else:
            kind = "spot / print line" if bright else "stain / dent"
        defects.append({"type": kind, "x": round(float(cents[i][0] / w), 3), "y": round(float(cents[i][1] / h), 3),
                        "w": round(float(stats[i, cv2.CC_STAT_WIDTH] / w), 3),
                        "h": round(float(stats[i, cv2.CC_STAT_HEIGHT] / h), 3),
                        "area_pct": round(100 * frac, 3), "strength": round(float(res[lab == i].mean()), 3)})
    penalty = sum(min(400.0, 40 + 20000 * d["area_pct"] / 100) for d in defects)
    return defects, warnings, round(max(0.0, 1000 - penalty))


def _combine(scores):
    v = list(scores)
    return 0.5 * float(np.mean(v)) + 0.5 * float(min(v))


def assess(photo, reference=None, side="front"):
    """photo: card warped to COND_W x COND_H (see warp in rectify.warp_to_canonical).
    reference: official scan (front) / clean back photo, any size, or None."""
    photo = cv2.resize(photo, (COND_W, COND_H), interpolation=cv2.INTER_AREA)
    template = card_template()
    n_inliers, H, clean = 0, None, None
    if reference is not None:
        clean = cv2.resize(reference, (COND_W, COND_H), interpolation=cv2.INTER_AREA)
        ref, n_inliers, H = align_reference(photo, clean)
    aligned = H is not None
    if aligned:
        ref = match_colors(ref, photo, template)
    else:
        ref = _side_medians(photo, template)
    cen = centering(photo, clean, H)
    zones = zone_scores(photo, ref, template, border_only=not aligned)
    if aligned:
        defects, warnings, surf = surface_defects(photo, ref, template)
    else:
        defects, warnings, surf = [], ["surface not assessed: no aligned reference image"], None
    corners = _combine(z["score"] for z in zones["corners"].values())
    edges = _combine(z["score"] for z in zones["edges"].values())
    parts = {"centering": (0.20, cen["score"]), "corners": (0.30, corners), "edges": (0.25, edges)}
    if surf is not None:
        parts["surface"] = (0.25, surf)
    overall = sum(wt * s for wt, s in parts.values()) / sum(wt for wt, _ in parts.values())
    return {"label": LABEL, "side": side, "overall": round(overall), "centering": cen,
            "corners": zones["corners"], "edges": zones["edges"],
            "surface": {"score": surf, "defects": defects, "warnings": warnings},
            "subtotals": {k: round(s) for k, (_, s) in parts.items()},
            "reference_aligned": aligned, "reference_inliers": n_inliers, "limits": LIMITS}


# Seen from the back, the card is mirrored left <-> right: the back photo's top-left corner is
# physically behind the front's top-right corner. Physical zone name -> zone name in the back photo.
MIRROR = {"top_left": "top_right", "top_right": "top_left", "bottom_left": "bottom_right",
          "bottom_right": "bottom_left", "left": "right", "right": "left", "top": "top", "bottom": "bottom"}


def combine(front=None, back=None):
    """One report for a card from its front and/or back assessment. Zones are listed per PHYSICAL
    corner/edge (named as seen from the front) with the front and the (mirrored) back values.
    overall = mean of the available sides' overall scores (front and back weigh the same)."""
    sides = {k: v for k, v in (("front", front), ("back", back)) if v}
    if not sides:
        return None
    zones = {"corners": {}, "edges": {}}
    for group in zones:
        for name in _zones()[group]:
            zones[group][name] = {"front": front[group][name] if front else None,
                                  "back": back[group][MIRROR[name]] if back else None}
    return {"label": LABEL, "overall": round(float(np.mean([s["overall"] for s in sides.values()]))),
            "sides": sides, "zones": zones,
            "overall_rule": "mean of the front and back overall scores" if len(sides) == 2
            else f"{next(iter(sides))} only - add a photo of the other side",
            "limits": LIMITS}


def draw_map(photo, result):
    """Card map for the interface/report: zone outlines coloured by score, numbered defect markers."""
    img = cv2.resize(photo, (COND_W, COND_H)).copy()
    col = lambda s: (60, 180, 60) if s >= 850 else (0, 190, 255) if s >= 600 else (40, 40, 230)
    for group in ("corners", "edges"):
        for name, (x0, y0, x1, y1) in _zones()[group].items():
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), col(result[group][name]["score"]), 3)
    for i, d in enumerate(result["surface"]["defects"], 1):
        c = (int(d["x"] * COND_W), int(d["y"] * COND_H))
        cv2.circle(img, c, 16, (255, 0, 255), 3)
        cv2.putText(img, str(i), (c[0] + 18, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return img
