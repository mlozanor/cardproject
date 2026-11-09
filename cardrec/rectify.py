"""Step 4: find card(s) in a photo and warp each to a canonical top-down view.

Classical pipeline (version 2, 2026-10-21):
  1. downscale to <= 800 px, blur, take two channels: grayscale and HSV saturation
     (a silver border can have the same brightness as the table but a different colourfulness).
  2. Canny edges with 3 threshold settings per channel (fixed, median-based, low for blurry photos),
     each dilated twice: 1x (keeps the thin border between touching cards) and 2x (closes gaps in
     blurry edges). ALL 12 edge maps are used (version 1 stopped at the first that found anything).
  3. for every contour: convex hull ("rubber band" around it) -> simplify to 4 corners with growing
     tolerance. This survives rounded corners and edges broken by glare (v1 needed the raw contour
     to be a 4-gon, and then often picked the inner text box instead of the card).
  4. keep a 4-gon only if it is big enough, does not touch the image border, has a card-like aspect
     ratio, the contour fills it (fill = contour area / quad area >= MIN_FILL), and its four sides lie
     on real edges (edge support = fraction of points sampled along the sides that are edge pixels
     >= MIN_SUPPORT). The last test rejects big shapes "stretched" around background noise.
  5. sort by score = area x support^2 (a shape that also covers some background has a lower support,
     so the true card outline wins); drop candidates that overlap an already kept one (IoU > 0.5) or
     whose centre lies inside it (inner frames such as the artwork box).
Known limits: cards that overlap each other merge into one non-rectangular contour; very strong
blur/glare removes the border edge. Rectification cannot tell a card from its 180-degree rotation.
Tuned ONLY on DEV sets (seeds 1000 / 2000), never on the test queries. Outline correct (IoU >= 0.8)
on 200 single-card dev queries: version 1 49.5%, version 2 83.5% (s1 100%, s2 100%, s3 93%, s4 95%,
s5 30%). Multi-card dev scenes (20 per level), recall: no overlap 0.75, 15% overlap 0.21, 30% overlap
0.34 (version 1: 0.68 / 0.07 / 0.10). Larger WORK_SIZE helps overlapping cards but hurts blurry ones.
"""
import cv2
import numpy as np

from . import config

MIN_FILL = 0.85
MIN_SUPPORT = 0.6
WORK_SIZE = 800                     # longest image side used for detection (px)
DILATE_ITERS = (1, 2)               # edge thickening (3x3 dilations); 2 closes gaps in blurry edges,
                                    # 1 keeps the thin border strip between touching cards open
BORDER_MARGIN = 3                   # px (at the working size)
CARD_ASPECT = (1.15, 1.95)          # long/short side; a real card is 88/63 = 1.40, perspective changes it


def order_corners(pts):
    """Return corners ordered top-left, top-right, bottom-right, bottom-left.
    Sort by angle around the centre (works at any rotation), then start at the corner with the smallest x+y."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]   # clockwise in image coordinates
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def _aspect_ok(quad):
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    long_side, short_side = max(w, h), max(min(w, h), 1e-6)
    return CARD_ASPECT[0] <= long_side / short_side <= CARD_ASPECT[1]


def _hull_to_quad(contour):
    """Convex hull -> 4 corners (growing approxPolyDP tolerance), or None."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return hull, order_corners(approx)
        if len(approx) < 4:
            break
    return hull, None


def _edge_support(quad, edges, n=60):
    """Fraction of n points per side (sampled along the quad's 4 sides) that fall on edge pixels."""
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    pts = np.vstack([quad[i] + t * (quad[(i + 1) % 4] - quad[i]) for i in range(4)]).round().astype(int)
    pts[:, 0] = pts[:, 0].clip(0, edges.shape[1] - 1)
    pts[:, 1] = pts[:, 1].clip(0, edges.shape[0] - 1)
    return float((edges[pts[:, 1], pts[:, 0]] > 0).mean())


def _edge_maps(small):
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    for channel in (cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), hsv[..., 1]):
        ch = cv2.GaussianBlur(channel, (5, 5), 0)
        med = float(np.median(ch))
        for lo, hi in [(50, 150), (int(max(0, 0.5 * med)), int(min(255, 1.0 * med))), (10, 40)]:
            edges = cv2.Canny(ch, lo, hi)
            for it in DILATE_ITERS:
                yield cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=it)


def find_card_quads(img, min_area_frac=0.03, max_cards=10):
    """Return a list of (4,2) float32 quads in original-image pixels, best (highest score) first."""
    h, w = img.shape[:2]
    scale = min(1.0, WORK_SIZE / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    img_area = small.shape[0] * small.shape[1]

    candidates = []
    for edges in _edge_maps(small):
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(cv2.convexHull(c)) < min_area_frac * img_area:
                continue
            _, quad = _hull_to_quad(c)
            if quad is None or not _aspect_ok(quad):
                continue
            hh, ww = small.shape[:2]
            if (quad.min() < BORDER_MARGIN or quad[:, 0].max() > ww - 1 - BORDER_MARGIN
                    or quad[:, 1].max() > hh - 1 - BORDER_MARGIN):
                continue                                # touches the image border
            quad_area = cv2.contourArea(quad)
            if quad_area < min_area_frac * img_area or cv2.contourArea(c) / quad_area < MIN_FILL:
                continue
            support = _edge_support(quad, edges)
            if support >= MIN_SUPPORT:
                candidates.append((quad_area * support ** 2, quad))   # score: big AND well on edges
    candidates.sort(key=lambda t: -t[0])

    kept = []
    for _, quad in candidates:
        center = tuple(float(v) for v in quad.mean(axis=0))
        # drop duplicates of an accepted quad and inner frames (artwork box) nested inside it
        if any(cv2.pointPolygonTest(k, center, False) >= 0 or polygon_iou(k, quad, small.shape, 0.5) > 0.5
               for k in kept):
            continue
        kept.append(quad)
        if len(kept) >= max_cards:
            break
    return [q / scale for q in kept]


def refine_quad(img, quad):
    """Precise outer outline for the condition estimate (identification does not need it).
    Two passes of _refine_once: a wide search (the detector may have locked onto a frame up to ~8%
    inside the card), then a narrow one around that result."""
    return _refine_once(img, _refine_once(img, quad, (-0.01, 0.09)), (-0.02, 0.02))


def _refine_once(img, quad, search, n_lines=40):
    """One refinement pass.

    The detector may lock onto the inner frame, and its corners lie on the rounded corners. Here,
    for each side: sample n_lines short profiles across the side (15%..85% of its length, away from
    the rounded corners), going from `search[0]` to `search[1]` x card size outward; on each profile
    take the OUTERMOST position whose colour change is >= EDGE_REL x that profile's strongest change
    (the card/background boundary); fit a straight line through these points (Huber fit, robust to
    outliers); intersect neighbouring lines = sharp virtual corners. The change is measured in Lab
    colour (lightness AND colour): a silver border can be as bright as the table but not the same
    colour. Assumes a fairly plain background just outside the card (see the photo protocol)."""
    EDGE_REL = 0.35
    q = order_corners(quad).astype(np.float64)
    lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), (3, 3), 0).astype(np.float32)
    center = q.mean(axis=0)
    size = np.sqrt(abs(cv2.contourArea(q.astype(np.float32))))
    offs = np.arange(search[0] * size, search[1] * size, 0.5)
    lines = []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        nrm = np.array([d[1], -d[0]])
        if np.dot(nrm, (p0 + p1) / 2 - center) < 0:
            nrm = -nrm                                          # make the normal point outward
        pts = []
        for t in np.linspace(0.15, 0.85, n_lines):
            base = p0 + t * (p1 - p0)
            xy = (base[None, :] + offs[:, None] * nrm[None, :]).astype(np.float32)
            prof = cv2.remap(lab, xy[:, 0].reshape(1, -1), xy[:, 1].reshape(1, -1), cv2.INTER_LINEAR)[0]
            grad = np.linalg.norm(np.diff(prof, axis=0), axis=1)   # colour distance between neighbours
            if grad.max() < 8:                                  # no clear edge on this profile
                continue
            k = np.flatnonzero(grad >= EDGE_REL * grad.max())[-1]  # outermost strong change
            pts.append(base + (offs[k] + 0.25) * nrm)
        if len(pts) < n_lines // 3:
            return q.astype(np.float32)                        # not enough evidence: keep the original
        vx, vy, x0, y0 = cv2.fitLine(np.float32(pts), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    corners = []
    for i in range(4):                                         # corner i = side (i-1) x side i
        (a, da), (b, db) = lines[i - 1], lines[i]
        A = np.array([da, -db]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return q.astype(np.float32)
        s = np.linalg.solve(A, b - a)[0]
        corners.append(a + s * da)
    return order_corners(np.float32(corners))


def polygon_iou(a, b, shape, scale=0.25):
    """IoU = area(intersection) / area(union) of two quads, on rasterised masks (4x downscaled for speed)."""
    h, w = int(shape[0] * scale), int(shape[1] * scale)
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.round(np.asarray(a) * scale).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(np.asarray(b) * scale).astype(np.int32)], 1)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


def warp_to_canonical(img, quad, out_w=config.CANON_W, out_h=config.CANON_H):
    quad = order_corners(quad)
    top = np.linalg.norm(quad[1] - quad[0])
    left = np.linalg.norm(quad[3] - quad[0])
    if top > left:                      # landscape -> make the long side vertical
        quad = np.roll(quad, -1, axis=0)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_AREA)


def rectify(img):
    """Single-card convenience: returns (canonical_image, quad_or_None, found_flag).

    If no card outline is found, falls back to the full image resized to canonical size.
    """
    quads = find_card_quads(img, max_cards=1)
    if quads:
        return warp_to_canonical(img, quads[0]), quads[0], True
    return cv2.resize(img, (config.CANON_W, config.CANON_H), interpolation=cv2.INTER_AREA), None, False


def rectify_all(img, max_cards=10):
    """Multi-card: returns a list of (canonical_image, quad)."""
    return [(warp_to_canonical(img, q), q) for q in find_card_quads(img, max_cards=max_cards)]
