"""Step 5: synthetic query / scene generator with exact ground truth.

render_scene() places 1..N reference cards on a random background with random pose,
controlled overlap, and global distortions (lighting, glare, blur, noise) whose strength
is set by a severity level 1..5. It returns the image plus, for every placed card, its
index, its corner quad (ground truth box) and its visible fraction (for overlap analysis).
"""
import cv2
import numpy as np

# severity -> distortion strengths. Tune these by eye on a few generated images.
SEVERITY = {
    1: dict(rot=5,   persp=0.02, blur=0.0, noise=2,  light=0.10, glare=0.0),
    2: dict(rot=15,  persp=0.04, blur=1.0, noise=5,  light=0.20, glare=0.2),
    3: dict(rot=30,  persp=0.07, blur=1.8, noise=8,  light=0.30, glare=0.4),
    4: dict(rot=60,  persp=0.10, blur=2.8, noise=12, light=0.40, glare=0.6),
    5: dict(rot=180, persp=0.14, blur=4.0, noise=18, light=0.50, glare=0.8),
}


def make_background(h, w, rng):
    base = rng.integers(40, 200, size=3)
    small = rng.integers(0, 255, size=(6, 8, 3)).astype(np.uint8)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    bg = 0.5 * bg + 0.5 * base
    bg += rng.normal(0, 6, size=bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _place_card(card, canvas_hw, center, height_frac, angle_deg, persp, rng):
    H, W = canvas_hw
    ch, cw = card.shape[:2]
    s = height_frac * H / ch
    corners = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2],
                        [cw / 2, ch / 2], [-cw / 2, ch / 2]]) * s
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = corners @ R.T
    corners = corners + rng.uniform(-persp, persp, size=(4, 2)) * height_frac * H
    corners = (corners + np.array(center)).astype(np.float32)
    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, corners)
    # optional 4th channel = alpha (used by the condition tests: rounded / cut corners show the background)
    alpha = card[..., 3] if card.ndim == 3 and card.shape[2] == 4 else np.full((ch, cw), 255, np.uint8)
    warped = cv2.warpPerspective(np.ascontiguousarray(card[..., :3]), M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(alpha, M, (W, H)) > 127
    return warped, mask, corners


def _inside(quad, H, W, margin=4):
    return bool(np.all(quad[:, 0] >= margin) and np.all(quad[:, 0] <= W - margin)
                and np.all(quad[:, 1] >= margin) and np.all(quad[:, 1] <= H - margin))


def add_lighting(img, strength, rng):
    if strength <= 0:
        return img
    H, W = img.shape[:2]
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = (xx * np.cos(theta) + yy * np.sin(theta))
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6) - 0.5
    factor = 1.0 + strength * 1.2 * grad
    gamma = 1.0 + rng.uniform(-strength, strength)
    out = np.clip(img.astype(np.float32) * factor[..., None], 0, 255) / 255.0
    return (np.power(out, gamma) * 255).astype(np.uint8)


def add_glare(img, strength, rng, center=None):
    if strength <= 0:
        return img
    H, W = img.shape[:2]
    if center is None:
        center = (rng.uniform(0.2 * W, 0.8 * W), rng.uniform(0.2 * H, 0.8 * H))
    sigma = rng.uniform(0.06, 0.16) * H
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    blob = np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * sigma ** 2)))
    out = img.astype(np.float32) + strength * 255 * blob[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_blur(img, sigma, rng):
    if sigma <= 0:
        return img
    sigma *= rng.uniform(0.7, 1.3)
    if rng.random() < 0.5:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    k = int(2 * round(2 * sigma) + 1)               # motion blur
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


def add_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def render_scene(cards, rng, severity=2, canvas=(960, 1280), max_overlap=0.0):
    """cards: list of BGR (or BGRA, alpha = card shape) images. Returns (scene_bgr, placed) where placed is a list of dicts:
    {index, quad (4,2) ground-truth corners, visible_frac}. A card that cannot be placed is skipped."""
    H, W = canvas
    p = SEVERITY[severity]
    img = make_background(H, W, rng)
    n = len(cards)
    height_frac = {1: 0.55, 2: 0.45, 3: 0.40, 4: 0.36}.get(n, 0.32)
    union = np.zeros((H, W), bool)
    masks, placed = [], []
    for idx, card in enumerate(cards):
        ok = False
        hf = height_frac
        for _shrink in range(6):
            for _ in range(60):
                center = (rng.uniform(0.25 * W, 0.75 * W), rng.uniform(0.3 * H, 0.7 * H))
                angle = rng.uniform(-p["rot"], p["rot"])
                warped, mask, quad = _place_card(card, (H, W), center, hf, angle, p["persp"], rng)
                if not _inside(quad, H, W):
                    continue
                if (mask & union).sum() / max(mask.sum(), 1) > max_overlap:
                    continue
                img[mask] = warped[mask]
                union |= mask
                masks.append(mask)
                placed.append({"index": idx, "quad": quad})
                ok = True
                break
            if ok:
                break
            hf *= 0.9
    for i, item in enumerate(placed):                # visible fraction after later cards on top
        later = np.zeros((H, W), bool)
        for m in masks[i + 1:]:
            later |= m
        item["visible_frac"] = float((masks[i] & ~later).sum() / max(masks[i].sum(), 1))

    glare_center = None
    if placed:
        glare_center = placed[int(rng.integers(len(placed)))]["quad"].mean(axis=0)
    img = add_lighting(img, p["light"], rng)
    img = add_glare(img, p["glare"], rng, glare_center)
    img = add_blur(img, p["blur"], rng)
    img = add_noise(img, p["noise"], rng)
    return img, placed


def make_single_query(card, rng, severity=2):
    scene, placed = render_scene([card], rng, severity)
    return scene, (placed[0]["quad"] if placed else None)
