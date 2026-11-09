"""Step 1b: fill the manifest's `art_group` column (cards that share identical artwork).

    python -m cardrec.art_groups

Why: the strict metric counts a prediction correct only for the exact card id. Reprints with the
same artwork (e.g. Ascended Heroes reprints some Journey Together cards) are impossible to tell apart
from the artwork alone, so we also report a *lenient* metric: correct if the predicted card has the
same art_group as the true card.

How: perceptual hash (pHash) = 32x32 grayscale -> DCT -> keep the 8x8 lowest frequencies ->
1 bit per coefficient (above/below the median) = 64-bit fingerprint. Similar images have a small
Hamming distance (number of different bits). Two cards are "same artwork" when BOTH the whole-card
hash and the artwork-box hash differ in at most MAX_DIST bits. Groups are formed by union-find
(if A~B and B~C then A, B, C share a group); the group name is the alphabetically first card id.

On the 785-card database (2026-10-21): 24 pairs at distance 0-4, next closest unrelated pair >= 12,
so MAX_DIST = 6 separates them with a margin. Check the contact sheet by eye anyway.

Outputs (in config.DATA_DIR): manifest.csv (art_group filled), art_group_pairs.csv, art_group_pairs.jpg
"""
import cv2
import numpy as np
import pandas as pd

from . import config
from .manifest import load_manifest
from .methods.base import load_canonical

MAX_DIST = 6
ART_BOX = config.ART_BOX            # y0, y1, x0, x1 on the 500x700 canonical card (artwork window)


def phash(gray):
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()
    return low > np.median(low[1:])                 # skip [0] (overall brightness) for the median


def hashes(paths):
    full, art = [], []
    y0, y1, x0, x1 = ART_BOX
    for p in paths:
        g = cv2.cvtColor(load_canonical(p), cv2.COLOR_BGR2GRAY)
        full.append(phash(g))
        art.append(phash(g[y0:y1, x0:x1]))
    return np.array(full), np.array(art)


def hamming_matrix(h):
    return (h[:, None, :] != h[None, :, :]).sum(-1)


def group(ids, pairs):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    members = {}
    for i in ids:
        members.setdefault(find(i), []).append(i)
    return {i: min(g) for g in members.values() for i in g}


def contact_sheet(manifest, pairs, path, thumb_h=210):
    rows = []
    for a, b in pairs:
        ims = [cv2.resize(load_canonical(manifest.loc[c, "abs_path"]), (150, thumb_h)) for c in (a, b)]
        row = np.hstack([ims[0], np.full((thumb_h, 6, 3), 255, np.uint8), ims[1]])
        cv2.putText(row, a, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(row, b, (159, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        rows.append(row)
    cols = 6                                           # 6 pairs per line
    rows += [np.zeros_like(rows[0])] * (-len(rows) % cols)
    sheet = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    m = load_manifest()
    full, art = hashes(m["abs_path"].tolist())
    d_full, d_art = hamming_matrix(full), hamming_matrix(art)
    ids = m["id"].tolist()
    i, j = np.triu_indices(len(m), 1)
    keep = (d_full[i, j] <= MAX_DIST) & (d_art[i, j] <= MAX_DIST)
    pairs = [(ids[a], ids[b]) for a, b in zip(i[keep], j[keep])]
    groups = group(ids, pairs)

    pd.DataFrame([{"card_a": a, "card_b": b, "name_a": m.name[ids.index(a)], "name_b": m.name[ids.index(b)],
                   "dist_full": int(d_full[ids.index(a), ids.index(b)]),
                   "dist_art": int(d_art[ids.index(a), ids.index(b)])} for a, b in pairs]
                 ).to_csv(config.DATA_DIR / "art_group_pairs.csv", index=False)
    if pairs:
        contact_sheet(m.set_index("id"), pairs, config.DATA_DIR / "art_group_pairs.jpg")

    out = pd.read_csv(config.MANIFEST, dtype=str).fillna("")   # rewrite the file without abs_path
    out["art_group"] = out["id"].map(groups)
    out.to_csv(config.MANIFEST, index=False, encoding="utf-8")
    n_groups = out["art_group"].nunique()
    shared = (out["art_group"].value_counts() > 1).sum()
    print(f"{len(pairs)} same-artwork pairs; {len(out)} cards -> {n_groups} artwork groups "
          f"({shared} groups contain more than one card)")
    print("check by eye:", config.DATA_DIR / "art_group_pairs.jpg")


if __name__ == "__main__":
    main()
