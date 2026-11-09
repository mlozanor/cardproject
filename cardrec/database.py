"""Step 1: download official card images and build the manifest (one row per unique card).

Usage (from the project root):
    python -m cardrec.database                 # all sets listed in config.SETS
    python -m cardrec.database --limit 20      # first 20 cards per set (quick test)
    python -m cardrec.database --sets me5      # only some sets (TCGdex fill only for sets in config.SETS)

Sources:
  - Pokemon TCG API (pokemontcg.io): card list, metadata and images. The API sometimes answers
    502/500 for a moment, so every request is retried with growing waits.
  - TCGdex (tcgdex.net): we store each card's TCGdex id (the app gets prices from TCGdex, because
    the Pokemon TCG API has no prices for Ascended Heroes / Pitch Black) and fill in illustrator
    names the first API leaves empty (e.g. all of Prismatic Evolutions). Cards are matched by
    set + number and the name is checked.
Optional: set the POKEMONTCG_API_KEY environment variable for higher rate limits.

Output (inside config.DATA_DIR):
    cards/<id>.png        RGB PNG, transparent rounded corners painted CORNER_FILL
    raw/<set>.json        the raw API answer, kept so tags can be re-derived without downloading
    manifest.csv          columns below; image_path is relative to DATA_DIR
Re-running is safe: saved card lists (raw/) and images already on disk are reused (use --refresh
to download the lists again), and the manifest is rewritten after each set, so a crash halfway
keeps the sets that were finished.

`art_group` (cards that share identical artwork, for the *lenient* metric) is filled by
`python -m cardrec.art_groups`; re-running this script keeps the existing values.
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import requests
from tqdm import tqdm

from . import config

API = "https://api.pokemontcg.io/v2/cards"
TCGDEX = "https://api.tcgdex.net/v2/en"
CORNER_FILL = (128, 128, 128)       # BGR used where the PNG is transparent (the rounded corners)

COLUMNS = ["id", "name", "set_id", "set_name", "set_code", "number", "printed_total",
           "supertype", "subtypes", "types", "rarity", "regulation_mark", "artist", "artist_source",
           "language", "image_path", "art_group", "tcgdex_id"]


def get_json(url, params=None, headers=None, retries=15):
    """GET with retries: waits 3, 6, 9 ... (max 30) s after a server error (5xx), rate limit (429)
    or network error. In Oct 2026 the Pokemon TCG API failed about half of all requests."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()                    # e.g. 404: retrying will not help
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(min(3 * (attempt + 1), 30))
    raise RuntimeError(f"giving up on {url} {params or ''} after {retries} tries")


def fetch_set(set_id, api_key=None):
    headers = {"X-Api-Key": api_key} if api_key else {}
    cards, page = [], 1
    while True:
        payload = get_json(API, {"q": f"set.id:{set_id}", "page": page, "pageSize": 250}, headers)
        cards += payload["data"]
        if not payload["data"] or len(cards) >= payload["totalCount"]:
            break
        page += 1
    return cards


def save_image(url, path, retries=3):
    """Download one image and store it as a 3-channel PNG with the transparent corners filled."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("not an image")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:                       # alpha blend onto a fixed colour
                alpha = img[..., 3:4].astype(np.float32) / 255.0
                img = (img[..., :3] * alpha + np.array(CORNER_FILL, np.float32) * (1 - alpha)).astype(np.uint8)
            cv2.imwrite(str(path), img)
            return True
        except (requests.RequestException, ValueError):
            time.sleep(1.5 * (attempt + 1))
    return False


def tidy_rarity(r):
    """A few new cards use codes like 'MEGA_ATTACK_RARE'; turn them into 'Mega Attack Rare'."""
    return r.replace("_", " ").title() if r and r.isupper() and "_" in r else (r or "")


def card_row(c, path):
    s = c["set"]
    return {
        "id": c["id"], "name": c.get("name", ""),
        "set_id": s["id"], "set_name": s["name"], "set_code": s.get("ptcgoCode", ""),
        "number": c.get("number", ""), "printed_total": s.get("printedTotal", ""),
        "supertype": c.get("supertype", ""),                       # Pokémon / Trainer / Energy
        "subtypes": ";".join(c.get("subtypes") or []),             # e.g. Basic;ex  or  Supporter
        "types": ";".join(c.get("types") or []),                   # Pokémon energy type(s)
        "rarity": tidy_rarity(c.get("rarity")),
        "regulation_mark": c.get("regulationMark", ""),
        "artist": c.get("artist", ""), "artist_source": "pokemontcg" if c.get("artist") else "",
        "language": "en",                                          # all reference images are English
        "image_path": str(path.relative_to(config.DATA_DIR)).replace("\\", "/"),
        "art_group": "", "tcgdex_id": "",
    }


def _same_name(a, b):
    norm = lambda s: s.lower().replace("é", "e").replace("’", "'").strip()
    return norm(a) == norm(b)


def link_tcgdex(rows, tcgdex_set):
    """Store each card's TCGdex id (used for prices) and fill missing illustrators from TCGdex.
    Cards are matched by number ('003' == '3') and the names must agree."""
    if not tcgdex_set:
        return 0
    listing = get_json(f"{TCGDEX}/sets/{tcgdex_set}")["cards"]
    by_number = {c["localId"].lstrip("0"): c for c in listing}
    for r in rows:
        brief = by_number.get(r["number"].lstrip("0"))
        if brief is not None and _same_name(brief["name"], r["name"]):
            r["tcgdex_id"] = brief["id"]
        else:
            print(f"  no safe TCGdex match for {r['id']} {r['name']!r}")
    missing = [r for r in rows if not r["artist"] and r["tcgdex_id"]]
    filled = 0
    for r in tqdm(missing, desc=f"illustrators {tcgdex_set} (TCGdex)"):
        artist = get_json(f"{TCGDEX}/cards/{r['tcgdex_id']}").get("illustrator")
        if artist:
            r["artist"], r["artist_source"] = artist, "tcgdex"
            filled += 1
    return filled


def write_manifest(rows):
    with open(config.MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def build(sets=None, size="large", limit=None, refresh=False):
    sets = sets or list(config.SETS)
    api_key = os.environ.get("POKEMONTCG_API_KEY")
    config.CARDS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "raw").mkdir(exist_ok=True)
    print("data folder:", config.DATA_DIR)
    old_groups = {}                                     # keep art_group from an earlier run (art_groups.py)
    if config.MANIFEST.exists():
        with open(config.MANIFEST, encoding="utf-8") as f:
            old_groups = {r["id"]: r.get("art_group", "") for r in csv.DictReader(f)}
    rows = []
    for set_id in sets:
        raw = config.DATA_DIR / "raw" / f"{set_id}.json"
        if raw.exists() and not refresh:                # reuse the saved card list (no API call)
            cards = json.loads(raw.read_text(encoding="utf-8"))
        else:
            cards = fetch_set(set_id, api_key)
            raw.write_text(json.dumps(cards), encoding="utf-8")
        if limit:
            cards = cards[:limit]
        set_rows = []
        for c in tqdm(cards, desc=f"images {set_id}"):
            url = c.get("images", {}).get(size)
            if not url:
                print("no image:", c["id"])
                continue
            path = config.CARDS_DIR / f"{c['id']}.png"
            if not path.exists() and not save_image(url, path):
                print("failed:", c["id"])
                continue
            row = card_row(c, path)
            row["art_group"] = old_groups.get(row["id"], "")
            set_rows.append(row)
        n = link_tcgdex(set_rows, config.SETS.get(set_id))
        print(f"{set_id}: {len(set_rows)} cards, {sum(bool(r['tcgdex_id']) for r in set_rows)} linked to TCGdex, "
              f"{n} illustrators filled from TCGdex, "
              f"{sum(not r['artist'] for r in set_rows)} still without illustrator")
        rows += set_rows
        if rows:
            write_manifest(rows)                        # save progress after every set
    if not rows:
        raise SystemExit("No cards were downloaded - check the set ids and your internet connection.")
    print(f"Manifest written: {len(rows)} unique cards -> {config.MANIFEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--size", choices=["small", "large"], default="large")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="download the card lists again from the API")
    a = ap.parse_args()
    build(a.sets, a.size, a.limit, a.refresh)
