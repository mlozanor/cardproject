"""Japanese reference cards for the app (TCGdex), kept apart from the English benchmark database.

    python -m cardrec.database_ja                 # all sets in JA_SETS (Battle Partners = SV9)
    python -m cardrec.database_ja --sets SV9

Output (inside config.DATA_DIR):
    cards_ja/ja-<tcgdex id>.png   official Japanese scan (transparent corners filled like database.py)
    raw/ja_<set>.json             the raw TCGdex answers, kept so the manifest can be rebuilt offline
    manifest_ja.csv               same columns as manifest.csv, plus en_id / name_en
Only the app reads manifest_ja.csv (config.APP_EXTRA_MANIFESTS -> manifest.load_app_manifest); the
experiments keep using the 785 English cards, so the report's numbers do not change.

Card ids are "ja-<tcgdex id>" (e.g. ja-SV9-061), so they never collide with English ids (sv9-...).
English twin: Japanese numbering differs from the English set, so each card is linked to the English
card with the same artwork using the rule of lang_data.py (checked by eye on the language test set):
same illustrator (+ same Pokedex number for Pokemon), then the closest artwork pHash, accepted only if
the distance is <= JA_MAX_DIST. A linked card gets the English card's art_group (so the app treats
them as the same artwork) and its name in name_en; unlinked cards get their own art_group.
Re-running is safe: images and the raw lists already on disk are reused (--refresh downloads the lists again).
"""
import argparse
import csv
import json

import pandas as pd
from tqdm import tqdm

from . import config
from .database import COLUMNS, get_json, save_image
from .lang_data import JA_MAX_DIST, _art_hash
from .manifest import load_manifest

TCGDEX_JA = "https://api.tcgdex.net/v2/ja"
JA_SETS = {"SV9": ("Battle Partners", "Journey Together")}     # TCGdex id: (English name, English counterpart)
CARDS_JA = config.DATA_DIR / "cards_ja"
MANIFEST_JA = config.APP_EXTRA_MANIFESTS[0]
EXTRA_COLUMNS = ["en_id", "name_en"]
STAGES = {"Stage1": "Stage 1", "Stage2": "Stage 2"}


def fetch_set(set_id, refresh=False):
    raw = config.DATA_DIR / "raw" / f"ja_{set_id}.json"
    if raw.exists() and not refresh:
        return json.loads(raw.read_text(encoding="utf-8"))
    listing = get_json(f"{TCGDEX_JA}/sets/{set_id}")
    cards = [get_json(f"{TCGDEX_JA}/cards/{c['id']}") for c in tqdm(listing["cards"], desc=f"cards {set_id}")]
    raw.parent.mkdir(exist_ok=True)
    raw.write_text(json.dumps({"set": {k: v for k, v in listing.items() if k != "cards"}, "cards": cards},
                              ensure_ascii=False), encoding="utf-8")
    return json.loads(raw.read_text(encoding="utf-8"))


def subtypes(c):
    """Same style as the English manifest: 'Basic;ex', 'Stage 1', 'Supporter', 'Special Energy'."""
    cat = c.get("category", "")
    if cat == "Pokemon":
        out = [STAGES.get(c.get("stage", ""), c.get("stage", ""))]
        if c.get("suffix"):
            out.append(c["suffix"].lower() if c["suffix"].upper() == "EX" else c["suffix"])
        return ";".join(s for s in out if s)
    if cat == "Trainer":
        return c.get("trainerType", "")
    if cat == "Energy":
        return f"{c['energyType']} Energy" if c.get("energyType") else ""
    return ""


def card_row(c, set_info, english_name, path):
    return {
        "id": f"ja-{c['id']}", "name": c.get("name", ""),
        "set_id": f"ja-{set_info['id']}", "set_name": f"{english_name} (Japanese)", "set_code": set_info["id"],
        "number": c.get("localId", ""), "printed_total": (set_info.get("cardCount") or {}).get("official", ""),
        "supertype": "Pokémon" if c.get("category") == "Pokemon" else c.get("category", ""),
        "subtypes": subtypes(c), "types": ";".join(c.get("types") or []),
        "rarity": "" if c.get("rarity") in (None, "None") else c["rarity"],
        "regulation_mark": c.get("regulationMark", ""),
        "artist": c.get("illustrator", ""), "artist_source": "tcgdex" if c.get("illustrator") else "",
        "language": "ja",
        "image_path": str(path.relative_to(config.DATA_DIR)).replace("\\", "/"),
        "art_group": "", "tcgdex_id": c["id"], "en_id": "", "name_en": "",
    }


def link_english(rows, cards, english):
    """Fill en_id / name_en / art_group from the English card with the same artwork (see module doc)."""
    dex = {}
    for set_id in config.SETS:
        for c in json.loads((config.DATA_DIR / "raw" / f"{set_id}.json").read_text(encoding="utf-8")):
            dex[c["id"]] = set(c.get("nationalPokedexNumbers") or [])
    norm = lambda s: str(s).lower().replace(" ", "")
    hashes = {}
    linked = 0
    for row, c in tqdm(list(zip(rows, cards)), desc="link to English"):
        row["art_group"] = row["id"]
        if not row["artist"]:
            continue
        cands = english[english["artist"].map(norm) == norm(row["artist"])]
        if c.get("dexId"):
            cands = cands[cands["id"].map(lambda cid: bool(dex.get(cid, set()) & set(c["dexId"])))]
        if cands.empty:
            continue
        h = _art_hash(config.DATA_DIR / row["image_path"])
        for cid, p in zip(cands["id"], cands["abs_path"]):
            if cid not in hashes:
                hashes[cid] = _art_hash(p)
        dist = {cid: int((h != hashes[cid]).sum()) for cid in cands["id"]}
        best = min(dist, key=dist.get)
        if dist[best] <= JA_MAX_DIST:
            en = english.set_index("id").loc[best]
            row.update(en_id=best, name_en=en["name"], art_group=en["art_group"] or best)
            linked += 1
    return linked


def build(sets=None, refresh=False):
    sets = sets or list(JA_SETS)
    CARDS_JA.mkdir(parents=True, exist_ok=True)
    english = load_manifest()
    rows = []
    for set_id in sets:
        data = fetch_set(set_id, refresh)
        set_rows, set_cards = [], []
        for c in tqdm(data["cards"], desc=f"images {set_id}"):
            if not c.get("image"):
                print("no image:", c["id"])
                continue
            path = CARDS_JA / f"ja-{c['id']}.png"
            if not path.exists() and not save_image(c["image"] + "/high.png", path):
                print("failed:", c["id"])
                continue
            set_rows.append(card_row(c, data["set"], JA_SETS[set_id][0], path))
            set_cards.append(c)
        n = link_english(set_rows, set_cards, english)
        print(f"{set_id}: {len(set_rows)} cards, {n} linked to an English card with the same artwork "
              f"({JA_SETS[set_id][1]} or another English set)")
        rows += set_rows
    with open(MANIFEST_JA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + EXTRA_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Japanese manifest written: {len(rows)} cards -> {MANIFEST_JA}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=None, help=f"TCGdex Japanese set ids (default: {list(JA_SETS)})")
    ap.add_argument("--refresh", action="store_true", help="download the card lists again from TCGdex")
    a = ap.parse_args()
    build(a.sets, a.refresh)
