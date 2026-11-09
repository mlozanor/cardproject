"""Step 7b: language test data from official Spanish / Japanese scans (TCGdex) + evaluation.

    python -m cardrec.lang_data build  [--split dev]   # download samples -> lang[_dev]/lang_cards.csv + review sheet
    python -m cardrec.lang_data photos [--split dev]   # synthetic phone-resolution photos of them
    python -m cardrec.lang_data eval   [--split dev] --methods sift sift@art hybrid-clip@art+rot

Splits: "test" (folder lang/, the first 180 cards, used for the reported numbers) and "dev"
(folder lang_dev/, other cards, used to tune the rules). Never tune on the test split.
History: version 1 of the rules was evaluated ONCE on the test split with low-resolution photos
(card ~530 px tall, OCR on the 500x700 card): accuracy 0.539. Version 2 (kana-only script test,
language-code box OCR, phone resolution, OCR at 750x1050) was tuned on dev only.

Samples (fixed seed):
  en  N_PER_LANG cards of our manifest (the English reference scans)
  es  N_PER_LANG cards of our 4 sets in Spanish. Same card id as English (same set, same number).
  ja  N_PER_LANG cards of the Japanese counterpart sets that have scans in TCGdex: SV8a (Terastal
      Festival ex ~ Prismatic Evolutions) and SV9 (Battle Partners ~ Journey Together). Japanese
      numbering differs, so the English equivalent is found WITHOUT the tested methods:
      same illustrator + same Pokedex number (Pokemon), then the smallest artwork pHash distance;
      accepted only if that distance <= JA_MAX_DIST, otherwise the card is used for the language
      test only. Check lang/ja_matches.jpg by eye; edit lang_cards.csv (true_id) if a match is wrong.
Evaluation: language accuracy + 3x3 confusion matrix (rows = true, cols = predicted, plus "unknown"),
and identification accuracy per language (strict = exact id; for ja the meaningful one is lenient:
same artwork group as the matched English card).
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from . import config
from .art_groups import phash
from .database import get_json
from .evaluate import bootstrap_ci, evaluate_method
from .make_queries import make_queries
from .manifest import load_manifest
from .methods import get_method
from .methods.base import load_canonical
from .methods.cache import build_or_load
from .rectify import rectify, warp_to_canonical

N_PER_LANG = 60
PHONE_CANVAS = (1920, 2560)         # card ~1050 px tall, like a 12 MP phone photo
SPLIT_SEED = {"test": 0, "dev": 1000}


def lang_dir(split):
    return config.DATA_DIR / ("lang" if split == "test" else f"lang_{split}")


def query_dir(split, severity):
    return config.QUERIES_DIR / f"lang_{split}_hr_s{severity}"
JA_SETS = ["SV8a", "SV9"]
JA_MAX_DIST = 14
LANGS = ["en", "es", "ja"]
TCGDEX = "https://api.tcgdex.net/v2"


def _download(url, path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image")
    cv2.imwrite(str(path), img)


def _art_hash(path):
    y0, y1, x0, x1 = config.ART_BOX
    return phash(cv2.cvtColor(load_canonical(path), cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1])


def build(manifest, split="test"):
    out = lang_dir(split)
    for lang in LANGS:
        (out / lang).mkdir(parents=True, exist_ok=True)
    seed = config.SEED + SPLIT_SEED[split]
    rng = np.random.default_rng(seed)
    exclude = set()
    if split != "test" and (lang_dir("test") / "lang_cards.csv").exists():      # dev must not reuse test cards
        exclude = set(pd.read_csv(lang_dir("test") / "lang_cards.csv", dtype=str)["source_id"])
    manifest = manifest[~manifest["id"].isin(exclude) & ~("es:" + manifest["tcgdex_id"]).isin(exclude)]
    rows = []
    for r in manifest.sample(N_PER_LANG, random_state=seed).itertuples():                   # English
        rows.append({"language": "en", "path": r.abs_path, "true_id": r.id, "source_id": r.id, "match_dist": ""})
    picks = manifest.sample(N_PER_LANG * 2, random_state=seed + 1)                           # Spanish
    for r in tqdm(list(picks.itertuples()), desc="es"):
        if sum(x["language"] == "es" for x in rows) >= N_PER_LANG:
            break
        path = out / "es" / f"{r.id}.png"
        try:
            if not path.exists():
                _download(get_json(f"{TCGDEX}/es/cards/{r.tcgdex_id}")["image"] + "/high.png", path)
        except Exception as e:                      # some Spanish scans are missing
            print("  es skip", r.id, str(e)[:60])
            continue
        rows.append({"language": "es", "path": str(path), "true_id": r.id, "source_id": f"es:{r.tcgdex_id}",
                     "match_dist": ""})
    rows += build_japanese(manifest, rng, out, exclude)
    pd.DataFrame(rows).to_csv(out / "lang_cards.csv", index=False, encoding="utf-8")
    print(pd.DataFrame(rows).groupby("language")["true_id"].apply(lambda s: f"{len(s)} cards, {(s != '').sum()} with id"))


def build_japanese(manifest, rng, out, exclude=()):
    raw = {}
    for set_id in config.SETS:
        for c in json.loads((config.DATA_DIR / "raw" / f"{set_id}.json").read_text(encoding="utf-8")):
            raw[c["id"]] = c.get("nationalPokedexNumbers") or []
    norm = lambda s: str(s).lower().replace(" ", "")
    listing = [c for s in JA_SETS for c in get_json(f"{TCGDEX}/ja/sets/{s}")["cards"]
               if c.get("image") and f"ja:{c['id']}" not in exclude]
    picks = rng.choice(len(listing), size=min(N_PER_LANG, len(listing)), replace=False)
    hashes = {}
    rows, sheet = [], []
    for i in tqdm(picks, desc="ja"):
        brief = listing[i]
        path = out / "ja" / f"{brief['id']}.png"
        try:
            card = get_json(f"{TCGDEX}/ja/cards/{brief['id']}")
            if not path.exists():
                _download(brief["image"] + "/high.png", path)
        except Exception as e:
            print("  ja skip", brief["id"], str(e)[:60])
            continue
        cands = manifest[manifest["artist"].map(norm) == norm(card.get("illustrator", ""))]
        if card.get("dexId"):
            cands = cands[cands["id"].map(lambda cid: bool(set(raw.get(cid, [])) & set(card["dexId"])))]
        best, dist = "", ""
        if len(cands):
            h = _art_hash(path)
            for cid, p in zip(cands["id"], cands["abs_path"]):
                hashes.setdefault(cid, _art_hash(p))
            d = {cid: int((h != hashes[cid]).sum()) for cid in cands["id"]}
            cid = min(d, key=d.get)
            dist = d[cid]
            if dist <= JA_MAX_DIST:
                best = cid
                sheet.append((path, manifest.set_index("id").loc[cid, "abs_path"], f"{brief['id']} -> {cid} d={dist}"))
        rows.append({"language": "ja", "path": str(path), "true_id": best, "source_id": f"ja:{brief['id']}",
                     "match_dist": dist})
    if sheet:                                       # contact sheet: Japanese scan | matched English card
        tiles = []
        for pj, pe, label in sheet:
            pair = np.hstack([cv2.resize(load_canonical(pj), (120, 168)), cv2.resize(load_canonical(pe), (120, 168))])
            cv2.putText(pair, label, (2, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            tiles.append(pair)
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
        cv2.imwrite(str(out / "ja_matches.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    return rows


def make_photos(severity=2, split="test"):
    cards = pd.read_csv(lang_dir(split) / "lang_cards.csv", dtype=str).fillna("")
    fake = pd.DataFrame({"abs_path": cards["path"], "id": cards["source_id"]})     # make_queries needs these
    q = make_queries(fake, len(fake), severity, seed=config.SEED + 77 + SPLIT_SEED[split],
                     out_dir=query_dir(split, severity), canvas=PHONE_CANVAS).astype(str)
    info = cards.set_index("source_id")
    q["language"] = q["true_id"].map(info["language"])
    q["source_id"] = q["true_id"]
    q["true_id"] = q["source_id"].map(info["true_id"])
    q.to_csv(query_dir(split, severity) / "queries.csv", index=False)
    print(q.groupby("language").size().to_string())


def confusion(df):
    cm = pd.crosstab(df["language"], df["pred_language"]).reindex(index=LANGS, columns=LANGS + ["unknown"],
                                                                 fill_value=0)
    cm.index.name, cm.columns.name = "true", "predicted"
    return cm


def evaluate(manifest, methods, severity=2, split="test", k=20):
    from .condition import COND_H, COND_W
    from .language import LanguageClassifier
    q = pd.read_csv(query_dir(split, severity) / "queries.csv", dtype=str).fillna("")
    out = config.RESULTS_DIR / ("language" if split == "test" else f"language_{split}")
    out.mkdir(parents=True, exist_ok=True)
    clf, rows = LanguageClassifier(), []
    for r in tqdm(list(q.itertuples()), desc="language"):
        img = cv2.imread(str(config.DATA_DIR / r.image_path))
        _, quad, found = rectify(img)
        # OCR needs more pixels than identification: warp the ORIGINAL photo at 750 x 1050
        card = warp_to_canonical(img, quad, COND_W, COND_H) if found else cv2.resize(img, (COND_W, COND_H))
        res = clf.classify(card)
        rows.append({"query_id": r.query_id, "language": r.language, "pred_language": res.pop("language"),
                     "rectify_found": found, **res})
    lang = pd.DataFrame(rows)
    lang.to_csv(out / f"language_v2_s{severity}.csv", index=False)
    ok = (lang["language"] == lang["pred_language"]).astype(float)
    lo, hi = bootstrap_ci(ok)
    cm = confusion(lang)
    cm.to_csv(out / f"confusion_v2_s{severity}.csv")
    print(f"language accuracy {ok.mean():.3f} [{lo:.3f}, {hi:.3f}] on {len(lang)} photos\n{cm.to_string()}")

    groups = dict(zip(manifest["id"], manifest["art_group"]))
    ident = []
    for name in methods:
        m = build_or_load(get_method(name, k=k), manifest)
        for lg in LANGS:
            sub = q[(q["language"] == lg) & (q["true_id"] != "")]
            if len(sub):
                s, log = evaluate_method(m, sub, True, out / f"{m.name}_{lg}_hr_s{severity}.csv", groups)
                ident.append({**s, "language": lg})
    if ident:
        df = pd.DataFrame(ident)
        df.to_csv(config.RESULTS_DIR / f"summary_language_id_{split}_hr.csv", index=False)
        print(df[["method", "language", "n_queries", "top1", "top1_lo", "top1_hi", "top1_len", "top1_len_lo",
                  "top1_len_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "photos", "eval"])
    ap.add_argument("--severity", type=int, default=2)
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--methods", nargs="*", default=["sift", "sift@art", "clip", "clip@art",
                                                       "hybrid-clip+rot", "hybrid-clip@art+rot"])
    a = ap.parse_args()
    man = load_manifest()
    if a.cmd == "build":
        build(man, a.split)
    elif a.cmd == "photos":
        make_photos(a.severity, a.split)
    else:
        evaluate(man, a.methods, a.severity, a.split)
