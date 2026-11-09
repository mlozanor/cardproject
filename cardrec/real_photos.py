"""Step 6: real phone photos - labelling helper and evaluation.

Folder: <data>/real/photos/   (put your .jpg/.png phone photos here, any file names)
    python -m cardrec.real_photos labels            # create/extend <data>/real/labels.csv with suggestions
    (open labels.csv in Excel, fill in true_ids and the condition columns, save as CSV UTF-8)
    python -m cardrec.real_photos check             # validate the labels (unknown ids, missing fields)
    python -m cardrec.real_photos eval --methods sift hybrid-clip+rot ...

labels.csv columns (one row per photo):
  photo         file name inside real/photos
  true_ids      the card(s) in the photo, separated by ';' ',' or '+'. Either the database id (me5-120) or the
                set code + number printed at the bottom of the card (PBL 120, "PBL 120/084" also works).
  side          front / back            language   en / es / ja (the printed language)
  background    plain / patterned / cluttered      lighting   good / dim / mixed
  glare, sleeve, blur, overlap   yes / no           notes      free text
  suggest_1..3  filled by the script (best guesses of hybrid-clip+rot): ONLY a typing aid, you must
                check the card yourself - the ground truth must not come from the method being tested.

Evaluation:
  single-card photos (one id, side=front)  -> same metrics as the synthetic queries (evaluate.py)
  multi-card photos (several ids)          -> detect all cards, identify each; compare the SET of
      predicted ids with the true set: id_precision = correct predictions / predictions,
      id_recall = true cards found / true cards (no corner ground truth is needed).
Logs: results/real/<method>_single.csv and <method>_multi.csv; summary: results/summary_real.csv.
The condition columns are copied into the logs so failures can be split by glare, sleeve, etc.
"""
import argparse
import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .evaluate import bootstrap_ci, evaluate_method
from .manifest import load_manifest
from .methods import get_method
from .methods.cache import build_or_load
from .rectify import find_card_quads, rectify, warp_to_canonical

REAL = config.DATA_DIR / "real"
PHOTOS = REAL / "photos"
LABELS = REAL / "labels.csv"
CONDITION_COLS = ["side", "language", "background", "lighting", "glare", "sleeve", "blur", "overlap"]
COLUMNS = ["photo", "true_ids", *CONDITION_COLS, "notes", "suggest_1", "suggest_2", "suggest_3"]
SUGGEST_METHOD = "hybrid-clip+rot"


def resolve_id(token, manifest):
    """'me5-120' -> 'me5-120';  'PBL 120' / 'pbl 120/084' / 'PBL-120' -> 'me5-120';  unknown -> None."""
    token = token.strip()
    ids = set(manifest["id"])
    if token in ids:
        return token
    mt = re.fullmatch(r"([A-Za-z]{2,4})[\s\-_]*0*(\d+)(?:\s*/\s*\d+)?", token)
    if not mt:
        return None
    hit = manifest[(manifest["set_code"].str.upper() == mt.group(1).upper())
                   & (manifest["number"].str.lstrip("0") == mt.group(2))]
    return hit["id"].iloc[0] if len(hit) == 1 else None


def parse_ids(cell, manifest):
    tokens = [t for t in re.split(r"[;,+]", str(cell)) if t.strip()]
    return [resolve_id(t, manifest) for t in tokens], tokens


def read_labels():
    """Excel may save with ',' or ';' between columns (depends on Windows region): detect it."""
    return pd.read_csv(LABELS, dtype=str, sep=None, engine="python", encoding="utf-8-sig").fillna("")


def make_labels(manifest):
    PHOTOS.mkdir(parents=True, exist_ok=True)
    old = read_labels() if LABELS.exists() else pd.DataFrame(columns=COLUMNS)
    photos = sorted(p.name for p in PHOTOS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    new = [p for p in photos if p not in set(old["photo"])]
    if not new:
        print(f"no new photos in {PHOTOS} ({len(old)} already labelled)")
        return
    method = build_or_load(get_method(SUGGEST_METHOD), manifest)
    names = dict(zip(manifest["id"], manifest["name"] + " (" + manifest["set_code"] + " " + manifest["number"] + ")"))
    rows = []
    for p in tqdm(new, desc="suggestions"):
        card, _, _ = rectify(cv2.imread(str(PHOTOS / p)))
        top = [cid for cid, _ in method.query(card, topk=3)]
        rows.append({"photo": p, **{c: "" for c in COLUMNS[1:]},
                     **{f"suggest_{i + 1}": f"{cid} = {names[cid]}" for i, cid in enumerate(top)}})
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)[COLUMNS]
    out.to_csv(LABELS, index=False, encoding="utf-8-sig")       # utf-8-sig: Excel opens accents correctly
    print(f"{len(new)} new photos added -> {LABELS}")


def check_labels(manifest, verbose=True):
    lab = read_labels()
    problems = []
    for r in lab.itertuples():
        ids, tokens = parse_ids(r.true_ids, manifest)
        if not tokens:
            problems.append(f"{r.photo}: true_ids is empty")
        for i, t in zip(ids, tokens):
            if i is None:
                problems.append(f"{r.photo}: cannot resolve '{t}'")
    if verbose:
        print(f"{len(lab)} photos, {len(problems)} problems")
        for p in problems[:50]:
            print("  ", p)
    return lab, problems


def eval_multi(method, rows, art_group):
    """Detect every card, identify each, compare predicted id set with the true id set."""
    out = []
    for r in tqdm(rows, desc=f"multi {method.name}"):
        img = cv2.imread(str(PHOTOS / r["photo"]))
        pred = [method.query(warp_to_canonical(img, q), topk=1)[0][0] for q in find_card_quads(img)]
        true = r["ids"]
        g = lambda c: art_group.get(c, c)                       # same-artwork group of a card id
        out.append({"photo": r["photo"], "n_true": len(true), "n_pred": len(pred),
                    "correct": len(set(pred) & set(true)),
                    "correct_lenient": len({g(c) for c in pred} & {g(c) for c in true}),
                    "pred_ids": ";".join(pred), "true_ids": ";".join(true),
                    **{c: r[c] for c in CONDITION_COLS}})
    return pd.DataFrame(out)


def evaluate_real(manifest, methods, k=20):
    lab, problems = check_labels(manifest, verbose=False)
    if problems:
        raise SystemExit(f"{len(problems)} label problems - run `python -m cardrec.real_photos check` first")
    groups = dict(zip(manifest["id"], manifest["art_group"]))
    single, multi = [], []
    for r in lab.to_dict("records"):
        r["ids"] = parse_ids(r["true_ids"], manifest)[0]
        if len(r["ids"]) == 1 and r["side"] != "back":
            single.append(r)
        elif len(r["ids"]) > 1:
            multi.append(r)
    queries = pd.DataFrame([{"query_id": r["photo"], "true_id": r["ids"][0], "severity": "real",
                             "image_path": f"real/photos/{r['photo']}"} for r in single])
    cond = pd.DataFrame([{"query_id": r["photo"], **{c: r[c] for c in CONDITION_COLS}} for r in single])
    out_dir = config.RESULTS_DIR / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in methods:
        m = build_or_load(get_method(name, k=k), manifest)
        if len(queries):
            s, log = evaluate_method(m, queries, True, None, groups)
            log.merge(cond, on="query_id").to_csv(out_dir / f"{m.name}_single.csv", index=False)
            summaries.append({**s, "subset": "single"})
        if multi:
            ml = eval_multi(m, multi, groups)
            ml.to_csv(out_dir / f"{m.name}_multi.csv", index=False)
            per_photo_recall = ml["correct"] / ml["n_true"]
            lo, hi = bootstrap_ci(per_photo_recall)
            summaries.append({"method": m.name, "subset": "multi", "n_queries": len(ml),
                              "id_precision": ml["correct"].sum() / max(ml["n_pred"].sum(), 1),
                              "id_recall": ml["correct"].sum() / ml["n_true"].sum(),
                              "id_recall_lenient": ml["correct_lenient"].sum() / ml["n_true"].sum(),
                              "photo_recall_lo": lo, "photo_recall_hi": hi})
    df = pd.DataFrame(summaries)
    df.to_csv(config.RESULTS_DIR / "summary_real.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels", "check", "eval"])
    ap.add_argument("--methods", nargs="*", default=["orb", "sift", "resnet", "clip", "hybrid-resnet+rot",
                                                       "hybrid-clip+rot"])
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    man = load_manifest()
    {"labels": lambda: make_labels(man), "check": lambda: check_labels(man),
     "eval": lambda: evaluate_real(man, a.methods, a.k)}[a.cmd]()
