"""Loads the reference database and a retrieval method (with an on-disk index cache) and
analyzes one uploaded photo: rectify -> rank all cards -> top-N candidates."""
import threading
import time

import cv2
import numpy as np
import pandas as pd

from .. import config
from ..condition import COND_H, COND_W, assess
from ..demo_offline import make_fake_card
from ..manifest import load_app_manifest
from ..methods import get_method
from ..methods.cache import build_or_load
from ..rectify import find_card_quads, rectify, refine_quad, warp_to_canonical
from ..synth import render_scene

METHODS = ["sift", "orb", "resnet", "clip", "hybrid-clip", "hybrid-resnet",
           "hybrid-clip+rot", "hybrid-clip@art+rot"]    # @art: artwork only, for non-English cards

# Heuristic confidence for keypoint-style scores (inlier counts). Calibrate on your own data!
MIN_INLIERS = 15
MIN_MARGIN = 1.4          # top-1 must have >= 1.4x the inliers of top-2


def build_demo_manifest(n_cards=30, n_samples=6):
    """Fake cards + a few distorted 'photos' you can upload to try the interface offline."""
    demo = config.DATA_DIR / "demo"
    (demo / "cards").mkdir(parents=True, exist_ok=True)
    (demo / "samples").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    rows, cards = [], []
    for i in range(n_cards):
        p = demo / "cards" / f"fake_{i:03d}.png"
        card = make_fake_card(rng, i)
        if not p.exists():
            cv2.imwrite(str(p), card)
        cards.append(card)
        rows.append({"id": f"fake_{i:03d}", "name": f"Demo Card {i:03d}", "set_name": "Demo Set",
                     "number": str(i), "abs_path": str(p)})
    if not any((demo / "samples").glob("*.png")):
        for j in range(n_samples):
            idx = int(rng.integers(n_cards))
            scene, _ = render_scene([cards[idx]], rng, severity=2 + j % 2)
            cv2.imwrite(str(demo / "samples" / f"photo_{j}_of_demo_card_{idx:03d}.png"), scene)
    return pd.DataFrame(rows)


class Engine:
    def __init__(self, demo=False, default_method="sift"):
        self.demo, self.default_method = demo, default_method
        if demo:
            self.manifest = build_demo_manifest()
        else:
            if not config.MANIFEST.exists():
                raise SystemExit("data/manifest.csv not found. Run `python -m cardrec.database` first "
                                 "(or start the app with --demo to try it without real data).")
            self.manifest = load_app_manifest()             # English + extra (e.g. Japanese) sets
        self.ids = self.manifest["id"].tolist()
        self.paths = self.manifest["abs_path"].tolist()
        self._by_id = self.manifest.set_index("id").to_dict("index")
        self._methods, self._lock, self.ready = {}, threading.Lock(), False
        self._lang = None                                   # OCR language classifier, loaded on first use

    def meta(self, card_id):
        row = self._by_id.get(card_id)
        return None if row is None else {"id": card_id, "name": row.get("name", ""),
                                         "set_name": row.get("set_name", ""), "number": row.get("number", ""),
                                         "tcgdex_id": row.get("tcgdex_id", ""),
                                         "language": row.get("language", "") or "en", "tags": self.tags(card_id)}

    def tags(self, card_id):
        """Tags of a card, taken from the database metadata (manifest) AFTER it was identified.
        Returns a list of {key, label, value}; empty values are left out. The printed language of
        the user's own copy is not here (it is stored per collection item, see db.py)."""
        r = self._by_id.get(card_id) or {}
        g = lambda k: str(r.get(k, "") or "")
        category = g("supertype").replace("é", "e")              # "Pokémon" -> "Pokemon"
        subtypes = [s for s in g("subtypes").split(";") if s]
        number = g("number") + (f"/{g('printed_total')}" if g("printed_total") else "")
        values = [
            ("set", "Set", f"{g('set_name')} ({g('set_code')})" if g("set_code") else g("set_name")),
            ("number", "Number", number),
            ("name", "Trainer name" if category == "Trainer" else "Card name", g("name")),
            ("category", "Category", category),
            ("trainer_subtype", "Trainer type", ", ".join(subtypes) if category == "Trainer" else ""),
            ("stage", "Stage", ", ".join(subtypes) if category == "Pokemon" else ""),
            ("energy_kind", "Energy type", ", ".join(subtypes) if category == "Energy" else ""),
            ("pokemon_type", "Pokemon type", ", ".join(t for t in g("types").split(";") if t)),
            ("rarity", "Rarity", g("rarity")),
            ("regulation_mark", "Regulation mark", g("regulation_mark")),
            ("illustrator", "Illustrator", g("artist")),
            # Japanese cards: the English card with the same artwork (database_ja.py)
            ("english_version", "English version", f"{g('name_en')} ({g('en_id')})" if g("en_id") else ""),
        ]
        return [{"key": k, "label": lbl, "value": v} for k, lbl, v in values if v]

    def path(self, card_id):
        row = self._by_id.get(card_id)
        return None if row is None else row["abs_path"]

    def get(self, name):
        with self._lock:
            if name not in self._methods:
                self._methods[name] = build_or_load(get_method(name, k=20), paths=self.paths, ids=self.ids)
            return self._methods[name]

    def warmup(self):
        try:
            self.get(self.default_method)
        finally:
            self.ready = True

    def language(self, img_bgr, quad):
        """Printed language of the card (en/es/ja) with OCR, or None if EasyOCR is not installed."""
        with self._lock:
            if self._lang is None:
                try:
                    from ..language import LanguageClassifier
                    self._lang = LanguageClassifier()
                except ImportError:
                    self._lang = False                     # remember: not available
        if not self._lang:
            return None
        card = warp_to_canonical(img_bgr, quad, COND_W, COND_H) if quad is not None else \
            cv2.resize(img_bgr, (COND_W, COND_H))
        r = self._lang.classify(card)
        return {"code": r["language"] if r["language"] != "unknown" else "", "rule": r["rule"]}

    def condition(self, img_bgr, card_id, side="front"):
        """Visual condition ESTIMATE of one side (see condition.py). Front: the official scan is the
        reference; back: <data>/condition/back_reference.jpg if you saved a clean back photo there.
        Returns (result, straightened card photo without any drawing on it) or (None, None)."""
        quads = find_card_quads(img_bgr, max_cards=1)
        if not quads:
            return None, None
        card = warp_to_canonical(img_bgr, refine_quad(img_bgr, quads[0]), COND_W, COND_H)
        if side == "front":
            ref = cv2.imread(self.path(card_id)) if self.path(card_id) else None
        else:
            back = config.DATA_DIR / "condition" / "back_reference.jpg"
            ref = cv2.imread(str(back)) if back.exists() else None
        return assess(card, ref, side), card

    def analyze(self, img_bgr, method=None, topn=5, detect_language=True):
        method = method or self.default_method
        m = self.get(method)
        t0 = time.perf_counter()
        card, quad, found = rectify(img_bgr)
        ranked = m.query(card, topk=topn)
        ms = 1000 * (time.perf_counter() - t0)
        language = self.language(img_bgr, quad) if detect_language else None
        kind = "inliers" if method.startswith(("sift", "orb", "hybrid")) else "similarity"
        cands = []
        for cid, score in ranked:
            c = self.meta(cid)
            c["score"] = round(score * 100, 1) if kind == "similarity" else float(score)
            cands.append(c)
        confident = None
        if kind == "inliers" and cands:
            s1 = cands[0]["score"]
            s2 = cands[1]["score"] if len(cands) > 1 else 0
            confident = bool(s1 >= MIN_INLIERS and s1 >= MIN_MARGIN * max(s2, 1))
        return {"rect": card, "found": bool(found), "method": method, "score_kind": kind,
                "candidates": cands, "confident": confident, "ms": round(ms), "language": language}
