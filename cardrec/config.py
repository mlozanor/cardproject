"""Central settings. Edit SETS and the sizes here; everything else reads from this file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Big data (card images, queries, caches, collection) can live outside the project folder
# (e.g. outside OneDrive): set the environment variable CARDREC_DATA. Default: <project>/data.
DATA_DIR = Path(os.environ.get("CARDREC_DATA") or ROOT / "data").resolve()
CARDS_DIR = DATA_DIR / "cards"
MANIFEST = DATA_DIR / "manifest.csv"
QUERIES_DIR = DATA_DIR / "queries"
RESULTS_DIR = ROOT / "results"

# Pokemon TCG API set ids in the reference database (checked against /v2/sets on 2026-10-21).
# Value = the same set's id in TCGdex, used only to fill in missing illustrator names.
SETS = {
    "sv8pt5": "sv08.5",    # Prismatic Evolutions (PRE)
    "sv9": "sv09",         # Journey Together (JTG)
    "me2pt5": "me02.5",    # Ascended Heroes (ASC)
    "me5": "me05",         # Pitch Black (PBL)
}

# Extra reference sets used ONLY by the app (CardDex), not by the experiments: the report's benchmark
# stays the 785 English cards of SETS. Built by `python -m cardrec.database_ja` (Japanese sets, TCGdex).
APP_EXTRA_MANIFESTS = [DATA_DIR / "manifest_ja.csv"]

# Canonical card size after rectification (2.5 x 3.5 inch card -> 5:7 ratio).
CANON_W, CANON_H = 500, 700
# Artwork window on the canonical card: y0, y1, x0, x1 (regular Scarlet & Violet / Mega Evolution
# layout; on full-art cards it is simply the upper-middle part of the art). Same in every language.
ART_BOX = (70, 340, 35, 465)

SEED = 42
