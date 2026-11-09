# Pokemon Card Detection and Recognition (CAP 5410 Final Project)

Compares classical (SIFT/ORB + RANSAC), CNN (ResNet-50) and transformer (CLIP) retrieval for identifying
Pokemon cards in photos, plus a hybrid (embedding shortlist re-ranked by SIFT + RANSAC).
Extras: card tags, printed-language recognition (en/es/ja), and a visual condition estimate.
Author: Mariana Lozano. Type B experimental comparison. Report: IEEE conference template.

**Rule for this README: every number below was measured; anything not run is marked NOT RUN.**

## 1. Setup (Windows)
The project folder is inside OneDrive, so the venv and the big data folder live OUTSIDE it
(OneDrive would otherwise try to sync tens of thousands of files).
```
py -3.12 -m venv %USERPROFILE%\.venvs\cardproject
%USERPROFILE%\.venvs\cardproject\Scripts\activate
pip install -r requirements.txt
setx CARDREC_DATA C:\Users\MarianaLozano\cardrec_data     (once; open a NEW terminal afterwards)
```
- `CARDREC_DATA` = where card images, queries, scenes, caches, uploads and the collection are stored.
  If it is not set, `data/` inside the project is used. `results/` (CSV logs, figures) stays in the project.
- VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> "Enter interpreter path" ->
  `C:\Users\MarianaLozano\.venvs\cardproject\Scripts\python.exe`. Restart VS Code after `setx`.
- Deep learning (installed and tested 2026-10-21, GPU: RTX 3050 Laptop 4 GB):
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install transformers
  pip install easyocr --no-deps          (--no-deps: it would install a second, conflicting OpenCV)
  pip install scipy scikit-image python-bidi pyclipper shapely ninja
  ```
  Tested versions: Python 3.12.10, torch 2.11.0+cu128, torchvision 0.26.0, transformers 5.20.2, easyocr 1.7.2
  (core packages pinned in requirements.txt). `pip check` warns that easyocr wants opencv-python-headless: expected.

## 2. Quick check (no internet or data needed)
```
python -m cardrec.demo_offline          # fake cards -> distort -> rectify -> retrieve -> metrics table
python -m cardrec.app.server --demo     # CardDex with fake cards
```

## 3. Reference database (built 2026-10-21)
```
python -m cardrec.database      # card lists + images + manifest.csv (re-run safe; --refresh re-downloads lists)
python -m cardrec.art_groups    # fills art_group (same artwork), writes art_group_pairs.csv/.jpg to check by eye
```

| Set | API id | Code | Cards in manifest | Printed set size |
|---|---|---|---|---|
| Prismatic Evolutions | sv8pt5 | PRE | 180 | 131 |
| Journey Together | sv9 | JTG | 190 | 159 |
| Ascended Heroes | me2pt5 | ASC | 295 | 217 |
| Pitch Black | me5 | PBL | 120 | 84 |
| **Total** | | | **785 unique card ids** | 591 printed |

- Counts include secret rares (numbers above the printed size). 629 Pokemon, 150 Trainer, 6 Energy.
  The Pitch Black count matches pokecottage.com's list (120, number 120 = Mega Darkrai ex).
- **Unique artworks: 761.** 24 pairs share identical artwork (Ascended Heroes reprints 9 PRE and 15 JTG cards,
  e.g. N's Reshiram sv9-116 = me2pt5-154). Found with a 64-bit perceptual hash (pHash) of the whole card and of
  the artwork box (both <= 6 bits different; the next closest unrelated pair is >= 12); all 24 checked by eye.
- **Strict metric** = exact card id. **Lenient metric** = same art_group (a reprint counts as correct).
  163 names appear on more than one card (391 cards, e.g. 8 versions of "Black Belt's Training") = lookalikes.
- Sources: Pokemon TCG API (pokemontcg.io): list, metadata, images (733x1024 PNG, transparent corners filled
  grey). In Oct 2026 it answered 500/502 to about half the requests, so every request is retried.
  TCGdex (tcgdex.net): illustrators missing in the first API (all 180 PRE cards + 7 PBL; 6 special Energy cards
  have none in either source), each card's TCGdex id (for prices), Spanish/Japanese scans (section 7).
- Manifest columns: id, name, set_id, set_name, set_code, number, printed_total, supertype, subtypes, types,
  rarity, regulation_mark, artist, artist_source, language, image_path, art_group, tcgdex_id.

### 3b. Japanese cards for the app (built 2026-11-08)
```
python -m cardrec.database_ja        # Battle Partners (SV9, Japanese) from TCGdex -> manifest_ja.csv, cards_ja/
```
- 132 cards (100 + 32 secret rares). 128 are linked to the English card with the same artwork (126 Journey
  Together, 2 Ascended Heroes reprints) with the lang_data rule (illustrator + Pokedex number + artwork pHash
  <= 14). All 128 pairs were checked by eye on `<data>/ja_links.jpg`. Unlinked: the 2 Spike Energy cards
  (no illustrator), N's Zorua 108 and Dudunsparce ex 120.
- Ids are `ja-<TCGdex id>` (e.g. `ja-SV9-061`). Linked cards share the English card's `art_group`; the app shows
  an "English version" tag. Secret rares have no rarity in TCGdex.
- **Used only by the app** (`config.APP_EXTRA_MANIFESTS` -> `manifest.load_app_manifest`). The experiments and
  the report's benchmark still use the 785 English cards.
- Prices of Japanese cards: TCGdex Japanese catalogue = Cardmarket average in EUR (no TCGplayer prices).
  The collection total adds USD and EUR prices as they are (no conversion).
- To add another Japanese set: add it to `JA_SETS` in `cardrec/database_ja.py` (TCGdex id: English name,
  English counterpart) and re-run. Delete nothing: the app builds new index caches automatically.

## 4. Methods (all behind the same interface in `cardrec/methods`)
`build_index(paths, ids)` and `query(card_image) -> full ranking`, so every method is evaluated identically.

| Name | What it does |
|---|---|
| `pixel` | 32x44 grey thumbnail, cosine similarity (sanity baseline) |
| `orb`, `sift` | keypoints + ratio test + RANSAC homography, score = inlier count, against every card |
| `resnet` | frozen ImageNet ResNet-50, 2048-d global embedding, cosine similarity |
| `clip` | frozen CLIP ViT-B/32 image embedding (512-d). Note: CLIP's preprocessing centre-crops the card to a square, so the top and bottom ~13% are not seen |
| `hybrid-<emb>` | top-k shortlist from the embedding, re-ranked by SIFT inliers (`--k`, default 20) |
| suffix `+rot` | embedding methods: also embed the query rotated 180 degrees, keep the better similarity |
| suffix `@art` | compare only the artwork window (language independent), e.g. `hybrid-clip@art+rot` |

Indexes are cached in `<data>/cache` (delete the folder after changing a method's settings).

## 5. Card localisation and rectification (`rectify.py`)
Classical: two channels (grey + saturation) x 3 Canny settings x 2 dilation levels -> contours -> convex hull ->
4 corners -> checks (size, not touching the border, aspect, fill >= 0.85, edge support >= 0.6) -> score
area x support^2 -> duplicates / inner frames removed -> homography to 500x700.
Tuned ONLY on dev sets (seeds 1000/2000), never on the test queries. Outline correct (IoU >= 0.8), 200 dev queries:

| | s1 | s2 | s3 | s4 | s5 | all |
|---|---|---|---|---|---|---|
| version 1 (original) | 72.5% | 67.5% | 60% | 45% | 2.5% | 49.5% |
| version 2 (current) | 100% | 100% | 93% | 95% | 30% | 83.5% |

Multi-card dev scenes (20 per level), recall: no overlap 0.75 / 15% overlap 0.21 / 30% overlap 0.34
(version 1: 0.68 / 0.07 / 0.10). **Overlapping cards are the weakest case** (they merge into one contour).
The warp cannot tell up from down: measured, cards rotated -45..+135 degrees come out upright, others rotated 180.
`refine_quad` (used for the condition estimate) re-fits each side to the outermost colour edge.

## 6. Experiments (synthetic) and failure analysis
```
python -m cardrec.experiments --exp all     # main (11 methods x severity 1-5), ksweep, norectify, scenes
python -m cardrec.failure_analysis          # one cause per failure -> results/failures.csv, summary_failures.csv
python -m cardrec.report_figures            # results/figures/*.png for the report
```
- 200 queries per severity (fixed seed), JPEG q92; every query logged (results/main/<method>_s<sev>.csv) with
  strict + lenient rank, rectification IoU vs the true outline, and latency. Summaries include 95% bootstrap CIs
  (1000 resamples). A finished run is never repeated (delete its log to redo it).
- ksweep: hybrid k = 5, 10, 20, 50 at severity 3 (accuracy vs latency, compared with SIFT/CLIP/ResNet alone).
- norectify: severity 3 with the whole photo resized instead of rectified.
- scenes: 100 scenes of 2-6 cards per overlap level (0, 15, 30%): detection P/R/F1 (IoU >= 0.5) + end-to-end top-1.
- Failure causes (first rule that applies): localization (IoU < 0.5), rectification (IoU < 0.8), upside down
  (rotation-sensitive methods only), same artwork, lookalike (same name), heavy distortion (s4-5), other.
- **RESULTS: the full run (started 2026-10-21) is still running - see section 11 when it is done.**
- Latency note: the run shared the PC with other jobs; re-run `main` on a quiet PC for the report's latency
  numbers (accuracy is not affected).
- Synthetic cards have the scans' square grey corners (not rounded) in the main experiments; the condition and
  language tests use rounded corners (alpha) and phone-like resolution.

## 7. Printed language (en / es / ja)
Identification uses the artwork (`@art` methods) because it is the same in every language; the language is
classified separately with OCR (EasyOCR) + rules (`language.py`): kana share -> ja; else the language code next to
the set code ("PRE EN" / "PRE ES"); else typical-word counts.
Test data (`python -m cardrec.lang_data build|photos|eval [--split dev]`): 60 English, 60 Spanish (same card ids)
and 60 Japanese scans per split. Japanese sets with scans in TCGdex: SV8a (~PRE) and SV9 (~JTG); the English
equivalent was found WITHOUT the tested methods (same illustrator + Pokedex number, then nearest artwork pHash)
for 43/60 (test) and 34/60 (dev); all 43 test matches checked by eye. The others are used for language only.
- **Version 1 (rules fixed before seeing data), test split, low-resolution photos: accuracy 0.539 [0.461, 0.611].**
  Confusion (rows true en/es/ja; cols en/es/ja/unknown): en 16/1/6/37, es 0/28/10/22, ja 0/0/53/7.
  Causes: OCR on a 500x700 card in a ~530 px photo cannot read the small text (-> "unknown"), and the reader
  invents kanji on blurry Latin text (-> false "ja").
- Version 2 (kana only, code-box OCR, phone resolution, OCR at 750x1050), tuned on the dev split:
  **dev accuracy 0.928 [0.889, 0.961]**; no language was confused with another, all 13 errors are "unknown" (no
  usable text found). The code-box rule never fired (OCR misreads the tiny box, e.g. "JTG" -> "IGO").
  **Test-split result: running (2026-10-21).**
- Not available: Japanese scans for M2a (~ASC) and M5 (~PBL) in TCGdex.
- **Identification per language** (test split, severity 2, low-resolution photos; en 60 / es 60 / ja 43 queries;
  lenient top-1 with 95% CI; for ja lenient is the meaningful metric; strict in results/summary_language_id.csv):

| Method | en | es | ja |
|---|---|---|---|
| sift | 1.000 | 1.000 | 1.000 |
| sift@art | 1.000 | 1.000 | 1.000 |
| clip | 0.983 | 0.883 [0.80, 0.95] | 0.628 [0.49, 0.77] |
| clip@art | 0.933 | 1.000 | 0.767 [0.63, 0.88] |
| hybrid-clip+rot (k=20) | 0.983 | 0.950 | 0.930 [0.84, 1.00] |
| hybrid-clip@art+rot (k=20) | 1.000 | 1.000 | 0.884 [0.77, 0.98] |

  Reading: SIFT+RANSAC is already language independent (the artwork keypoints dominate the homography);
  CLIP depends on the text and the artwork crop helps it (es 0.88 -> 1.00, ja 0.63 -> 0.77); for the hybrid,
  @art shows no measurable gain (CIs overlap). Caveat: the ja ground truth was matched by artwork similarity, so
  Japanese cards whose artwork differs from the English version are under-represented.

## 8. Visual condition ESTIMATE (`condition.py`, NOT an official grade)
Compares the photo with the identified card's official scan: outline refined, warped to 750x1050 (~12 px/mm),
scan aligned by SIFT + RANSAC (sanity-checked), colours matched. Output: overall 0-1000; centering L/R and T/B;
4 corners + 4 edges with whitening / fill / fray sub-scores; surface defects (outside the corner zones) with type
and position. The app asks for a FRONT and a BACK photo and reports per physical corner/edge both sides (the back
is mirrored: its top-left corner is behind the front's top-right); overall = mean of the two sides.
Back side: put a clean back photo at `<data>/condition/back_reference.jpg`; without it only centering, corners and
edges are measured (border strip only) and the surface is skipped. Limits are listed in the output.
Validation (`python -m cardrec.condition_eval --n 80 --split test`): official scans damaged by known amounts,
rendered as phone-resolution photos (severity 1), tuned on the dev split only. FRONT only (no back scans exist).
**Test split, 80 cards (final code, 2026-10-21):**

| Measure | Result |
|---|---|
| reference alignment accepted | 76/80 |
| centering error (percentage points), median / 90th pct | L/R 0.58 / 2.33, T/B 0.70 / 2.33 |
| Spearman rho, true strength vs sub-score | corner cut vs fill -0.74, corner whitening +0.62, edge whitening +0.78, edge fray +0.41 |
| surface defects (match within 4% of card size) | recall 0.61 (68/111), correct type 59/111, precision 0.69 (72/105) |
| part score vs true damage (rho) | corners -0.53, edges -0.32, surface -0.86, centering -0.90; overall -0.46 |

Reading: centering is reliable; corner damage and whitening are detected; fray (outline damage < 0.6 mm) is weak.
History: an earlier version counted damaged corners also as surface "spots" (precision 0.34); the corner zones
are now excluded from the surface test (a logic fix found on a separate example card, not by tuning on this set).
The back side has NOT been validated (no back reference); it needs your clean back photo.

## 9. Real phone photos - protocol and tools (`real_photos.py`)
1. Photograph 70-100 cards of EACH set you own (all must be in the database; they are). Mix conditions on purpose:
   plain and patterned backgrounds, good and dim light, some glare, some in sleeves, a few rotated / upside down,
   and 20-30 photos with 2-6 cards (some touching or overlapping). For the condition estimate: card flat, plain
   contrasting background, no glare, card filling most of the photo; also one clean card BACK as reference.
2. Copy the photos to `<data>/real/photos/`, then `python -m cardrec.real_photos labels` -> `labels.csv` with the
   3 best guesses as a typing aid. Fill `true_ids` (e.g. `PBL 120` or `me5-120`; several cards: `;`) and the
   condition columns (side, language, background, lighting, glare, sleeve, blur, overlap). The guesses must be
   checked by you: the ground truth may not come from the method being tested.
3. `python -m cardrec.real_photos check`, then `python -m cardrec.real_photos eval` -> results/summary_real.csv;
   `python -m cardrec.failure_analysis` then also splits real failures by condition.
**Status: tools tested on synthetic stand-ins; NO real photos yet.**

## 10. CardDex interface
```
python -m cardrec.app.server                         # real database
python -m cardrec.app.server --method hybrid-clip+rot --port 5000
```
Easiest: double-click `run_app.bat` (project root); it always uses the venv. A plain `python` in a new terminal is
the system Python 3.12, which has no torch/transformers; since 2026-11-08 the server notices this at start-up and
restarts itself with `%USERPROFILE%\.venvs\cardproject\Scripts\python.exe` (same arguments), so CLIP/ResNet work
either way. `.vscode/settings.json` also makes VS Code use the venv for new terminals.

Upload a photo -> detect + straighten -> top-5 matches (click to correct) -> tags (set + code, number, name,
category, Trainer type / stage, Pokemon type, rarity, regulation mark, illustrator) -> printed language (OCR,
editable) -> optional condition estimate from a front and a back photo -> save with quantity, price, language
and condition. The condition report follows a grading-report layout: a drawn card diagram (corner brackets,
centering C values per side, overall score in the centre, numbered defect markers, FRONT/BACK toggle) with the
8 corner/edge panels (Total, Fray, Fill, Whitening for front and back), then the straightened photos (nothing
drawn on them) and the surface analysis per side. Card size (W x H) is NOT shown: it cannot be measured.
The collection panel has tag filters and search; clicking a tag filters by it.
- Prices: TCGdex first (has prices for all 4 sets), Pokemon TCG API as fallback (it has NO prices for ASC and
  PBL); mean of the TCGplayer market prices of the card's variants, else the Cardmarket average (EUR); cached 24 h.
- Tested with Flask's test client (analyze, tags, language, condition, save, filters' data); JavaScript syntax
  checked. **The screens themselves have not been checked in a browser since the tag/condition additions.**
- Speed: plain SIFT against 785 cards took ~5.5 s per photo in the app; use a hybrid method.

## 11. Status

| Part | Status |
|---|---|
| Database (785 cards), art groups, prices | done, verified (section 3, 10) |
| SIFT, ORB, pixel, ResNet, CLIP, hybrids, +rot, @art | all run on real cards (smoke tests); full comparison running |
| Rectification v2 | done; dev numbers in section 5 |
| Synthetic experiments + CIs, k sweep, ablation, scenes | code done and dry-run tested; **full run in progress** |
| Failure analysis + figures | code done; tested on partial logs |
| Real phone photos | tools ready; **photos not taken yet** |
| Tags (UI) | done; API tested; not checked visually in a browser |
| Language | v1 test 0.539; v2 dev 0.928; **v2 test running**; real photos needed |
| Condition estimate | done; front validated on synthetic damage (section 8); back not validated; real photos needed |
| Condition report layout (app) | done; checked with a headless-browser screenshot of the panel |
| Learned detector | optional; NOT started (overlapping cards would be the motivation) |

## 12. Layout
```
cardrec/
  config.py         settings (sets, sizes, CARDREC_DATA, art box)
  database.py       download cards + manifest            art_groups.py   same-artwork groups (pHash)
  rectify.py        find card(s), refine outline, warp    synth.py        synthetic scenes with ground truth
  make_queries.py   single-card synthetic queries          scenes.py       multi-card scenes + detection eval
  evaluate.py       metrics, CIs, per-query logs           experiments.py  all synthetic experiments
  failure_analysis.py  causes of failures                  report_figures.py  figures for the report
  real_photos.py    labelling + evaluation of real photos
  language.py       OCR language rules                     lang_data.py    language test data + eval
  condition.py      condition estimate                     condition_eval.py  synthetic validation
  methods/          base (interface), sift, embedding (+pixel), cnn, clip_method, hybrid, cache
  app/              CardDex: server.py, engine.py, db.py, prices.py, static/index.html
```

## 13. Credits and citations
Card images/metadata: Pokemon TCG API (pokemontcg.io); illustrators, prices, Spanish/Japanese scans: TCGdex
(tcgdex.net). Check both services' terms of use before publishing. Pokemon and related marks belong to their owners.
Libraries: OpenCV, NumPy, pandas, PyTorch, torchvision, Hugging Face Transformers, EasyOCR, Flask, matplotlib.
Cite in the report: SIFT (Lowe 2004), ORB (Rublee et al. 2011), RANSAC (Fischler & Bolles 1981),
ResNet (He et al. 2016), CLIP (Radford et al. 2021), pHash/DCT hashing, EasyOCR (CRAFT detector, Baek et al. 2019).
