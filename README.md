# Pokemon Card Detection and Recognition (CAP 5410 Final Project)

Compares classical (SIFT/ORB + RANSAC), CNN (ResNet-50) and transformer (CLIP) retrieval for identifying
Pokemon cards in photos, plus a hybrid (embedding shortlist re-ranked by SIFT + RANSAC).
Extras: card tags, printed-language recognition (en/es/ja), a visual condition estimate, and **CardDex**, a local
web app to catalogue a collection (English cards + the Japanese set Battle Partners).
Author: Mariana Lozano. Type B experimental comparison. **Final version: 2026-11-09.**

**Rule for this README: every number below was measured (logs in `results/`); nothing is estimated.**

| Where | What |
|---|---|
| `cardrec/` | all the code (section 12); a file-by-file explanation is in `PRESENTATION_GUIDE.md` |
| `results/` | every per-query log and summary table of the experiments |
| `run_app.bat` | double-click to start the CardDex app |
| `app images/` | screenshots of the app |

## 0. Final results in one table

Strict top-1 accuracy (exact card) on 200 synthetic photos per severity level (S1 mild … S5 extreme), 785 English
cards in the database. Median time per photo at S3 (indicative: the PC was running other jobs).

| Method | S1 | S2 | S3 | S4 | S5 | time per photo (S3) |
|---|---|---|---|---|---|---|
| ORB + RANSAC | 0.995 | 0.975 | **0.970** | 0.725 | 0.085 | ~7.0 s |
| SIFT + RANSAC | 0.995 | 0.970 | 0.930 | **0.745** | **0.110** | ~8.9 s |
| **Hybrid CLIP → SIFT (k=20)** | 0.995 | 0.965 | 0.935 | 0.530 | 0.020 | **~0.37 s** |
| Hybrid ResNet → SIFT (k=20) | 0.995 | 0.940 | 0.755 | 0.370 | 0.050 | ~0.39 s |
| CLIP ViT-B/32 | 0.970 | 0.955 | 0.845 | 0.250 | 0.020 | ~0.17 s |
| ResNet-50 | 0.970 | 0.780 | 0.375 | 0.120 | 0.025 | ~0.12 s |
| Pixel thumbnail (baseline) | 0.970 | 0.930 | 0.900 | 0.600 | 0.065 | ~0.09 s |

Main conclusions:
1. **Keypoints + RANSAC are the most accurate**: ORB and SIFT are statistically tied (overlapping 95% CIs).
   At S1–S2 their only errors are reprints with identical artwork.
2. **The CLIP hybrid is the best practical choice**: it matches exhaustive SIFT at S3 (0.935 vs 0.930) about 24×
   faster. With k=50 it reaches 0.950.
3. **CLIP is much better than ResNet-50**, but neither is enough alone. On synthetic photos even the pixel
   thumbnail beats them at S3–S4, because the queries are rendered from the scans themselves.
4. **Rectification is essential and detection is the bottleneck.** Without it, SIFT drops to 0.425, ORB to 0.165
   and every global method to 0.005–0.10. At S5 the outline is correct in only 27% of the photos. With
   overlapping cards, detection recall falls to 0.29.
5. **+rot** (also trying the card upside down) removes the upside-down errors but does not reduce the total.
6. **Real photos** (the 7 cards saved in the app): SIFT, ORB, CLIP and the CLIP hybrid got 7/7, ResNet 5/7 and
   pixel only 3/7. The synthetic benchmark favours pixel similarity.

## 1. Setup (Windows)
The project folder is inside OneDrive, so the venv and the big data folder live OUTSIDE it (otherwise OneDrive
would try to sync tens of thousands of files).
```
py -3.12 -m venv %USERPROFILE%\.venvs\cardproject
%USERPROFILE%\.venvs\cardproject\Scripts\activate
pip install -r requirements.txt
setx CARDREC_DATA C:\Users\MarianaLozano\cardrec_data     (once; open a NEW terminal afterwards)
```
- `CARDREC_DATA` = where card images, queries, scenes, caches, uploads and the collection are stored.
  If it is not set, `data/` inside the project is used. `results/` (CSV logs, figures) stays in the project.
- VS Code: `.vscode/settings.json` already selects the venv (`%USERPROFILE%\.venvs\cardproject\Scripts\python.exe`)
  for Python and new terminals. Restart VS Code after `setx`.
- Deep learning (installed and tested 2026-10-21, GPU: RTX 3050 Laptop 4 GB):
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install transformers
  pip install easyocr --no-deps          (--no-deps: it would install a second, conflicting OpenCV)
  pip install scipy scikit-image python-bidi pyclipper shapely ninja
  ```
  Tested versions: Python 3.12.10, torch 2.11.0+cu128, torchvision 0.26.0, transformers 5.20.2, easyocr 1.7.2
  (core packages pinned in requirements.txt). `pip check` warns that easyocr wants opencv-python-headless; this
  is expected.
- **Use the venv Python.** The system Python 3.12 has no torch/transformers. The app restarts itself in the venv
  (section 10); for every other command, activate the venv first.

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

- Counts include secret rares (numbers above the printed size): 629 Pokemon, 150 Trainer, 6 Energy.
  The Pitch Black count matches pokecottage.com's list (120, number 120 = Mega Darkrai ex).
- **Unique artworks: 761.** 24 pairs share identical artwork (Ascended Heroes reprints 9 PRE and 15 JTG cards,
  e.g. N's Reshiram sv9-116 = me2pt5-154). They were found with a 64-bit perceptual hash (pHash) of the whole
  card and of the artwork box: both are <= 6 bits different, while the next closest unrelated pair is >= 12.
  All 24 were checked by eye.
- **Strict metric** = exact card id. **Lenient metric** = same art_group (a reprint counts as correct).
  163 names appear on more than one card (391 cards, e.g. 8 versions of "Black Belt's Training"); these are the
  lookalikes.
- Sources:
  - Pokemon TCG API (pokemontcg.io): list, metadata, images (733x1024 PNG, transparent corners filled grey).
    In Oct 2026 it answered 500/502 to about half the requests, so every request is retried.
  - TCGdex (tcgdex.net): illustrators missing in the first API (all 180 PRE cards + 7 PBL; 6 special Energy cards
    have none in either source), each card's TCGdex id (for prices), Spanish/Japanese scans (section 7).
- Manifest columns: id, name, set_id, set_name, set_code, number, printed_total, supertype, subtypes, types,
  rarity, regulation_mark, artist, artist_source, language, image_path, art_group, tcgdex_id.

### 3b. Japanese cards for the app (built 2026-11-08)
```
python -m cardrec.database_ja        # Battle Partners (SV9, Japanese) from TCGdex -> manifest_ja.csv, cards_ja/
```
- 132 cards (100 + 32 secret rares). 128 are linked to the English card with the same artwork (126 Journey
  Together, 2 Ascended Heroes reprints) using the lang_data rule (illustrator + Pokedex number + artwork
  pHash <= 14). All 128 pairs were checked by eye on `<data>/ja_links.jpg`. Unlinked: the 2 Spike Energy cards
  (no illustrator), N's Zorua 108 and Dudunsparce ex 120.
- Ids are `ja-<TCGdex id>` (e.g. `ja-SV9-061`). Linked cards share the English card's `art_group`, and the app
  shows an "English version" tag. Secret rares have no rarity in TCGdex.
- **Used only by the app** (`config.APP_EXTRA_MANIFESTS` -> `manifest.load_app_manifest`): the app searches
  785 + 132 = 917 cards. The experiments (the benchmark) use the 785 English cards.
- Prices of Japanese cards: the TCGdex Japanese catalogue gives the Cardmarket average in EUR (no TCGplayer prices).
  The app's collection total adds USD and EUR prices as they are (no conversion).
- To add another Japanese set, add it to `JA_SETS` in `cardrec/database_ja.py` (TCGdex id: English name,
  English counterpart) and re-run. Don't delete anything: the app builds new index caches automatically.

## 4. Methods (all behind the same interface in `cardrec/methods`)
Every method has `build_index(paths, ids)` and `query(card_image) -> full ranking`, so all of them are evaluated
the same way.

| Name | What it does |
|---|---|
| `pixel` | 32x44 grey thumbnail, cosine similarity (sanity baseline) |
| `orb`, `sift` | keypoints + ratio test (0.75) + RANSAC homography, score = inlier count, against every card (800 features) |
| `resnet` | frozen ImageNet ResNet-50, 2048-d global embedding, cosine similarity |
| `clip` | frozen CLIP ViT-B/32 image embedding (512-d). CLIP's preprocessing centre-crops the card to a square, so the top and bottom ~13% are not seen |
| `hybrid-<emb>` | top-k shortlist from the embedding, re-ranked by SIFT inliers (`--k`, default 20) |
| suffix `+rot` | embedding methods: also embed the query rotated 180 degrees and keep the better similarity |
| suffix `@art` | compare only the artwork window (language independent), e.g. `hybrid-clip@art+rot` |

Indexes are cached in `<data>/cache`; delete that folder after changing a method's settings.

## 5. Card localisation and rectification (`rectify.py`)
Classical pipeline:
1. Two channels (grey + saturation) × 3 Canny settings × 2 dilation levels → contours → convex hull → 4 corners.
2. Checks: size, not touching the border, aspect ratio, fill >= 0.85, edge support >= 0.6.
3. Score = area × support²; duplicates and inner frames are removed.
4. Homography to 500x700.

It was tuned ONLY on dev sets (seeds 1000/2000), never on the test queries.

Outline found / outline correct (IoU >= 0.8) **on the 200 test queries per level** (final run):

| | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| card found | 100% | 100% | 100% | 91% | 36% |
| outline correct (IoU >= 0.8) | 98.5% | 99% | 98% | 89% | 27% |

On the dev queries, version 1 of the detector had 49.5% correct outlines overall and version 2 (current) 83.5%.
- **Overlapping cards are the weakest case**, because touching cards merge into one contour.
- The warp cannot tell up from down: cards rotated -45..+135 degrees come out upright, all others rotated 180.
- `refine_quad` (used for the condition estimate) re-fits each side to the outermost colour edge.

## 6. Experiments (synthetic) and failure analysis: DONE (finished 2026-11-08)
```
python -m cardrec.experiments --exp all     # main (11 configs x severity 1-5), ksweep, norectify, scenes
python -m cardrec.failure_analysis          # one cause per failure -> results/failures.csv, summary_failures.csv
python -m cardrec.report_figures            # results/figures/*.png
```
- 200 queries per severity (fixed seed), JPEG q92. Every query is logged (`results/main/<method>_s<sev>.csv`)
  with strict and lenient rank, rectification IoU against the true outline, and latency. Summaries include 95%
  bootstrap CIs (1000 resamples). A finished run is never repeated; delete its log to redo it.
- Finished runs: 55 main, 8 k-sweep, 11 no-rectify, 12 scenes.
- Latency note: the PC was running other jobs, so the times are indicative (orders of magnitude). Accuracy is not
  affected.
- Synthetic cards in the main experiments keep the scans' square grey corners (not rounded). The condition and
  language tests use rounded corners (alpha) and phone-like resolution.

**Main comparison**: the table in section 0. Full table with CIs, top-5 and lenient scores:
`results/summary_main.csv`. It also has the `+rot` variants.

**k-sweep** (hybrid shortlist size, S3, strict top-1 / ms):

| k | 5 | 10 | 20 | 50 |
|---|---|---|---|---|
| hybrid CLIP | 0.910 / 234 | 0.925 / 279 | 0.935 / 414 | 0.950 / 833 |
| hybrid ResNet | 0.565 / 237 | 0.660 / 280 | 0.755 / 392 | 0.880 / 696 |

**No rectification** (S3, whole photo resized): SIFT 0.425, ORB 0.165, hybrids 0.075–0.100, CLIP 0.020,
ResNet / pixel 0.005.

**Multi-card scenes** (100 scenes of 2–6 cards, severity 2):

| Overlap | Detection precision | Detection recall | End-to-end top-1, hybrid CLIP | Identified if detected |
|---|---|---|---|---|
| 0% | 0.911 | 0.706 | 0.643 | 0.911 |
| 15% | 0.759 | 0.330 | 0.285 | 0.865 |
| 30% | 0.721 | 0.291 | 0.247 | 0.849 |

**Failure causes** (first rule that applies): localization (IoU < 0.5), rectification (IoU < 0.8), upside down
(rotation-sensitive methods only), same artwork, lookalike (same name), heavy distortion (S4–5), other.
- **Localization** is the largest shared cause: about 160 failures per method, almost all at S4–S5.
- The global embeddings also fail on well-rectified cards ("other": ResNet 160, CLIP 18, SIFT 9, ORB 0).
- Upside-down errors are 31–43 per rotation-sensitive method, and 0 with `+rot`.
- In scenes, most misses are cards that were not detected or were covered by other cards.

## 7. Printed language (en / es / ja)
Identification uses the artwork (`@art` methods), because the artwork is the same in every language. The language
is classified separately with OCR (EasyOCR) + rules (`language.py`): a share of kana characters → ja; otherwise
the language code next to the set code ("PRE EN" / "PRE ES"); otherwise typical-word counts.

Test data (`python -m cardrec.lang_data build|photos|eval [--split dev]`): 60 English, 60 Spanish (same card ids)
and 60 Japanese scans per split. Japanese sets with scans in TCGdex: SV8a (~PRE) and SV9 (~JTG). The English
equivalent was found WITHOUT the tested methods (same illustrator + Pokedex number, then nearest artwork pHash)
for 43/60 (test) and 34/60 (dev). All 43 test matches were checked by eye.

| Language classifier | Split | Accuracy [95% CI] | Notes |
|---|---|---|---|
| version 1 (rules fixed before seeing data), low resolution | test | 0.539 [0.461, 0.611] | 66 "unknown", 16 false "ja" |
| version 2 (kana rule, phone resolution, OCR at 750x1050) | dev | 0.928 [0.889, 0.961] | tuned here |
| **version 2** | **test** | **0.911 [0.872, 0.950]** | 14 "unknown", only 2 wrong languages |

**Identification per language** (test split, lenient top-1; ja = 43 queries):

| Method | en (low / phone res.) | es (low / phone) | ja (low / phone) |
|---|---|---|---|
| sift | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.977 |
| sift@art | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| clip | 0.983 / 1.000 | 0.883 / 0.967 | 0.628 / 0.767 |
| clip@art | 0.933 / 0.967 | 1.000 / 1.000 | 0.767 / 0.860 |
| hybrid-clip+rot (k=20) | 0.983 / 1.000 | 0.950 / 0.950 | 0.930 / 1.000 |
| hybrid-clip@art+rot (k=20) | 1.000 / 1.000 | 1.000 / 1.000 | 0.884 / 0.977 |

Reading the table:
- SIFT+RANSAC is already language independent, because the artwork keypoints dominate the homography.
- CLIP depends on the text, and the artwork crop helps it.
- The CLIP hybrid recovers almost everything in ~0.55 s, against ~34 s for exhaustive SIFT at phone resolution.

Caveat: the ja ground truth was matched by artwork similarity, so Japanese cards whose artwork differs from the
English version are under-represented.

## 8. Visual condition ESTIMATE (`condition.py`, NOT an official grade)
The estimate compares the photo with the identified card's official scan:
- The outline is refined and the card is warped to 750x1050 (~12 px/mm).
- The scan is aligned by SIFT + RANSAC (sanity-checked) and its colours are matched to the photo.

Output: an overall score 0-1000; centering L/R and T/B; 4 corners + 4 edges with whitening / fill / fray
sub-scores; surface defects (outside the corner zones) with their type and position.

The app asks for a FRONT and a BACK photo and reports each physical corner and edge for both sides. The back is
mirrored (its top-left corner is behind the front's top-right). The overall score is the mean of the two sides.

Back side: put a clean back photo at `<data>/condition/back_reference.jpg`. Without it, only centering, corners and
edges are measured (border strip only) and the surface is skipped.

Validation (`python -m cardrec.condition_eval --n 80 --split test`): official scans damaged by known amounts,
rendered as phone-resolution photos (severity 1), tuned on the dev split only. FRONT only (no back scans exist).
**Test split, 80 cards:**

| Measure | Result |
|---|---|
| reference alignment accepted | 76/80 |
| centering error (percentage points), median / 90th pct | L/R 0.58 / 2.33, T/B 0.70 / 2.33 |
| Spearman rho, true strength vs sub-score | corner cut vs fill -0.74, corner whitening +0.62, edge whitening +0.78, edge fray +0.41 |
| surface defects (match within 4% of card size) | recall 0.61 (68/111), correct type 59/111, precision 0.69 (72/105) |
| part score vs true damage (rho) | corners -0.53, edges -0.32, surface -0.86, centering -0.90; overall -0.46 |

- **On synthetic damage:** centering is reliable, corner damage and whitening are detected, and fray (outline
  damage < 0.6 mm) is weak.
- **On real photos** (the 7 sleeved cards in the app), the overall scores were only 132–320 / 1000, which is
  implausibly low for cards in good condition. Sleeves, reflections and the missing back reference are the likely
  causes, so the estimate is **not yet reliable on real photos**.

## 9. Real phone photos
- **Planned systematic set** (70–100 photos per set, `real_photos.py`): **not collected**. The tools are ready and
  tested on synthetic stand-ins:
  1. Copy the photos to `<data>/real/photos/`.
  2. `python -m cardrec.real_photos labels` creates `labels.csv` with the 3 best guesses as a typing aid. Fill in
     `true_ids` and the condition columns yourself; the ground truth must not come from the method being tested.
  3. Run `python -m cardrec.real_photos check`, then `python -m cardrec.real_photos eval`.
- **Real-photo evidence**: the 7 cards saved in CardDex (5 English Pitch Black cards: Mega
  Chandelure ex, Morpeko, Goldeen, Toucannon, Thievul; 2 Japanese Battle Partners cards: Maractus, Swinub).
  Ground truth = the card that was confirmed when saving. Logs: `results/app/`.

| Method on the 7 real photos | Correct (strict) | Median time |
|---|---|---|
| sift, orb, clip, hybrid-clip+rot | 7/7 | sift/orb ~7 s, clip 0.27 s, hybrid ~1.0 s |
| hybrid-clip@art+rot | 5/7 strict, 7/7 lenient (Japanese → its English twin) | 0.74 s |
| resnet | 5/7 | 0.28 s |
| pixel | **3/7** | 0.14 s |

## 10. CardDex app
```
run_app.bat                                          # easiest: double-click (always uses the venv)
python -m cardrec.app.server                         # real database, opens on http://127.0.0.1:5000
python -m cardrec.app.server --method hybrid-clip+rot --port 5000
```
If it is started with the system Python (no torch/transformers), the server notices this and restarts itself with
`%USERPROFILE%\.venvs\cardproject\Scripts\python.exe` (same arguments), so CLIP and ResNet always work.

What it does:
1. **Analyse a card**: upload a photo → detect + straighten → top-5 matches (click to correct). A method selector
   shows a short description of each method, and "What do the methods mean?" lists all of them.
2. **Tags**: set + code, number, name, category, Trainer type / stage, Pokemon type, rarity, regulation mark,
   illustrator, language, and "English version" for Japanese cards.
3. **Printed language**: OCR, editable. Japanese reference cards are pre-selected as Japanese.
4. **Condition estimate** (optional) from a front and a back photo. It is shown in a grading-report layout: a card
   diagram with centering values, the overall score and numbered defects, 8 corner/edge panels, the straightened
   photos and the surface analysis.
5. **Save** with quantity, price, language and condition. The collection panel has tag filters, search and totals.

- Prices: TCGdex first (TCGplayer market price in USD for English cards, Cardmarket average in EUR for Japanese
  cards), with the Pokemon TCG API as a fallback for English cards. Cached for 24 h.
- Data: `<data>/collection.db` (SQLite) and `<data>/uploads/`.
- Recommended method: `hybrid-clip+rot`. It was correct on all 7 real photos in ~1 s; exhaustive SIFT takes several
  seconds per photo against 917 cards.

## 11. Status (final)

| Part | Status |
|---|---|
| Database (785 English cards), art groups, prices | done, verified |
| Japanese Battle Partners (132 cards, 128 linked) | done, verified by eye; used by the app |
| SIFT, ORB, pixel, ResNet, CLIP, hybrids, +rot, @art | done |
| Rectification v2 | done (section 5) |
| Synthetic experiments + CIs, k-sweep, ablation, scenes | **done** (section 6) |
| Failure analysis + figures | **done** |
| Language: identification and classifier | **done** (v2 test accuracy 0.911) |
| Condition estimate | done; validated on synthetic damage; not reliable yet on real sleeved cards |
| CardDex app (analyse, tags, language, condition, collection, prices, method guide, Japanese cards) | done; used to catalogue 7 real cards |
| Real photos | 7 app photos evaluated; systematic set NOT collected |
| Learned detector | not done (future work: overlapping cards and sleeves) |

## 12. Layout
```
cardproject/
  README.md             this file
  PRESENTATION_GUIDE.md what each file does, presentation order, likely questions
  requirements.txt      pinned core packages
  run_app.bat           start CardDex with the venv
  .vscode/              VS Code settings (uses the venv Python)
  app images/           screenshots of the CardDex app
  data/                 fallback data folder (only used if CARDREC_DATA is not set; demo data)
  results/              all experiment logs, summary tables and figures
  cardrec/              all the code
    config.py           settings (sets, sizes, CARDREC_DATA, art box)
    manifest.py         load card lists (+ Japanese for the app)
    database.py         download English cards + manifest
    database_ja.py      Japanese Battle Partners cards (app only)
    art_groups.py       same-artwork groups (pHash)
    rectify.py          find card(s), refine outline, warp
    synth.py            synthetic scenes with ground truth
    make_queries.py     single-card synthetic queries
    scenes.py           multi-card scenes + detection eval
    evaluate.py         metrics, CIs, per-query logs
    experiments.py      all synthetic experiments
    failure_analysis.py causes of failures
    report_figures.py   figures (PNG)
    real_photos.py      labelling + evaluation of real photos
    demo_offline.py     smoke test without data
    language.py         OCR language rules
    lang_data.py        language test data + eval
    condition.py        condition estimate
    condition_eval.py   synthetic validation of the condition estimate
    methods/            the retrieval methods (pixel, SIFT/ORB, ResNet, CLIP, hybrid) and the index cache
    app/                the CardDex web app (server, analysis engine, collection database, prices, interface)
```

## 13. Credits and citations
- Card images and metadata: Pokemon TCG API (pokemontcg.io).
- Illustrators, prices, Spanish and Japanese scans: TCGdex (tcgdex.net).
- Check both services' terms of use before publishing. Pokemon and related marks belong to their owners.
- Libraries: OpenCV, NumPy, pandas, PyTorch, torchvision, Hugging Face Transformers, EasyOCR, Flask, matplotlib.
- Main references: SIFT (Lowe 2004), ORB (Rublee et al. 2011), RANSAC
  (Fischler & Bolles 1981), ResNet (He et al. 2016), CLIP (Radford et al. 2021), perceptual hashing, EasyOCR /
  CRAFT (Baek et al. 2019).
