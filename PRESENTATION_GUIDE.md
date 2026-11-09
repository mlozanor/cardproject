# Presentation guide: what each file in `cardrec/` does

CAP 5410 final project (Mariana Lozano): Pokémon card recognition from phone photos.

This guide covers every file in `cardrec/` in the order the pipeline runs. For each file it gives one plain sentence
("what to say"), the most important functions, and how to run it. Numbers are the final ones from the report.

---

## 1. The big picture (say this first)

```
 phone photo
     │
     ▼
 rectify.py ── find the card outline, warp it to a flat 500×700 image
     │
     ▼
 methods/ ──── compare the flat card with the 785 official scans → ranked list of cards
     │
     ▼
 evaluate.py ─ was the top answer correct? (strict = exact card, lenient = same artwork)
     │
     ▼
 app/ ──────── the same pipeline in a web app: analyse, save to a collection, show prices
```

Around this core sit the files that **build the data** (database, art groups, Japanese cards), the files that
**create test photos** (synth, make_queries, scenes), the files that **run and analyse experiments**
(experiments, failure_analysis, report_figures), and two **extensions** (language, condition).

---

## 2. Settings and data loading

| File | What it does | Key parts |
|---|---|---|
| `config.py` | One place for every setting: the 4 English sets, the data folder (`CARDREC_DATA`), the canonical size 500×700, the artwork window (`ART_BOX`), the number of queries per severity, the extra Japanese manifest. | constants only |
| `manifest.py` | Reads the card list (`manifest.csv`, one row per card). `load_app_manifest()` adds the Japanese cards for the app only. | `load_manifest`, `load_app_manifest` |

**What to say:** "All settings live in `config.py`, so every experiment uses the same values."

---

## 3. Building the reference database

| File | What it does | Run |
|---|---|---|
| `database.py` | Downloads the official card images and metadata from the Pokémon TCG API, links each card to its TCGdex id (needed for prices), fills transparent corners, and writes `manifest.csv`. Retries when the API returns 500/502 errors. Result: **785 English cards** from 4 sets. | `python -m cardrec.database` |
| `art_groups.py` | Finds reprints with identical artwork. It computes a 64-bit **perceptual hash** (pHash: 32×32 grey → DCT → 8×8 low frequencies → 1 bit each) and groups cards whose hashes differ by ≤ 6 bits. This is the basis of the **lenient metric**. | `python -m cardrec.art_groups` |
| `database_ja.py` | Downloads the **132 Japanese cards of Battle Partners (SV9)** from TCGdex into a separate `manifest_ja.csv`. Links each one to its English twin (same illustrator + same Pokédex number + similar artwork hash): 128 linked, all checked by eye. Only the app uses these cards; the English benchmark is unchanged. | `python -m cardrec.database_ja` |

**What to say:** "Some cards are reprinted with exactly the same picture, and no image method can tell those
apart. So I group them with a perceptual hash and also report a lenient score."

---

## 4. Rectification (classical computer vision)

| File | What it does |
|---|---|
| `rectify.py` | Finds the card in the photo and warps it flat. Steps: downscale → blur → use **two channels** (grey and HSV saturation, because a silver border can have the same brightness as the table) → **Canny** edges with 3 threshold settings and 2 dilations (12 edge maps) → contours → **convex hull** → `approxPolyDP` to get 4 corners → quality checks (aspect ratio close to 5:7, edge support, size) → **homography** warp to 500×700. `rectify_all` returns every card for multi-card photos; `refine_quad` makes the corners more precise for the condition estimate; `polygon_iou` measures how good the outline is. |

**Key functions:** `find_card_quads`, `order_corners`, `warp_to_canonical`, `rectify`, `rectify_all`, `polygon_iou`.

**What to say:** "Rectification is essential. Without it, SIFT drops from 0.930 to 0.425 and the global methods
to almost zero. Under strong distortion, most errors come from not finding the outline."

---

## 5. Retrieval methods (`cardrec/methods/`)

Every method follows the same interface: `index(cards)` once, then `rank(query)` → scores for all cards.
This is what makes the comparison fair.

| File | What it does |
|---|---|
| `methods/base.py` | The common interface (`Retriever`) and `load_canonical` (loads a reference scan at 500×700). |
| `methods/__init__.py` | The **factory** `get_method("hybrid-clip@art+rot")`: turns a name into a method object. `@art` = compare only the artwork window (language independent); `+rot` = also try the query rotated 180°. Heavy libraries (torch) are imported only when they are needed. |
| `methods/embedding.py` | Base class for **global embeddings**: one L2-normalised vector per card, ranked by **cosine similarity**. Also has `TinyPixelRetriever`, the **pixel baseline** (a tiny 32×44 grey thumbnail used as the vector). |
| `methods/cnn.py` | **ResNet-50** pre-trained on ImageNet, frozen; the 2048-d average-pooled vector. |
| `methods/clip_method.py` | **CLIP ViT-B/32** image encoder (a Transformer), frozen; 512-d vector. Note: CLIP's preprocessing centre-crops to a square, so it does not see the top and bottom ~13% of the card. |
| `methods/sift.py` | **Local keypoints**: SIFT (float descriptors, L2 distance) or ORB (binary descriptors, Hamming distance), 800 features. Steps: match descriptors → **Lowe's ratio test** (0.75) → **RANSAC homography** → score = number of inliers. Very accurate, but it compares the photo with every card (about 9 s per photo). |
| `methods/hybrid.py` | **Hybrid**: CLIP (or ResNet) picks a shortlist of the top k = 20 cards, then SIFT + RANSAC re-ranks only those 20. Ties are broken by the embedding score. |
| `methods/cache.py` | Saves each built index (keypoints or embedding matrix) to disk, so the next run loads it in seconds instead of rebuilding it. The file name includes an md5 of the card list. |

**Results to quote (strict top-1, 200 synthetic photos per level):**

| Method | S1 | S3 | S5 | Time per photo |
|---|---|---|---|---|
| ORB | 0.995 | 0.970 | 0.085 | seconds |
| SIFT | 0.995 | 0.930 | 0.110 | ≈ 9 s |
| CLIP + SIFT hybrid | 0.995 | 0.935 | 0.020 | ≈ 0.4 s |
| CLIP | 0.970 | 0.845 | 0.020 | fast |
| ResNet-50 | 0.970 | 0.375 | 0.025 | fast |
| Pixel | 0.970 | 0.900 | 0.065 | fastest |

**What to say:** "Keypoints with geometric verification are the most accurate, but slow. The hybrid uses CLIP to
pick 20 candidates and SIFT to check them. It matches SIFT's accuracy about 20 times faster."

---

## 6. Test data with exact ground truth

| File | What it does | Run |
|---|---|---|
| `synth.py` | The **synthetic photo generator**. It places 1–N official scans on a random background with a random perspective, then adds lighting, glare, blur and noise. A **severity of 1–5** controls the strength. It returns the image plus each card's true corners and visible fraction, so the ground truth is exact. | used by the two below |
| `make_queries.py` | Makes the single-card test sets: 200 photos per severity, with a fixed seed, plus `queries.csv` (true id and true corners). | `python -m cardrec.make_queries --n 200 --severity 3` |
| `scenes.py` | **Multi-card scenes** with 0 / 15 / 30% overlap. It evaluates detection (a card counts as found if its IoU with a true card is ≥ 0.5; precision, recall, F1) and end-to-end identification. Recall falls from 0.706 to 0.330 to 0.291 as the overlap grows. | `python -m cardrec.scenes make/eval` |
| `real_photos.py` | Helper for real phone photos: creates a labels sheet, checks it, and evaluates methods on it. (The planned systematic real-photo set was not collected. The report uses the 7 photos saved in the app instead.) | `python -m cardrec.real_photos labels/check/eval` |
| `demo_offline.py` | **Smoke test** with no internet and no data: draws fake cards and runs the whole pipeline, to check the setup. | `python -m cardrec.demo_offline` |

**What to say:** "Because I generate the photos myself, I know the exact card and the exact corners. That lets me
measure the rectification and the identification separately."

---

## 7. Evaluation and experiments

| File | What it does | Run |
|---|---|---|
| `evaluate.py` | Runs one method over one query set and **logs every query** (true id, top-5, rectification IoU, time) to a CSV. Computes strict and lenient top-1 and top-5 with **95% bootstrap confidence intervals**. | `python -m cardrec.evaluate --method sift --queries ...` |
| `experiments.py` | Runs all the report's experiments and writes one summary table per experiment: `main` (11 configurations × 5 severities), `ksweep` (hybrid shortlist k = 5/10/20/50), `norectify` (ablation without rectification), `scenes` (overlap). | `python -m cardrec.experiments --exp all` |
| `failure_analysis.py` | Gives every wrong answer **one cause** using fixed rules, checked in order: localization (IoU < 0.5) → rectification (IoU < 0.8) → upside down → same artwork (reprint) → lookalike → heavy distortion → other. Then counts the causes per method. | `python -m cardrec.failure_analysis` |
| `report_figures.py` | Draws PNG figures from the result tables using a colour-blind-safe palette and markers, so they also read in grayscale. (The LaTeX report uses `report/make_figures.py`, which works the same way.) | `python -m cardrec.report_figures` |

**What to say:** "Every query is logged, so every number in the report can be traced back to a CSV file, and the
confidence intervals show which differences are real."

---

## 8. Extensions

| File | What it does | Run |
|---|---|---|
| `language.py` | **Printed-language classifier** (English / Spanish / Japanese). It reads the card text with **EasyOCR**, then applies simple rules: kana characters mean Japanese; otherwise count English and Spanish keywords. Identification itself uses the artwork window (`@art`), which is the same in every language. | used by the app and lang_data |
| `lang_data.py` | Builds the language test set from official Spanish and Japanese scans (dev split for tuning, test split for the reported numbers), makes phone-resolution photos of them, and evaluates identification and language accuracy. Result: SIFT is nearly language-independent; CLIP alone drops on Japanese cards. | `python -m cardrec.lang_data build/photos/eval` |
| `condition.py` | **Visual condition estimate** (not an official grade). It aligns the official scan to the photo (SIFT + RANSAC, colour matching), then measures **centering** (border widths), **corners and edges** (difference from the reference in thin zones) and **surface** defects. These are combined into a 0–10 score and a damage map. | used by the app |
| `condition_eval.py` | Validates the condition estimate on **synthetically damaged** cards with a known damage strength. It uses the **Spearman** rank correlation between the true damage and the score. It works on synthetic damage but not yet on real sleeved cards. | `python -m cardrec.condition_eval --n 80 --split test` |

---

## 9. The application (`cardrec/app/`)

| File | What it does |
|---|---|
| `app/server.py` | The **Flask** web server (CardDex) that runs on your PC. Routes: analyse a photo, save a card, list or delete the collection, look up prices. `relaunch_in_venv()` restarts it with the project's Python if torch or transformers are missing. Run it with `run_app.bat` or `python -m cardrec.app.server`. |
| `app/engine.py` | The **brain of the app**: loads the English and Japanese cards and the chosen method (using the cache), then for each photo: rectify → rank all cards → top-N candidates, plus tags (set, rarity, language, "English version"). `--demo` mode uses fake cards. |
| `app/db.py` | Small **SQLite** database (`collection.db`): the saved cards (photo, chosen card, language, condition) and a price cache. |
| `app/prices.py` | Fetches **market prices** on demand and caches them for 24 h: TCGdex (TCGplayer USD for English cards, Cardmarket EUR for Japanese cards), with the Pokémon TCG API as a fallback. |
| `app/static/index.html` | The **user interface** (one page): upload a photo, choose a method (with a short description of each), see the rectified card and candidates, choose the language, save to the collection, see the totals. |

**App results to quote:** 7 real cards saved (5 English, 2 Japanese). On their real photos, SIFT, ORB, CLIP and the
CLIP hybrid were correct 7/7, ResNet 5/7, and the pixel baseline only **3/7** (against 0.90–0.97 on synthetic
S1–S3). This shows the synthetic benchmark is easier than real photos.

---

## 10. Suggested presentation order (≈ 10–12 min)

1. **Problem** (1 min): many versions and reprints of the same Pokémon; phone photos add perspective, glare and blur.
2. **Pipeline diagram** (1 min): section 1 above.
3. **Data** (1 min): `database.py`, `art_groups.py` → strict vs lenient.
4. **Rectification** (1.5 min): `rectify.py`, show one before/after example.
5. **Methods** (2 min): `methods/`. Classical keypoints vs deep embeddings vs the hybrid.
6. **How I tested** (1 min): `synth.py`, severities, exact ground truth, bootstrap CIs.
7. **Results** (2 min): main table, k-sweep, no-rectification ablation, scenes, failure causes.
8. **App demo** (2 min): `run_app.bat`, analyse a card, Japanese card, collection and prices.
9. **Conclusions and limitations** (1 min): hybrid = best compromise; detection is the bottleneck; few real photos.

## 11. Likely questions and short answers

- **Why does the pixel baseline beat ResNet?** The synthetic photos are made from the same scans, so pixels match
  almost perfectly. On real photos the pixel baseline fails (3/7).
- **Why is CLIP better than ResNet?** CLIP was trained on 400M image–text pairs and is more robust to changes in
  appearance. ResNet was trained to classify objects, so cards look alike to it.
- **Why RANSAC?** The ratio test leaves wrong matches; RANSAC keeps only the matches consistent with one plane
  (the card), and their number is a reliable score.
- **Why the hybrid?** SIFT is accurate but compares the photo with all 785 cards. CLIP narrows them to 20 in
  milliseconds, giving the same accuracy (0.935 vs 0.930) in about 0.4 s instead of about 9 s.
- **What does +rot do?** A rectified card can be upside down. +rot also tries the 180° rotation. It fixes the
  upside-down errors but adds other errors, so the total does not improve.
- **Why does everything fail at S5?** The outline is often not found (localization), so the methods never see
  the card.
- **What is the lenient metric?** A prediction counts as correct if it has the same artwork (art group) as the true
  card. Reprints with identical art cannot be separated from the image alone.
