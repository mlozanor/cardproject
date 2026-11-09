# Report status: CAP 5410 final report (2026-11-08)

Template: the original GIRS "Assignment Report Template". It's no longer in the project. A copy is in
`C:\Users\MarianaLozano\Downloads\Assignment Report Template`. The report in this folder is the filled-in copy.

## How the report works

- Every number in the text is a macro, like `\res{orb-s3-top1}`, written by `make_figures.py` from the CSV logs in
  `../results/`. A missing number shows up as a red **??key**, a missing run as a red **n/r** in tables.
- After new experiments or new cards in the app:
  1. `python report/app_evidence.py` (only if the app collection changed): this reads the app database read-only
  2. `python report/make_figures.py`: tables, figures, numbers
  3. In VS Code, open `report/main.tex` and press Ctrl+Alt+B (MiKTeX, pdflatex → bibtex → pdflatex ×2)
- Overleaf alternative: upload `report_overleaf.zip`. The compiler is pdfLaTeX and the main file is `main.tex`.
- Written sentences that interpret results don't update themselves. If results change, re-read Ch. 6–8.

## Complete

| Part | Source |
|---|---|
| Problem, motivation, related work, technical approach, dataset, methods (as implemented) | code, manifest |
| **Main comparison**: 11 configurations × 5 severities × 200 queries (Table 6.1, Fig. 6.2, App. table) | `results/main/` |
| **k-sweep** of both hybrids at S3 (Fig. 6.3) | `results/ksweep/` |
| **Rectification ablation** at S3 (Table 6.2) | `results/norectify/` |
| **Multi-card scenes**, 3 overlap levels × 4 methods (Table 6.3) | `results/scenes/` |
| Failure analysis of all methods (recomputed from the current logs) | `results/main/` + query ground truth |
| Cross-language identification, language classification, condition validation | `results/language*/`, `results/condition/` |
| **Application**: description, method guide, Japanese Battle Partners cards (132), screenshots, saved collection (7 cards: 5 English, 2 Japanese), re-identification of the 7 saved real photos with 7 methods (incl. pixel), detection and condition on real photos | `results/app/`, `<data>/manifest_ja.csv`, `app images/` |
| Abstract, introduction, conclusions, limitations, future work: rewritten as a final report | n/a |

## All experiments are done (finished 2026-11-08 18:06)

Every run of `python -m cardrec.experiments --exp all` exists: 55 main runs, 8 k-sweep runs, 11 no-rectify runs, 12 scene runs.
`results/failures.csv` was refreshed with `python -m cardrec.failure_analysis`. No table cell is "n/r" any more.

Findings from the last (SIFT) runs that the text now includes:
- ORB and SIFT are statistically tied at every severity (ORB ahead at S3, SIFT ahead at S4–S5).
- The CLIP hybrid (k=20) matches exhaustive SIFT at S3 (0.935 vs 0.930) in about 0.4 s instead of about 9 s.
- Without rectification, SIFT keeps 0.425 and ORB 0.165, while all global methods drop to 0.005–0.10.

## Remaining red TODOs (need YOU, not experiments)

| # | Where | What |
|---|---|---|
| 1 | Ch. 2.5 | Peer-reviewed work on trading-card recognition: search IEEE Xplore / ACM DL / Google Scholar, or delete the TODO and keep the sentence that no such work was found |
| 2 | Acknowledgments | Check the pokemontcg.io and tcgdex.net terms of use (or delete the TODO if not publishing) |
| 3 | Acknowledgments + Ch. 9 | AI-assistant disclosure per course policy; confirm the contribution statement |

## Decisions I made in the final version (so you can explain them in the viva)

- **Real photos:** the planned systematic set (70–100 photos per set) was not collected. Instead of a TODO, the report states this
  as a limitation and uses the 7 photos saved in the app as a small real-photo test (ground truth = the card you saved).
- **Latency:** not re-measured on an idle PC. It's stated as indicative (orders of magnitude) in Ch. 4 and the limitations.
- **Pixel baseline on real photos:** added to the app re-identification to test the claim that the synthetic benchmark favours
  pixel similarity. Result: only 3/7 on real photos against 0.90–0.97 on synthetic S1–S3.
- **Japanese card names** are printed as "English twin name (JP)", because pdfLaTeX can't typeset Japanese characters.
- **Prices:** USD (English cards) and EUR (Japanese cards) are reported separately, never added together.
- **Screenshots** are from 2026-11-01 (collection at that date). New screenshots can replace the files in `app images/`;
  crop boxes are in `report/app_evidence.py` (CROPS).

## Things in the app data you may want to fix (the report describes them as they are)

- Mega Chandelure ex is a Japanese print saved with language "en" (it was saved before the Japanese pre-selection existed).
- Goldeen, Toucannon and Thievul have identical back-side condition scores (the same back photo seems to have been reused).

## References to verify before submission (marked `VERIFY` in references.bib)

suzuki1985contours (pages), douglas1973peucker (journal name), nister2006vocabtree (CD-cover claim), radenovic2018revisiting,
razavian2014cnnfeatures, babenko2014neuralcodes, gordo2016deepretrieval, radenovic2019finetuning, cao2020delg (pages),
chandrasekhar2011smvs (authors/pages/DOI), venkatesan2000hashing (venue/pages), paszke2019pytorch, wolf2020transformers (pages).
