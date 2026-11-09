"""Regenerate every table, figure and number of the report from the CSV logs in results/.

    (venv active, from the project root)
    python report/make_figures.py

Writes
  report/generated/numbers.tex   one macro per number used in the text:  \\res{orb-s5-top1} -> 0.085
  report/generated/*.tex         tables and figure environments, \\input by the chapters
  report/figures/*.pdf|png       plots (vector PDF) and qualitative examples
Nothing in results/ is modified.

Rules
  * Every number comes from a per-query log (results/main, results/language, results/condition, ...).
    Summaries are recomputed with the project's own functions (cardrec.evaluate.summarize: same
    metrics and the same 95% bootstrap CI as the experiments), so they match the experiment printouts.
  * A run that has no log yet is printed as a red "n/r" (not run) in tables, and a figure whose data
    is missing is replaced by a red \\todo box saying which command produces it. After running new
    experiments, run this script again and recompile: the report updates itself.
  * Failure causes are recomputed from the CURRENT logs with cardrec.failure_analysis (the rules of
    the project), because results/failures.csv may be older than the newest logs. This needs the data
    folder (CARDREC_DATA) for the query ground truth; without it, results/failures.csv is used.
  * Qualitative figures (example photos) need the data folder too; they are skipped without it.
  * The application section uses results/app/*.csv, written by report/app_evidence.py.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
sys.path.insert(0, str(ROOT))

from cardrec import config  # noqa: E402
from cardrec.condition_eval import CORNER_PTS, spearman  # noqa: E402
from cardrec.evaluate import bootstrap_ci, summarize  # noqa: E402
from cardrec.experiments import (HYBRID_K, KS, KSWEEP_SEVERITY, MAIN_METHODS, N_QUERIES,  # noqa: E402
                                 SCENE_METHODS, SCENE_OVERLAPS, SEVERITIES)
from cardrec.synth import SEVERITY  # noqa: E402

RES = config.RESULTS_DIR
GEN, FIG = REPORT / "generated", REPORT / "figures"
DTYPES = {"query_id": str, "true_id": str, "pred_id": str}

# same palette / markers as cardrec/report_figures.py (colour-blind checked, works in grayscale)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
W = 6.0                                                     # inches, report text width is 6.5
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans", "pdf.fonttype": 42})

LABEL = {"pixel": "Pixel thumbnail", "orb": "ORB + RANSAC", "sift": "SIFT + RANSAC",
         "resnet": "ResNet-50", "clip": "CLIP ViT-B/32",
         "hybrid_resnet_k20": "Hybrid ResNet$\\to$SIFT", "hybrid_clip_k20": "Hybrid CLIP$\\to$SIFT",
         "resnet+rot": "ResNet-50 +rot", "clip+rot": "CLIP +rot",
         "hybrid_resnet+rot_k20": "Hybrid ResNet$\\to$SIFT +rot", "hybrid_clip+rot_k20": "Hybrid CLIP$\\to$SIFT +rot",
         "sift@art": "SIFT + RANSAC @art", "clip@art": "CLIP @art",
         "hybrid_clip@art+rot_k20": "Hybrid CLIP$\\to$SIFT @art +rot"}
PLAIN = {k: v.replace("$\\to$", "→") for k, v in LABEL.items()}       # for matplotlib text

numbers = {}


def put(key, value, fmt="{:.3f}"):
    numbers[key] = fmt.format(value) if isinstance(value, (float, np.floating)) else str(value)


def run_name(spec, k=HYBRID_K):
    """Method spec as in experiments.py ('hybrid-clip+rot') -> log name used by the run ('hybrid_clip+rot_k20')."""
    return f"hybrid_{spec[7:]}_k{k}" if spec.startswith("hybrid-") else spec


def read_log(path):
    return pd.read_csv(path, dtype=DTYPES) if path.exists() else None


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")
    print("written", GEN / name)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("written", FIG / name)


def todo_figure(label, caption, what):
    """Placeholder figure environment, so \\ref still works and the gap is impossible to miss."""
    return ("\\begin{figure}[htbp]\\centering\n\\fbox{\\parbox{0.9\\textwidth}{\\centering\\vspace{2em}"
            f"\\todo{{{what}}}\\vspace{{2em}}}}}}\n\\caption{{{caption}}}\\label{{{label}}}\n\\end{{figure}}\n")


def ci(s, key):
    return f"{s[key]:.3f} {{\\scriptsize[{s[key + '_lo']:.2f}, {s[key + '_hi']:.2f}]}}"


# --------------------------------------------------------------------------- main experiment
def main_experiment():
    """Single-card synthetic queries: every method x severity 1-5 (results/main/<run>_s<sev>.csv)."""
    rows = []
    for spec in MAIN_METHODS:
        name = run_name(spec)
        for sev in SEVERITIES:
            log = read_log(RES / "main" / f"{name}_s{sev}.csv")
            if log is None:
                rows.append({"method": name, "severity": sev, "done": False})
                continue
            s = summarize(log, name)
            ok = log["rect_iou"] >= 0.8
            s.update(done=True, severity=sev, n_failed=int((~log["top1"].astype(bool)).sum()),
                     top1_given_rect=float(log.loc[ok, "top1"].astype(float).mean()) if ok.any() else float("nan"),
                     top1_given_bad_rect=float(log.loc[~ok, "top1"].astype(float).mean()) if (~ok).any() else float("nan"),
                     n_bad_rect=int((~ok).sum()))
            rows.append(s)
            key = f"{name}-s{sev}"
            for m in ["top1", "top1_lo", "top1_hi", "top5", "mAP", "top1_len", "rect_ok", "rectify_found",
                      "top1_given_rect", "top1_given_bad_rect"]:
                put(f"{key}-{m}", s[m])
            put(f"{key}-n", s["n_queries"], "{}")
            put(f"{key}-nfailed", s["n_failed"], "{}")
            put(f"{key}-nbadrect", s["n_bad_rect"], "{}")
            put(f"{key}-ms", s["ms_total_median"], "{:,.0f}")
            put(f"{key}-msrect", s["ms_rectify"], "{:.0f}")
    df = pd.DataFrame(rows)
    done = df[df["done"]]
    put("main-runs-done", len(done), "{}")
    put("main-runs-total", len(df), "{}")
    put("main-n-queries", N_QUERIES, "{}")
    # rectification is the same for every method (same detector, same query images)
    for sev in SEVERITIES:
        d = done[done["severity"] == sev]
        if len(d):
            put(f"rect-s{sev}-ok", 100 * d["rect_ok"].iloc[0], "{:.1f}")
            put(f"rect-s{sev}-found", 100 * d["rectify_found"].iloc[0], "{:.1f}")

    # Table: strict / lenient top-1 per method and severity
    lines = ["\\begin{table}[htbp]\\centering\\footnotesize",
             "\\caption[Top-1 accuracy per method and severity]{Top-1 identification accuracy on the synthetic "
             f"single-card test queries ({N_QUERIES} per severity), shown as strict / lenient. Strict = exact card id; "
             "lenient = any card with the same artwork (reprint). "
             + ("\\nr{} = run not finished yet. " if not df["done"].all() else "") +
             "95\\% bootstrap intervals are in Table~\\ref{tab:main-detail} and Figure~\\ref{fig:severity}.}",
             "\\label{tab:main}",
             "\\begin{tabular}{@{}l" + "c" * len(SEVERITIES) + "@{}}", "\\toprule",
             "Method & " + " & ".join(f"S{s}" for s in SEVERITIES) + "\\\\", "\\midrule"]
    for spec in MAIN_METHODS:
        name = run_name(spec)
        cells = []
        for sev in SEVERITIES:
            r = df[(df["method"] == name) & (df["severity"] == sev)].iloc[0]
            cells.append(f"{r['top1']:.3f} / {r['top1_len']:.3f}" if r["done"] else "\\nr")
        lines.append(f"{LABEL[name]} & " + " & ".join(cells) + "\\\\")
        if spec == "pixel":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_main.tex", "\n".join(lines) + "\n")

    # Appendix table: every finished run with CIs
    lines = ["\\begin{table}[p]\\centering\\scriptsize\\renewcommand{\\arraystretch}{0.92}",   # 55 rows: own page
             "\\caption[All finished single-card runs]{All finished runs of the main experiment. Top-1 with 95\\% "
             "bootstrap interval (1000 resamples); MRR = mean reciprocal rank (= mAP, one relevant card per query); "
             "Rect.\\ ok = fraction of queries whose detected outline has IoU $\\geq$ 0.8 with the true outline; "
             "Top-1$\\mid$ok = top-1 among those queries; latency = median per photo (shared PC, see text).}",
             "\\label{tab:main-detail}",
             "\\begin{tabular}{@{}llcccccr@{}}", "\\toprule",
             "Method & Sev. & Top-1 [95\\% CI] & Top-5 & MRR & Rect.\\ ok & Top-1$\\mid$ok & ms \\\\", "\\midrule"]
    for r in done.to_dict("records"):
        lines.append(f"{LABEL[r['method']]} & {r['severity']} & {ci(r, 'top1')} & {r['top5']:.3f} & {r['mAP']:.3f} & "
                     f"{r['rect_ok']:.3f} & {r['top1_given_rect']:.3f} & {r['ms_total_median']:,.0f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_main_detail.tex", "\n".join(lines) + "\n")

    # Figure: accuracy vs severity with CIs
    missing = sorted({LABEL[run_name(s)] for s in MAIN_METHODS} -
                     {LABEL[m] for m in done["method"].unique() if len(done[done["method"] == m]) == len(SEVERITIES)})
    # the 7 base methods only (one colour each); the +rot variants are in Table tab:main and the text
    fig, ax = plt.subplots(figsize=(W, 3.0))
    base = [run_name(s) for s in MAIN_METHODS if "+rot" not in s]
    for j, name in enumerate(m for m in base if m in set(done["method"])):
        d = done[done["method"] == name].sort_values("severity")
        err = np.vstack([d["top1"] - d["top1_lo"], d["top1_hi"] - d["top1"]])
        ax.errorbar(d["severity"] + (j - 3) * 0.03, d["top1"], yerr=err, color=PALETTE[j], marker=MARKERS[j], ms=4,
                    lw=1.4, ls=LINES[j], capsize=2, elinewidth=0.8, label=PLAIN[name])
    ax.set(xlabel="Synthetic distortion severity", ylabel="Top-1 accuracy (strict)", ylim=(0, 1.03),
           xticks=SEVERITIES, xlim=(0.8, 5.2))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    save(fig, "accuracy_vs_severity.pdf")
    note = (f" \\todo{{Missing or incomplete (not all 5 severities): {', '.join(missing)}. Run \\texttt{{python -m cardrec.experiments --exp main}}.}}"
            if missing else "")
    write("fig_severity.tex",
          "\\begin{figure}[htbp]\\centering\n\\includegraphics[width=\\textwidth]{figures/accuracy_vs_severity.pdf}\n"
          "\\caption[Accuracy versus distortion severity]{Strict top-1 accuracy versus synthetic distortion severity "
          f"for the seven base methods, with 95\\% bootstrap intervals ({N_QUERIES} queries per point; points slightly "
          f"shifted horizontally for readability; the +rot variants are in Table~\\ref{{tab:main}}).{note}}}"
          "\\label{fig:severity}\n\\end{figure}\n")
    return df


# --------------------------------------------------------------------------- k sweep / ablation / scenes
def ksweep(main_df):
    rows = []
    for emb in ["resnet", "clip"]:
        for k in KS:
            log = read_log(RES / "ksweep" / f"hybrid_{emb}_k{k}_s{KSWEEP_SEVERITY}.csv")
            if log is not None:
                s = summarize(log, f"hybrid_{emb}_k{k}")
                rows.append({**s, "emb": emb, "k": k})
                for m in ["top1", "top1_lo", "top1_hi", "top5"]:
                    put(f"ks-{emb}-k{k}-{m}", s[m])
                put(f"ks-{emb}-k{k}-ms", s["ms_total_median"], "{:,.0f}")
    caption = (f"Hybrid shortlist size $k$ ({', '.join(map(str, KS))}) at severity {KSWEEP_SEVERITY}: "
               "accuracy versus median latency, compared with each component alone.")
    if not rows:
        write("fig_ksweep.tex", todo_figure("fig:ksweep", caption,
              "k-sweep not run. Run \\texttt{python -m cardrec.experiments --exp ksweep}, then this script."))
        return
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for i, emb in enumerate(["resnet", "clip"]):
        e = d[d["emb"] == emb].sort_values("k")
        if len(e):
            ax.plot(e["ms_total_median"], e["top1"], color=PALETTE[i], marker=MARKERS[i], lw=1.4, ls=LINES[i],
                    label=f"Hybrid {'CLIP' if emb == 'clip' else 'ResNet'}→SIFT")
            for r in e.itertuples():
                ax.annotate(f"k={r.k}", (r.ms_total_median, r.top1), xytext=(3, -10), textcoords="offset points",
                            fontsize=7, color=MUTED)
    for j, m in enumerate(["orb", "sift", "clip", "resnet"]):    # each method alone, same severity
        r = main_df[(main_df["method"] == m) & (main_df["severity"] == KSWEEP_SEVERITY) & main_df["done"]]
        if len(r):
            ax.scatter(r["ms_total_median"], r["top1"], color=PALETTE[2 + j], marker=MARKERS[2 + j], s=26,
                       zorder=3, label=f"{PLAIN[m]} alone (exhaustive)" if m in ("orb", "sift") else f"{PLAIN[m]} alone")
    ax.set_xscale("log")
    ticks = [t for t in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
             if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xlabel="Median latency per photo (ms, log scale)", ylabel="Top-1 accuracy (strict)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "ksweep.pdf")
    write("fig_ksweep.tex", "\\begin{figure}[htbp]\\centering\n\\includegraphics[width=\\textwidth]{figures/ksweep.pdf}\n"
          f"\\caption{{{caption}}}\\label{{fig:ksweep}}\n\\end{{figure}}\n")


def norectify(main_df):
    rows = []
    for spec in MAIN_METHODS:
        name = run_name(spec)
        log = read_log(RES / "norectify" / f"{name}_s3.csv")
        with_rect = main_df[(main_df["method"] == name) & (main_df["severity"] == 3) & main_df["done"]]
        if log is not None and len(with_rect):
            s = summarize(log, name)
            rows.append(f"{LABEL[name]} & {with_rect['top1'].iloc[0]:.3f} & {s['top1']:.3f}\\\\")
            put(f"norect-{name}-top1", s["top1"])
    if not rows:
        write("tab_norectify.tex", "\\noindent\\todo{Rectification ablation not run (Table placeholder). Run "
              "\\texttt{python -m cardrec.experiments --exp norectify}: every method at severity 3 with the whole photo "
              "resized instead of rectified, then this script.}\n")
        return
    write("tab_norectify.tex", "\n".join([
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption[Rectification ablation]{Ablation at severity 3: strict top-1 with rectification versus the whole "
        "photo resized to the canonical size.}\\label{tab:norectify}",
        "\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Method & Rectified & Whole photo\\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n")


def scenes():
    rows = []
    for spec in SCENE_METHODS:
        name = run_name(spec)
        for ov in SCENE_OVERLAPS:
            p = RES / "scenes" / f"{name}_o{int(100 * ov)}.json"
            if p.exists():
                s = json.loads(p.read_text(encoding="utf-8"))
                rows.append({**s, "overlap": ov})
                key = f"sc-{name}-o{int(100 * ov)}"
                for m in ["precision", "recall", "f1", "e2e_top1", "e2e_top1_lo", "e2e_top1_hi", "id_acc_given_detected"]:
                    put(f"{key}-{m}", s[m])
                put(f"{key}-ncards", s["n_true_cards"], "{}"), put(f"sc-o{int(100 * ov)}-recall", s["recall"])
                put(f"sc-o{int(100 * ov)}-precision", s["precision"])
    if not rows:
        write("tab_scenes.tex", "\\noindent\\todo{Multi-card scene experiment not run on the test scenes. Run "
              "\\texttt{python -m cardrec.experiments --exp scenes} (100 scenes of 2--6 cards per overlap level "
              "0/15/30\\%), then this script.}\n")
        return
    lines = ["\\begin{table}[htbp]\\centering\\small",
             "\\caption[Multi-card scenes]{Multi-card synthetic scenes (severity 2): detection precision / recall / F1 "
             "(IoU $\\geq$ 0.5) and end-to-end top-1 (detected and correctly identified / all cards).}\\label{tab:scenes}",
             "\\begin{tabular}{@{}lccccc@{}}", "\\toprule", "Method & Overlap & P & R & F1 & End-to-end top-1\\\\",
             "\\midrule"]
    for r in rows:
        lines.append(f"{LABEL.get(r['method'], r['method'])} & {int(100 * r['overlap'])}\\% & {r['precision']:.3f} & "
                     f"{r['recall']:.3f} & {r['f1']:.3f} & {ci(r, 'e2e_top1')}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_scenes.tex", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- failures
def failure_rows():
    """Recompute causes from the current logs with the project's rules; fall back to results/failures.csv."""
    try:
        from cardrec import failure_analysis as fa
        from cardrec.manifest import load_manifest
        m = load_manifest()
        art, name = dict(zip(m["id"], m["art_group"])), dict(zip(m["id"], m["name"]))
        return pd.DataFrame(fa.single_card(m, art, name)), m, "recomputed from the current logs"
    except Exception as e:                               # no data folder on this PC
        print("failure analysis from results/failures.csv (could not recompute:", str(e)[:80], ")")
        return pd.read_csv(RES / "failures.csv", dtype=str), None, "from results/failures.csv"


def failures():
    from cardrec.failure_analysis import CAUSES
    df, manifest, source = failure_rows()
    df = df[df["set"] == "synthetic"]
    methods = [run_name(s) for s in MAIN_METHODS if run_name(s) in set(df["method"])]
    piv = (df.pivot_table(index="method", columns="cause", values="query_id", aggfunc="count", fill_value=0)
           .reindex(index=methods, columns=CAUSES, fill_value=0))
    for m in methods:
        for c in CAUSES:
            put(f"fail-{m}-{c.replace(' ', '')}", int(piv.loc[m, c]), "{}")
        put(f"fail-{m}-total", int(piv.loc[m].sum()), "{}")
    for (m, sev), n in df.groupby(["method", "severity"]).size().items():
        put(f"fail-{m}-s{sev}", int(n), "{}")
    for (m, c, sev), n in df.groupby(["method", "cause", "severity"]).size().items():
        put(f"fail-{m}-{c.replace(' ', '')}-s{sev}", int(n), "{}")
    lines = ["\\begin{table}[htbp]\\centering\\small",
             "\\caption[Failure causes]{Cause of every strict top-1 failure on the synthetic single-card queries "
             f"(all finished severities pooled; {source}). Each failure gets the first cause that applies, in "
             "the column order.}\\label{tab:failures}",
             "\\begin{tabular}{@{}l" + "r" * (len(CAUSES) + 1) + "@{}}", "\\toprule",
             "Method & " + " & ".join(f"\\rotatebox{{70}}{{{c}}}" for c in CAUSES) + " & \\rotatebox{70}{total}\\\\",
             "\\midrule"]
    for m in methods:
        sevs = sorted(df.loc[df["method"] == m, "severity"].astype(int).unique())
        lines.append(f"{LABEL[m]} {{\\scriptsize(S{min(sevs)}--{max(sevs)})}} & " +
                     " & ".join(str(int(piv.loc[m, c])) for c in CAUSES) + f" & {int(piv.loc[m].sum())}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_failures.tex", "\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(W, 0.5 + 0.45 * len(methods)))
    left = np.zeros(len(methods))
    for i, c in enumerate(CAUSES):
        ax.barh(range(len(methods)), piv[c], left=left, color=PALETTE[i], label=c, edgecolor="white", linewidth=1)
        left += piv[c].values
    ax.set(yticks=range(len(methods)), yticklabels=[PLAIN[m] for m in methods], xlabel="Failed queries")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    save(fig, "failure_causes.pdf")
    return df, manifest


def qualitative(failed, manifest):
    """Example photos: one correct pipeline run, then failures of different causes (needs the data folder)."""
    import cv2

    from cardrec.rectify import rectify
    if manifest is None:
        for f, lab in [("fig_pipeline_example.tex", "fig:pipeline-example"), ("fig_failure_examples.tex", "fig:failure-examples")]:
            write(f, todo_figure(lab, "Qualitative examples.", "Run make\\_figures.py on the PC that has the data folder."))
        return
    path = manifest.set_index("id")["abs_path"]

    def query(sev, qid):
        q = pd.read_csv(config.QUERIES_DIR / f"s{sev}" / "queries.csv", dtype=str).set_index("query_id").loc[qid]
        return cv2.imread(str(config.DATA_DIR / q["image_path"])), np.array(json.loads(q["true_quad"]))

    def draw(img, true_quad, found_quad):
        out = img.copy()
        cv2.polylines(out, [true_quad.astype(np.int32)], True, (0, 200, 0), 4)
        if found_quad is not None:
            cv2.polylines(out, [found_quad.astype(np.int32)], True, (0, 0, 255), 3)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def ref(cid):
        return cv2.cvtColor(cv2.resize(cv2.imread(path[cid]), (config.CANON_W, config.CANON_H)), cv2.COLOR_BGR2RGB)

    # 1) a correct run at severity 3 (first query that ORB identified)
    log = read_log(RES / "main" / "orb_s3.csv")
    r = log[log["top1"].astype(bool)].iloc[0]
    img, tq = query(3, r["query_id"])
    card, quad, _ = rectify(img)
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.83, 1, 1]})
    for a, im, t in zip(axs, [draw(img, tq, quad), cv2.cvtColor(card, cv2.COLOR_BGR2RGB), ref(r["true_id"])],
                        ["(a) synthetic query, severity 3", "(b) rectified 500×700", f"(c) top-1: {r['pred_id']}"]):
        a.imshow(im)
        a.set_title(t, fontsize=8)
        a.axis("off")
    save(fig, "pipeline_example.pdf")
    write("fig_pipeline_example.tex",
          "\\begin{figure}[htbp]\\centering\n\\includegraphics[width=\\textwidth]{figures/pipeline_example.pdf}\n"
          "\\caption[Pipeline example]{Example run of the pipeline. (a) Synthetic query "
          f"\\texttt{{{r['query_id'].replace('_', '\\_')}}}; green = true outline, red = outline found by the detector. "
          "(b) The card warped to the canonical view. (c) The reference scan ranked first by ORB + RANSAC "
          f"(correct; {int(r['pred_score'])} RANSAC inliers versus {int(r['second_score'])} for the second card).}}"
          "\\label{fig:pipeline-example}\n\\end{figure}\n")

    # 2) failures: same artwork (keypoints), localisation (heavy distortion), identification under distortion
    picks = []
    for method, cause in [("sift", "same artwork"), ("orb", "same artwork"), ("orb", "localization"),
                          ("orb", "heavy distortion"), ("pixel", "upside down")]:
        sub = failed[(failed["method"] == method) & (failed["cause"] == cause)]
        if len(sub) and cause not in [c for _, c, _ in picks]:
            picks.append((method, cause, sub.iloc[0]))
    fig, axs = plt.subplots(len(picks), 4, figsize=(W, 2.2 * len(picks)),
                            gridspec_kw={"width_ratios": [1.83, 1, 1, 1]})
    cap = []
    for row, (method, cause, f) in zip(np.atleast_2d(axs), picks):
        img, tq = query(int(f["severity"]), f["query_id"])
        card, quad, _ = rectify(img)
        ims = [draw(img, tq, quad), cv2.cvtColor(card, cv2.COLOR_BGR2RGB), ref(f["pred_id"]), ref(f["true_id"])]
        titles = [f"{PLAIN[method]}, S{f['severity']}: {cause}", "rectified", f"predicted {f['pred_id']}",
                  f"true {f['true_id']}"]
        for a, im, t in zip(row, ims, titles):
            a.imshow(im)
            a.set_title(t, fontsize=7)
            a.axis("off")
        cap.append(f"{cause} ({PLAIN[method].replace('+', '+')}, severity {f['severity']})")
    save(fig, "failure_examples.pdf")
    write("fig_failure_examples.tex",
          "\\begin{figure}[htbp]\\centering\n"
          "\\includegraphics[width=\\textwidth,height=0.8\\textheight,keepaspectratio]{figures/failure_examples.pdf}\n"
          "\\caption[Failure examples]{Failure examples, one per row: " + "; ".join(cap) +
          ". Columns: query (green = true outline, red = detected outline), rectified card, predicted reference, "
          "true reference.}\\label{fig:failure-examples}\n\\end{figure}\n")


# --------------------------------------------------------------------------- language
LANG_METHODS = ["sift", "sift@art", "clip", "clip@art", "hybrid_clip+rot_k20", "hybrid_clip@art+rot_k20"]


def language():
    # (a) identification per printed language: low-resolution (card ~530 px) and phone-resolution (~1050 px)
    res = {}
    for m in LANG_METHODS:
        for lg in ["en", "es", "ja"]:
            for tag, suffix in [("low", ""), ("hr", "_hr")]:
                log = read_log(RES / "language" / f"{m}_{lg}{suffix}_s2.csv")
                if log is not None:
                    s = summarize(log, m)
                    res[m, lg, tag] = s
                    put(f"lang-{m}-{lg}-{tag}-len", s["top1_len"])
                    put(f"lang-{m}-{lg}-{tag}-len-lo", s["top1_len_lo"])
                    put(f"lang-{m}-{lg}-{tag}-len-hi", s["top1_len_hi"])
                    put(f"lang-{m}-{lg}-{tag}-strict", s["top1"])
                    put(f"lang-{lg}-n", s["n_queries"], "{}")
    for m in LANG_METHODS:                            # pooled median latency, phone resolution
        logs = [read_log(RES / "language" / f"{m}_{lg}_hr_s2.csv") for lg in ["en", "es", "ja"]]
        logs = [x for x in logs if x is not None]
        if logs:
            d = pd.concat(logs)
            put(f"lang-{m}-ms", float((d["ms_rectify"] + d["ms_retrieval"]).median()), "{:,.0f}")
    summ = RES / "summary_language_id.csv"
    if summ.exists():
        idx = pd.read_csv(summ).groupby("method")["index_MB"].first()
        for m, mb in idx.items():
            put(f"index-{m}-MB", float(mb), "{:.1f}")

    def cell(m, lg, tag):
        s = res.get((m, lg, tag))
        return ci(s, "top1_len") if s else "\\nr"
    lines = ["\\begin{table}[htbp]\\centering\\footnotesize",
             "\\caption[Identification per printed language]{Lenient top-1 identification per printed language "
             "(test split, severity 2) with 95\\% bootstrap intervals, at two photo resolutions: card about 530~px tall "
             "(``low'') and about 1050~px tall (``phone'', like a 12~MP photo). @art = only the artwork window is "
             f"compared; +rot = the query is also tried rotated 180\\textdegree. Queries: en {numbers.get('lang-en-n', '?')}, "
             f"es {numbers.get('lang-es-n', '?')}, ja {numbers.get('lang-ja-n', '?')}.}}\\label{{tab:language-id}}",
             "\\begin{tabular}{@{}lllll@{}}", "\\toprule", "Method & Res. & English & Spanish & Japanese\\\\", "\\midrule"]
    for m in LANG_METHODS:
        for tag, lab in [("low", "low"), ("hr", "phone")]:
            lines.append(f"{LABEL[m] if tag == 'low' else ''} & {lab} & " +
                         " & ".join(cell(m, lg, tag) for lg in ["en", "es", "ja"]) + "\\\\")
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_language_id.tex", "\n".join(lines) + "\n")

    # (b) printed-language classification (OCR + rules)
    runs = [("v1", "test", "low", RES / "language" / "language_s2.csv"),
            ("v2", "dev", "phone", RES / "language_dev" / "language_v2_s2.csv"),
            ("v2", "test", "phone", RES / "language" / "language_v2_s2.csv")]
    lines = ["\\begin{table}[htbp]\\centering\\small",
             "\\caption[Printed-language classification]{Printed-language classification (English / Spanish / Japanese) "
             "with OCR + rules. Version 1 rules were fixed before seeing any data; version 2 was tuned on the dev split "
             "only. ``Unknown'' = no usable text; ``wrong'' = another language predicted.}\\label{tab:language-clf}",
             "\\begin{tabular}{@{}lllrcrr@{}}", "\\toprule",
             "Rules & Split & Res. & $n$ & Accuracy [95\\% CI] & Unknown & Wrong\\\\", "\\midrule"]
    for ver, split, reso, p in runs:
        if not p.exists():
            lines.append(f"{ver} & {split} & {reso} & \\multicolumn{{4}}{{c}}{{\\nr}}\\\\")
            continue
        d = pd.read_csv(p)
        ok = (d["language"] == d["pred_language"]).astype(float)
        lo, hi = bootstrap_ci(ok)
        unk = int((d["pred_language"] == "unknown").sum())
        wrong = int(((d["pred_language"] != d["language"]) & (d["pred_language"] != "unknown")).sum())
        key = f"langclf-{ver}-{split}"
        put(f"{key}-acc", ok.mean()), put(f"{key}-lo", lo), put(f"{key}-hi", hi)
        put(f"{key}-n", len(d), "{}"), put(f"{key}-unknown", unk, "{}"), put(f"{key}-wrong", wrong, "{}")
        put(f"{key}-falseja", int(((d["pred_language"] == "ja") & (d["language"] != "ja")).sum()), "{}")
        put(f"{key}-coderule", int((d["rule"] == "language code").sum()), "{}")
        lines.append(f"{ver} & {split} & {reso} & {len(d)} & {ok.mean():.3f} {{\\scriptsize[{lo:.2f}, {hi:.2f}]}} & "
                     f"{unk} & {wrong}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_language_clf.tex", "\n".join(lines) + "\n")

    cms = [(p, t) for p, t in [(RES / "language" / "confusion_s2.csv", "(a) version 1, test, low resolution"),
                               (RES / "language" / "confusion_v2_s2.csv", "(b) version 2, test, phone resolution")]
           if p.exists()]
    fig, axs = plt.subplots(1, len(cms), figsize=(W, 2.3))
    for a, (p, title) in zip(np.atleast_1d(axs), cms):
        cm = pd.read_csv(p, index_col=0)
        a.imshow(cm.values, cmap="Blues", vmin=0)
        a.grid(False)
        for (i, j), v in np.ndenumerate(cm.values):
            a.text(j, i, int(v), ha="center", va="center", fontsize=9,
                   color="white" if v > 0.6 * cm.values.max() else INK)
        a.set(xticks=range(cm.shape[1]), xticklabels=cm.columns, yticks=range(cm.shape[0]), yticklabels=cm.index,
              xlabel="predicted", ylabel="true")
        a.set_title(title, fontsize=8)
    save(fig, "language_confusion.pdf")


# --------------------------------------------------------------------------- condition
def condition():
    p = RES / "condition" / "validation_test_s1.csv"
    if not p.exists():
        write("tab_condition.tex", "\\noindent\\todo{Condition validation not run: "
              "\\texttt{python -m cardrec.condition\\_eval --n 80 --split test}.}\n")
        return
    df = pd.read_csv(p)
    d = df[df["rectified"] == True]                                       # noqa: E712
    put("cond-n", len(df), "{}"), put("cond-rectified", len(d), "{}"), put("cond-aligned", int(d["aligned"].sum()), "{}")
    dev = RES / "condition" / "validation_dev_s1.csv"
    if dev.exists():
        dev_cards = set(pd.read_csv(dev)["card"])
        put("cond-dev-n", len(dev_cards), "{}"), put("cond-dev-overlap", len(dev_cards & set(df["card"])), "{}")
    example = RES / "condition" / "example_test_0.jpg"
    if example.exists():
        shutil.copyfile(example, FIG / "condition_example.jpg")
        print("written", FIG / "condition_example.jpg")
    lr, tb = abs(d.true_left_pct - d.est_left_pct), abs(d.true_top_pct - d.est_top_pct)
    for k, v in [("lr-med", lr.median()), ("lr-p90", np.percentile(lr, 90)),
                 ("tb-med", tb.median()), ("tb-p90", np.percentile(tb, 90))]:
        put(f"cond-centering-{k}", float(v), "{:.2f}")
    rho = {}
    for grp, names, pairs in [("corner", CORNER_PTS, [("white", "whitening"), ("cut", "fill")]),
                              ("edge", ["top", "right", "bottom", "left"], [("white", "whitening"), ("fray", "fray")])]:
        for t, e in pairs:
            true = pd.concat([d[f"{grp}_{n}_{t}"] for n in names], ignore_index=True)
            est = pd.concat([d[f"est_{grp}_{n}_{e}"] for n in names], ignore_index=True)
            rho[f"{grp}-{t}"] = spearman(true, est)
            put(f"cond-rho-{grp}-{t}", rho[f"{grp}-{t}"], "{:+.2f}")
    nt, nf, ne, nr = d.n_true_defects.sum(), d.n_found.sum(), d.n_est_defects.sum(), d.n_est_real.sum()
    put("cond-surf-recall", nf / max(nt, 1), "{:.2f}"), put("cond-surf-precision", nr / max(ne, 1), "{:.2f}")
    put("cond-surf-ntrue", int(nt), "{}"), put("cond-surf-found", int(nf), "{}")
    put("cond-surf-typed", int(d.n_typed.sum()), "{}"), put("cond-surf-nest", int(ne), "{}"), put("cond-surf-real", int(nr), "{}")
    dmg = {"corners": d.filter(regex="^corner_").sum(axis=1), "edges": d.filter(regex="^edge_").sum(axis=1),
           "surface": d.n_true_defects, "centering": np.maximum(abs(d.true_left_pct - 50), abs(d.true_top_pct - 50))}
    part = {k: spearman(v, d[f"sub_{k}"]) for k, v in dmg.items()}
    for k, v in part.items():
        put(f"cond-part-{k}", v, "{:+.2f}")
    total = dmg["corners"] + dmg["edges"] + d.n_true_defects + dmg["centering"] / 10
    put("cond-overall", spearman(total, d.overall), "{:+.2f}")
    n = numbers
    write("tab_condition.tex", "\n".join([
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption[Condition estimate validation]{Validation of the visual condition estimate on "
        f"{n['cond-n']} test cards with synthetic damage of known strength (front side only). "
        "$\\rho$ = Spearman rank correlation between true damage strength and the estimate.}\\label{tab:condition}",
        "\\begin{tabular}{@{}lr@{}}", "\\toprule", "Measure & Result\\\\", "\\midrule",
        f"Cards rectified / reference aligned & {n['cond-rectified']} / {n['cond-aligned']}\\\\",
        f"Centering error L/R, median / 90th pct.\\ (percentage points) & {n['cond-centering-lr-med']} / {n['cond-centering-lr-p90']}\\\\",
        f"Centering error T/B, median / 90th pct.\\ (percentage points) & {n['cond-centering-tb-med']} / {n['cond-centering-tb-p90']}\\\\",
        f"$\\rho$ corner cut vs.\\ fill sub-score (expect $<0$) & {n['cond-rho-corner-cut']}\\\\",
        f"$\\rho$ corner whitening vs.\\ whitening sub-score & {n['cond-rho-corner-white']}\\\\",
        f"$\\rho$ edge whitening vs.\\ whitening sub-score & {n['cond-rho-edge-white']}\\\\",
        f"$\\rho$ edge fray vs.\\ fray sub-score & {n['cond-rho-edge-fray']}\\\\",
        f"Surface defects: recall / precision & {n['cond-surf-found']}/{n['cond-surf-ntrue']} = {n['cond-surf-recall']} / "
        f"{n['cond-surf-real']}/{n['cond-surf-nest']} = {n['cond-surf-precision']}\\\\",
        f"Surface defects with correct type & {n['cond-surf-typed']}/{n['cond-surf-ntrue']}\\\\",
        f"$\\rho$ part score vs.\\ true damage: corners / edges / surface / centering & {n['cond-part-corners']} / "
        f"{n['cond-part-edges']} / {n['cond-part-surface']} / {n['cond-part-centering']}\\\\",
        f"$\\rho$ overall score vs.\\ total damage & {n['cond-overall']}\\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}"]) + "\n")


# --------------------------------------------------------------------------- dataset / setup tables
def dataset():
    try:
        from cardrec.manifest import load_manifest
        m = load_manifest()
    except Exception as e:
        print("dataset table skipped (no manifest):", str(e)[:80])
        return
    g = m.groupby(["set_name", "set_code", "set_id"], sort=False).agg(n=("id", "size"), printed=("printed_total", "first"))
    lines = ["\\begin{table}[htbp]\\centering\\small",
             "\\caption[Reference database]{Reference database: one official scan per card id. ``Printed'' is the set "
             "size printed on the cards; the extra cards are secret rares numbered above it.}\\label{tab:dataset}",
             "\\begin{tabular}{@{}llrr@{}}", "\\toprule", "Set (API id) & Code & Cards & Printed\\\\", "\\midrule"]
    for (name, code, sid), r in g.iterrows():
        lines.append(f"{name} (\\texttt{{{sid}}}) & {code} & {r['n']} & {r['printed']}\\\\")
    lines += ["\\midrule", f"Total & & {len(m)} & {pd.to_numeric(g['printed']).sum()}\\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_dataset.tex", "\n".join(lines) + "\n")
    names = m["name"].value_counts()
    put("db-cards", len(m), "{}"), put("db-sets", m["set_id"].nunique(), "{}")
    put("db-artgroups", m["art_group"].nunique(), "{}")
    put("db-shared-groups", int((m["art_group"].value_counts() > 1).sum()), "{}")
    put("db-dup-names", int((names > 1).sum()), "{}"), put("db-dup-name-cards", int(names[names > 1].sum()), "{}")
    put("db-max-name", names.index[0]), put("db-max-name-n", int(names.iloc[0]), "{}")
    sup = m["supertype"].str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").value_counts()
    for k, v in sup.items():
        put(f"db-{k.lower()}", int(v), "{}")
    ja = config.DATA_DIR / "manifest_ja.csv"                # Japanese cards: used only by the app
    if ja.exists():
        j = pd.read_csv(ja, dtype=str).fillna("")
        put("db-ja-cards", len(j), "{}"), put("db-ja-linked", int((j["en_id"] != "").sum()), "{}")
    pairs = config.DATA_DIR / "art_group_pairs.csv"
    if pairs.exists():
        p = pd.read_csv(pairs)
        put("db-art-pairs", len(p), "{}"), put("db-art-maxfull", int(p.dist_full.max()), "{}")
        put("db-art-maxart", int(p.dist_art.max()), "{}")
        sets = p["card_a"].str.split("-").str[0] + "/" + p["card_b"].str.split("-").str[0]
        for k, v in sets.value_counts().items():
            put(f"db-art-pairs-{k.replace('/', '-')}", int(v), "{}")


def severity_table():
    keys = [("rot", "Max.\\ in-plane rotation ($\\pm$deg)", "{:g}"), ("persp", "Corner jitter (frac.\\ of card height)", "{:g}"),
            ("blur", "Blur $\\sigma$ (px, Gaussian or motion)", "{:g}"), ("noise", "Gaussian noise $\\sigma$ (grey levels)", "{:g}"),
            ("light", "Lighting gradient + gamma strength", "{:g}"), ("glare", "Glare blob strength", "{:g}")]
    lines = ["\\begin{table}[htbp]\\centering\\small",
             "\\caption[Synthetic distortion severities]{Distortion parameters per severity level "
             "(\\texttt{cardrec/synth.py}). Each value is the maximum of a random draw.}\\label{tab:severity}",
             "\\begin{tabular}{@{}l" + "c" * len(SEVERITY) + "@{}}", "\\toprule",
             "Parameter & " + " & ".join(f"S{s}" for s in SEVERITY) + "\\\\", "\\midrule"]
    for k, lab, fmt in keys:
        lines.append(f"{lab} & " + " & ".join(fmt.format(SEVERITY[s][k]) for s in SEVERITY) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_severity.tex", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- application (CardDex)
APP_METHODS = ["pixel", "sift", "orb", "resnet", "clip", "hybrid_clip+rot_k20", "hybrid_clip@art+rot_k20"]


def app():
    """Saved collection of the app + re-identification of its photos (written by report/app_evidence.py)."""
    snap_p, reid_p = RES / "app" / "collection_snapshot.csv", RES / "app" / "collection_reid.csv"
    if not snap_p.exists():
        for f in ["tab_app_collection.tex", "tab_app_reid.tex", "fig_app_photos.tex"]:
            write(f, "\\noindent\\todo{Run \\texttt{python report/app\\_evidence.py}, then this script.}\n")
        return
    snap = pd.read_csv(snap_p, dtype={"card_id": str})
    try:                                                     # the app's database: English + Japanese cards
        from cardrec.manifest import load_app_manifest
        app_man = load_app_manifest().set_index("id")
        rarity = app_man["rarity"].to_dict()
        names_en = app_man["name_en"].to_dict() if "name_en" in app_man else {}
    except Exception:
        rarity, names_en = {}, {}

    def display_name(card_id, name):
        """pdfLaTeX cannot print Japanese: Japanese cards are shown by their English twin's name + (JP)."""
        if not card_id.startswith("ja-"):
            return name
        return f"{names_en.get(card_id) or card_id} (JP)"
    snap["currency"] = np.where(snap["price_source"].fillna("").str.startswith("Cardmarket"), "EUR", "USD")
    snap["value"] = snap["price"].fillna(0) * snap["quantity"]
    put("app-n-items", len(snap), "{}"), put("app-n-copies", int(snap["quantity"].sum()), "{}")
    put("app-n-ja", int(snap["card_id"].str.startswith("ja-").sum()), "{}")
    put("app-n-en", int((~snap["card_id"].str.startswith("ja-")).sum()), "{}")
    put("app-total-usd", float(snap.loc[snap["currency"] == "USD", "value"].sum()), "{:,.2f}")
    put("app-total-eur", float(snap.loc[snap["currency"] == "EUR", "value"].sum()), "{:,.2f}")
    put("app-snapshot-date", snap["snapshot_utc"].iloc[0][:10])
    put("app-n-cond", int(snap["cond_overall"].notna().sum()), "{}")
    if snap["cond_overall"].notna().any():
        put("app-cond-min", int(snap["cond_overall"].min()), "{}"), put("app-cond-max", int(snap["cond_overall"].max()), "{}")
    lines = ["\\begin{table}[htbp]\\centering\\footnotesize",
             "\\caption[Cards saved in the application]{Cards saved in the CardDex collection "
             f"(snapshot of {numbers['app-snapshot-date']}). Price when saved: English cards = mean TCGplayer market "
             "price of the card's variants in USD; Japanese cards = Cardmarket average in EUR (both via TCGdex). "
             "Score of the saved card with the method used at save time: RANSAC inliers for SIFT, ORB and the hybrids, "
             "cosine similarity in \\% for ResNet and CLIP. Condition: overall visual estimate (0--1000; mean of the two "
             "sides) and the front / back side scores as stored by the app (-- = not assessed). "
             "Lang.: printed language stored with the item.}\\label{tab:app-collection}",
             "\\begin{tabular}{@{}P{1.45in}P{1.05in}cP{0.6in}rrP{1.15in}@{}}", "\\toprule",
             "Card (id) & Rarity & Lang. & Method & Price & Score & Condition (overall; front / back)\\\\", "\\midrule"]
    for r in snap.sort_values("id").to_dict("records"):
        key = f"app-{r['card_id']}"
        put(f"{key}-price", float(r["price"]), "{:,.2f}"), put(f"{key}-score", float(r["score"]), "{:g}")
        if pd.notna(r["cond_overall"]):
            put(f"{key}-cond", int(r["cond_overall"]), "{}")
        for side in ["front", "back"]:
            for k in ["overall", "n_defects", "centering", "corners", "edges", "surface", "inliers"]:
                v = r.get(f"cond_{side}_{k}")
                if pd.notna(v) if not isinstance(v, str) else v:
                    put(f"{key}-{side}-{k.replace('_', '')}", int(v) if not isinstance(v, str) else v, "{}")
        sides = []
        for side in ["front", "back"]:
            o = r.get(f"cond_{side}_overall")
            sides.append(f"{int(o)}" if pd.notna(o) else "--")
        method = str(r["method"]).replace("hybrid-clip+rot", "hybrid CLIP +rot")
        cond = f"{int(r['cond_overall'])}; {sides[0]} / {sides[1]}" if pd.notna(r["cond_overall"]) else "--"
        lines.append(f"{display_name(r['card_id'], r['name'])} (\\texttt{{{r['card_id']}}}) & {rarity.get(r['card_id']) or '--'} & "
                     f"{r['language']} & {method} & {r['currency']} {r['price']:,.2f} & {r['score']:g} & {cond}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_app_collection.tex", "\n".join(lines) + "\n")

    if not reid_p.exists():
        return
    reid = pd.read_csv(reid_p, dtype={"true_id": str, "pred_id": str})
    cards = list(dict.fromkeys(reid["true_id"]))
    names = {c: display_name(c, n) for c, n in snap.set_index("card_id")["name"].items()}
    n_ref = int(numbers.get("db-cards", 0)) + int(numbers.get("db-ja-cards", 0))
    put("app-reid-n", len(cards), "{}"), put("app-n-ref", n_ref, "{}")
    lines = ["\\begin{table}[htbp]\\centering\\footnotesize",
             "\\caption[Re-identification of the saved photos]{Re-identification of the photos saved in the application "
             "with every method (same path as the app: rectification, then ranking of all "
             f"{n_ref} reference cards of the app, English and Japanese). Each cell is the rank of the saved card "
             "(1 = correct top-1); $\\dagger$ = wrong exact card but same artwork (lenient top-1, e.g.\\ the English twin "
             "of a Japanese card); bold = wrong even under the lenient metric. Ground truth = the card the user saved. "
             "Top-1 = strict / lenient. Latency: median over the photos (measured while other experiments were "
             "running, indicative only).}\\label{tab:app-reid}",
             "\\begin{tabular}{@{}l" + "c" * len(cards) + "rr@{}}", "\\toprule",
             "Method & " + " & ".join(f"\\rotatebox{{70}}{{{names.get(c, c)}}}" for c in cards) +
             " & Top-1 & ms\\\\", "\\midrule"]
    for m in APP_METHODS:
        d = reid[reid["method"] == m].set_index("true_id")
        if d.empty:
            continue
        hits, hits_len = int(d["top1"].sum()), int(d["top1_lenient"].sum())
        put(f"app-{m}-hits", hits, "{}"), put(f"app-{m}-hits-len", hits_len, "{}")
        put(f"app-{m}-ms", float(d["ms_total"].median()), "{:,.0f}")
        for c in cards:
            put(f"app-{m}-{c}-rank", int(d.loc[c, "rank_of_true"]), "{}")
            put(f"app-{m}-{c}-pred", d.loc[c, "pred_id"])

        def cell(c):
            rank = int(d.loc[c, "rank_of_true"])
            if rank == 1:
                return "1"
            return f"{rank}$^\\dagger$" if d.loc[c, "top1_lenient"] else f"\\textbf{{{rank}}}"
        lines.append(f"{LABEL[m]} & " + " & ".join(cell(c) for c in cards) +
                     f" & {hits}/{hits_len} & {d['ms_total'].median():,.0f}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab_app_reid.tex", "\n".join(lines) + "\n")
    for c in cards:
        sub = reid[reid["true_id"] == c]
        put(f"app-{c}-found", "yes" if sub["rectify_found"].iloc[0] else "no")
        s = reid[(reid["true_id"] == c) & (reid["method"] == "sift")]
        if len(s):
            put(f"app-{c}-sift-top1", int(s["pred_score"].iloc[0]), "{}"), put(f"app-{c}-sift-second", int(s["second_score"].iloc[0]), "{}")

    # figure: each saved photo and the card that the detector straightened from it
    import cv2
    up = config.DATA_DIR / "uploads"
    pairs = [(r["card_id"], names.get(r["card_id"], r["name"]), up / r["photo"], up / r["photo"].replace(".png", "_rect.png"))
             for r in snap.sort_values("id").to_dict("records") if isinstance(r.get("photo"), str)]
    pairs = [p for p in pairs if p[2].exists() and p[3].exists()]
    if not pairs:
        write("fig_app_photos.tex", todo_figure("fig:app-photos", "Saved photos.", "The uploaded photos are not on this PC."))
        return
    fig, axs = plt.subplots(2, len(pairs), figsize=(W, 1.0 + 3.6 * 7 / max(len(pairs), 7)),
                            gridspec_kw={"height_ratios": [1.15, 1]}, squeeze=False)
    for j, (cid, name, photo, rect) in enumerate(pairs):
        for i, p in enumerate([photo, rect]):
            a = axs[i, j]
            a.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
            a.axis("off")
        axs[0, j].set_title(f"{name}\n{cid}", fontsize=7)
    axs[0, 0].text(-0.08, 0.5, "uploaded", rotation=90, transform=axs[0, 0].transAxes, va="center", ha="right", fontsize=8)
    axs[1, 0].text(-0.08, 0.5, "straightened", rotation=90, transform=axs[1, 0].transAxes, va="center", ha="right", fontsize=8)
    save(fig, "app_photos.pdf")
    write("fig_app_photos.tex",
          "\\begin{figure}[htbp]\\centering\n\\includegraphics[width=\\textwidth]{figures/app_photos.pdf}\n"
          "\\caption[Photos saved in the application]{The photos stored with the saved cards (top) and the card image "
          "that the application straightened from each one and passed to the retrieval method (bottom).}"
          "\\label{fig:app-photos}\n\\end{figure}\n")


def write_numbers():
    body = ["% generated by report/make_figures.py - do not edit by hand",
            *[f"\\expandafter\\def\\csname res:{k}\\endcsname{{{v}}}" for k, v in sorted(numbers.items())]]
    write("numbers.tex", "\n".join(body) + "\n")


if __name__ == "__main__":
    GEN.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    dataset()
    severity_table()
    main_df = main_experiment()
    ksweep(main_df)
    norectify(main_df)
    scenes()
    failed, manifest = failures()
    qualitative(failed, manifest)
    language()
    condition()
    app()
    write_numbers()
