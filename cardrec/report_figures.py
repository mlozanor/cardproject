"""Step 10: report figures (PNG, IEEE single-column width) from the result tables.

    python -m cardrec.report_figures          # writes results/figures/*.png; skips figures without data

Style: thin marks, light grid, at most 5 colours per chart from a colour-blind-checked palette
(validated with the dataviz validator: all hard checks pass in light mode), and every series also has
its own marker / line style and a direct label, so the figures still work in grayscale print.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
OUT = config.RESULTS_DIR / "figures"
W = 3.5                                                     # inches, IEEE column

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def _read(name):
    p = config.RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("written", OUT / name)


def accuracy_vs_severity(df):
    methods = [("orb", "ORB"), ("sift", "SIFT"), ("resnet", "ResNet-50"), ("clip", "CLIP"),
               ("hybrid_clip_k20", "Hybrid CLIP+SIFT")]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for i, (m, label) in enumerate(methods):
        d = df[df["method"] == m].sort_values("severity")
        if d.empty:
            continue
        err = np.vstack([d["top1"] - d["top1_lo"], d["top1_hi"] - d["top1"]])
        ax.errorbar(d["severity"], d["top1"], yerr=err, color=PALETTE[i], marker=MARKERS[i], ms=4, lw=1.4,
                    ls=LINES[i], capsize=2, elinewidth=0.8, label=label)
        ax.annotate(label, (d["severity"].iloc[-1], d["top1"].iloc[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.set(xlabel="Synthetic distortion severity", ylabel="Top-1 accuracy (strict)", ylim=(0, 1.02),
           xticks=[1, 2, 3, 4, 5], xlim=(0.8, 6.3))
    ax.legend(loc="lower left", fontsize=7)
    _save(fig, "accuracy_vs_severity.png")


def rotation_effect(df, severity=5):
    pairs = [("resnet", "resnet+rot", "ResNet"), ("clip", "clip+rot", "CLIP"),
             ("hybrid_resnet_k20", "hybrid_resnet+rot_k20", "Hybrid ResNet"),
             ("hybrid_clip_k20", "hybrid_clip+rot_k20", "Hybrid CLIP")]
    d = df[df["severity"] == severity].set_index("method")
    pairs = [p for p in pairs if p[0] in d.index and p[1] in d.index]
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(len(pairs))
    for j, (col, label) in enumerate([(0, "one orientation"), (1, "+rot (also 180°)")]):
        vals = [d.loc[p[col], "top1"] for p in pairs]
        lo = [d.loc[p[col], "top1"] - d.loc[p[col], "top1_lo"] for p in pairs]
        hi = [d.loc[p[col], "top1_hi"] - d.loc[p[col], "top1"] for p in pairs]
        bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=PALETTE[j], label=label, yerr=[lo, hi],
                      error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": MUTED}, hatch=["", "///"][j],
                      edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.2f", fontsize=6, color=INK, padding=2)
    ax.set(xticks=x, xticklabels=[p[2] for p in pairs], ylabel=f"Top-1 at severity {severity}", ylim=(0, 1.05))
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, "rotation_effect_s5.png")


def ksweep(ks, main):
    fig, ax = plt.subplots(figsize=(W, 2.4))
    for i, emb in enumerate(["resnet", "clip"]):
        d = ks[ks["method"].str.startswith(f"hybrid_{emb}_k")].sort_values("k")
        if d.empty:
            continue
        ax.plot(d["ms_total_median"], d["top1"], color=PALETTE[i], marker=MARKERS[i], ms=4, lw=1.4, ls=LINES[i],
                label=f"Hybrid {emb.upper() if emb == 'clip' else 'ResNet'}+SIFT")
        for r in d.itertuples():
            ax.annotate(f"k={r.k}", (r.ms_total_median, r.top1), xytext=(3, -9), textcoords="offset points",
                        fontsize=6, color=MUTED)
    if main is not None:                                    # the components alone, same severity
        sev = int(ks["severity"].iloc[0])
        for j, (m, label) in enumerate([("sift", "SIFT alone"), ("clip", "CLIP alone"), ("resnet", "ResNet alone")]):
            r = main[(main["method"] == m) & (main["severity"] == sev)]
            if len(r):
                ax.scatter(r["ms_total_median"], r["top1"], color=PALETTE[2 + j], marker=MARKERS[2 + j], s=22,
                           zorder=3, edgecolor="white", linewidth=0.8, label=label)
    ax.set_xscale("log")
    ax.set(xlabel="Median latency per photo (ms, log scale)", ylabel="Top-1 accuracy (strict)")
    ax.legend(fontsize=6.5, loc="lower right")
    _save(fig, "ksweep_accuracy_latency.png")


def rectification(df):
    d = df.copy()
    piv = d.pivot_table(index="method", columns="rectify", values="top1")
    order = [m for m in ["orb", "sift", "resnet", "clip", "hybrid_resnet_k20", "hybrid_clip_k20"] if m in piv.index]
    if not order:
        return
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(len(order))
    for j, (flag, label) in enumerate([(True, "with rectification"), (False, "whole photo resized")]):
        if flag in piv.columns:
            ax.bar(x + (j - 0.5) * 0.38, piv[flag], 0.36, color=PALETTE[j], label=label, hatch=["", "///"][j],
                   edgecolor="white", linewidth=0.5)
    names = {"hybrid_resnet_k20": "Hyb. ResNet", "hybrid_clip_k20": "Hyb. CLIP", "resnet": "ResNet", "clip": "CLIP",
             "sift": "SIFT", "orb": "ORB"}
    ax.set(xticks=x, xticklabels=[names[m] for m in order], ylabel="Top-1 at severity 3", ylim=(0, 1.05))
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    _save(fig, "rectification_ablation.png")


def scenes_fig(df):
    d = df[df["method"] == "hybrid_clip_k20"].sort_values("overlap")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for i, (col, label) in enumerate([("recall", "detection recall"), ("precision", "detection precision"),
                                      ("e2e_top1", "end-to-end top-1 (Hybrid CLIP)")]):
        ax.plot(100 * d["overlap"], d[col], color=PALETTE[i], marker=MARKERS[i], ms=4, lw=1.4, ls=LINES[i], label=label)
    ax.set(xlabel="Maximum overlap between cards (%)", ylabel="Fraction", ylim=(0, 1.05), xticks=[0, 15, 30])
    ax.legend(fontsize=7, loc="lower left")
    _save(fig, "scenes_overlap.png")


def confusion_fig(path, name):
    cm = pd.read_csv(path, index_col=0)
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    ax.imshow(cm.values, cmap="Blues", vmin=0)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm.values):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 * cm.values.max() else INK)
    ax.set(xticks=range(cm.shape[1]), xticklabels=cm.columns, yticks=range(cm.shape[0]), yticklabels=cm.index,
           xlabel="predicted language", ylabel="true language")
    _save(fig, name)


def failures_fig(df):
    d = df[df["set"] == "synthetic"]
    if d.empty:
        return
    causes = ["localization", "rectification", "upside down", "same artwork", "lookalike", "heavy distortion", "other"]
    order = [m for m in ["orb", "sift", "resnet", "clip", "hybrid_clip_k20", "hybrid_clip+rot_k20"] if m in set(d["method"])]
    piv = d[d["method"].isin(order)].pivot_table(index="method", columns="cause", values="query_id",
                                                 aggfunc="count", fill_value=0).reindex(order)
    fig, ax = plt.subplots(figsize=(W, 2.6))
    left = np.zeros(len(order))
    for i, c in enumerate(causes):
        if c in piv.columns:
            ax.barh(range(len(order)), piv[c], left=left, color=PALETTE[i], label=c, edgecolor="white", linewidth=1)
            left += piv[c].values
    ax.set(yticks=range(len(order)), yticklabels=order, xlabel="Failed queries (all severities, 200 each)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    _save(fig, "failure_causes.png")


def condition_fig():
    p = config.RESULTS_DIR / "condition" / "validation_test_s1.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d["rectified"] == True]                                             # noqa: E712
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.plot([20, 80], [20, 80], color=MUTED, lw=0.8, ls="--")
    ax.scatter(d["true_left_pct"], d["est_left_pct"], s=10, color=PALETTE[0], marker="o", label="left/right",
               edgecolor="white", linewidth=0.4)
    ax.scatter(d["true_top_pct"], d["est_top_pct"], s=10, color=PALETTE[1], marker="s", label="top/bottom",
               edgecolor="white", linewidth=0.4)
    ax.set(xlabel="True centering (% of border on left / top)", ylabel="Estimated (%)", xlim=(15, 85), ylim=(15, 85))
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, "condition_centering.png")


def main():
    m = _read("summary_main.csv")
    if m is not None:
        accuracy_vs_severity(m)
        rotation_effect(m)
    k = _read("summary_ksweep.csv")
    if k is not None:
        ksweep(k, m)
    r = _read("summary_norectify.csv")
    if r is not None:
        rectification(r)
    s = _read("summary_scenes.csv")
    if s is not None:
        scenes_fig(s)
    for path, name in [(config.RESULTS_DIR / "language" / "confusion_v2_s2.csv", "language_confusion.png"),
                       (config.RESULTS_DIR / "language" / "confusion_s2.csv", "language_confusion_v1.png")]:
        if path.exists():
            confusion_fig(path, name)
    f = _read("failures.csv")
    if f is not None:
        failures_fig(f)
    condition_fig()


if __name__ == "__main__":
    main()
