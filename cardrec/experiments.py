"""Run the report's synthetic experiments and write one summary table per experiment.

    python -m cardrec.experiments --exp main        # every method x severity 1-5 (strict + lenient, 95% CI)
    python -m cardrec.experiments --exp ksweep      # hybrid shortlist size k (accuracy vs latency)
    python -m cardrec.experiments --exp norectify   # ablation: skip rectification (severity 3)
    python -m cardrec.experiments --exp scenes      # multi-card scenes, 3 overlap levels
    python -m cardrec.experiments --exp all

Query sets are generated on first use (N_QUERIES per severity, fixed seed). Every run writes its
per-query log to results/<exp>/<run>.csv; if that file already exists the run is NOT repeated
(the summary is recomputed from the log), so you can stop and restart at any time.
To redo a run, delete its log file (for scenes: the .json next to it). Summaries: results/summary_<exp>.csv
"""
import argparse
import json

import pandas as pd

from . import config
from .evaluate import evaluate_method, summarize
from .make_queries import make_queries
from .manifest import load_manifest
from .methods import clip_method, cnn, embedding, get_method, hybrid, sift  # noqa: F401
from .methods.cache import build_or_load
from .scenes import evaluate_scenes, make_scenes

# The method modules are imported above on purpose (normally get_method imports them on first use):
# a run takes hours, and loading every file at the start means that editing the code meanwhile
# cannot mix old and new versions inside one run.

N_QUERIES = 200
SEVERITIES = [1, 2, 3, 4, 5]
MAIN_METHODS = ["pixel", "orb", "sift", "resnet", "clip", "hybrid-resnet", "hybrid-clip",
                "resnet+rot", "clip+rot", "hybrid-resnet+rot", "hybrid-clip+rot"]
HYBRID_K = 20                       # shortlist size used in the main experiment
KS = [5, 10, 20, 50]
KSWEEP_SEVERITY = 3
SCENE_METHODS = ["resnet", "clip", "hybrid-resnet", "hybrid-clip"]
SCENE_OVERLAPS = [0.0, 0.15, 0.30]
N_SCENES, SCENE_SEVERITY = 100, 2


def query_set(manifest, severity):
    path = config.QUERIES_DIR / f"s{severity}" / "queries.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return make_queries(manifest, N_QUERIES, severity).astype(str)


def run(exp, run_name, method, queries, art_group, rectify_first=True):
    """One (method, query set) run, skipped if its log already exists."""
    log_path = config.RESULTS_DIR / exp / f"{run_name}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log = pd.read_csv(log_path, dtype={"query_id": str, "true_id": str, "pred_id": str})
        return summarize(log, method.name, method.index_size_bytes())
    summary, _ = evaluate_method(method, queries, rectify_first, log_path, art_group)
    return summary


def save(rows, exp):
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / f"summary_{exp}.csv", index=False)
    cols = [c for c in ["method", "k", "severity", "rectify", "overlap", "top1", "top1_lo", "top1_hi", "top5",
                        "mAP", "top1_len", "ms_total_median", "rectify_found", "rect_ok", "precision", "recall", "f1",
                        "e2e_top1", "e2e_top1_lo", "e2e_top1_hi"] if c in df.columns]
    print(df[cols].round(3).to_string(index=False))


def exp_main(manifest, groups):
    rows = []
    for name in MAIN_METHODS:
        m = build_or_load(get_method(name, k=HYBRID_K), manifest)
        for sev in SEVERITIES:
            s = run("main", f"{m.name}_s{sev}", m, query_set(manifest, sev), groups)
            rows.append({**s, "severity": sev})
            print(f"{m.name} s{sev}: top1={s['top1']:.3f}")
    save(rows, "main")


def exp_ksweep(manifest, groups):
    rows, q = [], query_set(manifest, KSWEEP_SEVERITY)
    for emb in ["resnet", "clip"]:
        m = build_or_load(get_method(f"hybrid-{emb}", k=KS[0]), manifest)
        for k in KS:
            m.k, m.name = k, f"hybrid_{emb}_k{k}"       # same index, only the shortlist size changes
            rows.append({**run("ksweep", f"{m.name}_s{KSWEEP_SEVERITY}", m, q, groups), "k": k,
                         "severity": KSWEEP_SEVERITY})
    save(rows, "ksweep")


def exp_norectify(manifest, groups):
    rows, q = [], query_set(manifest, 3)
    for name in MAIN_METHODS:
        m = build_or_load(get_method(name, k=HYBRID_K), manifest)
        for rect in (True, False):
            exp = "main" if rect else "norectify"       # the rectified run is the same as in "main"
            rows.append({**run(exp, f"{m.name}_s3", m, q, groups, rectify_first=rect), "severity": 3, "rectify": rect})
    save(rows, "norectify")


def exp_scenes(manifest, groups):
    rows = []
    for ov in SCENE_OVERLAPS:
        path = config.DATA_DIR / "scenes" / f"s{SCENE_SEVERITY}_o{int(100 * ov)}" / "scenes.json"
        if not path.exists():
            make_scenes(manifest, N_SCENES, SCENE_SEVERITY, ov)
        scenes = json.loads(path.read_text(encoding="utf-8"))
        for name in SCENE_METHODS:
            m = build_or_load(get_method(name, k=HYBRID_K), manifest)
            log_path = config.RESULTS_DIR / "scenes" / f"{m.name}_o{int(100 * ov)}.csv"
            summary_path = log_path.with_suffix(".json")     # false positives are not in the per-card log
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if summary_path.exists():
                s = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                s, _ = evaluate_scenes(m, scenes, groups, log_path)
                summary_path.write_text(json.dumps(s), encoding="utf-8")
            rows.append({**s, "overlap": ov, "severity": SCENE_SEVERITY})
    save(rows, "scenes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["main", "ksweep", "norectify", "scenes", "all"])
    a = ap.parse_args()
    manifest = load_manifest()
    groups = dict(zip(manifest["id"], manifest["art_group"]))
    todo = ["main", "ksweep", "norectify", "scenes"] if a.exp == "all" else [a.exp]
    for e in todo:
        print(f"===== {e} =====")
        globals()[f"exp_{e}"](manifest, groups)
