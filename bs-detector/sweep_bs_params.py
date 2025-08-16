#!/usr/bin/env python3
"""
Fast sweep with live progress + streaming CSV writes.
One heavy base run, then everything in-memory. Each combo appended to sweep_results.csv.
"""
import subprocess, sys, os, time, hashlib, json, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

PYTHON = sys.executable
SCRIPT = "dax_bs_detector.py"
MAX_DOCS = 400
NLI_BATCH = 32
USE_ZEROSHOT = True
LOAD_TINY = True
TINY_CLF_PATH = "tiny_clf.joblib"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

FUSES = ["none", "avg", "and", "or"]
ALPHAS = [0.25, 0.5, 0.75]
QUANTILES = [0.6, 0.7, 0.8, 0.9, None]
THRESHOLDS = [0.30, 0.35, 0.40, 0.50]

OUTDIR = Path("sweep_runs"); OUTDIR.mkdir(exist_ok=True)
BASE_CSV = OUTDIR / "base_run.csv"
BASE_HTML = OUTDIR / "base_run.html"
RESULTS_CSV = Path("sweep_results.csv")
SAVE_PER_COMBO_CSV = True

def _run_cmd(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    # stream stdout live so you see NLI batches moving
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")

def _hash_params(d: dict) -> str:
    s = json.dumps(d, sort_keys=True)
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:8]

def _metrics(df: pd.DataFrame, score_col: str) -> dict:
    m = {
        "rows": len(df),
        "bs_rate": float(df["label"].mean()) if "label" in df.columns else np.nan,
        "score_mean": float(df[score_col].mean()),
        "score_std": float(df[score_col].std(ddof=0)),
    }
    if "tiny_pred" in df.columns:
        tiny = df["tiny_pred"].astype(int)
        m["tiny_pos_rate"] = float(tiny.mean())
        if "label" in df.columns:
            m["agree_rate_vs_label"] = float((tiny == df["label"].astype(int)).mean())
        if "tiny_prob" in df.columns:
            try:
                m["corr_tinyprob_score"] = float(np.corrcoef(df["tiny_prob"], df[score_col])[0,1])
            except Exception:
                m["corr_tinyprob_score"] = np.nan
        for q in [0.5, 0.7, 0.8, 0.9]:
            thr = df[score_col].quantile(q)
            sub = df[df[score_col] >= thr]
            m[f"cal_q{int(q*100)}_tiny_pos_rate"] = float(sub["tiny_pred"].mean()) if len(sub) else np.nan
    return m

def _ensure_base_run():
    if BASE_CSV.exists():
        print(f"[base] using cached {BASE_CSV}")
        return
    cmd = [PYTHON, SCRIPT, "--max_docs", str(MAX_DOCS)]
    if USE_ZEROSHOT:
        cmd += ["--zeroshot", "--nli_batch", str(NLI_BATCH)]
    cmd += ["--fuse", "none", "--quantile", "0.80"]
    if LOAD_TINY and Path(TINY_CLF_PATH).exists():
        cmd += ["--load_clf", TINY_CLF_PATH, "--embed_model", EMBED_MODEL]
    _run_cmd(cmd)
    src_csv = Path("dax_vagueness_scores.csv"); src_html = Path("preview.html")
    if not src_csv.exists():
        raise FileNotFoundError("Expected dax_vagueness_scores.csv was not produced.")
    shutil.move(src_csv, BASE_CSV)
    if src_html.exists():
        shutil.move(src_html, BASE_HTML)
    print(f"[base] cached to {BASE_CSV}")

def _compute_score(df: pd.DataFrame, fuse: str, alpha: float | None) -> pd.Series:
    if fuse == "none" or "nli_vague" not in df.columns:
        return df["vagueness_score"]
    if fuse == "avg":
        a = 0.5 if alpha is None else float(alpha)
        return a*df["vagueness_score"] + (1-a)*df["nli_vague"]
    if fuse == "and":
        return df[["vagueness_score","nli_vague"]].min(axis=1)
    if fuse == "or":
        return df[["vagueness_score","nli_vague"]].max(axis=1)
    return df["vagueness_score"]

def _append_row(path: Path, row: dict):
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)

def main():
    _ensure_base_run()
    base = pd.read_csv(BASE_CSV)

    # build combo grid
    combos = []
    for fuse in FUSES:
        alphas = ALPHAS if fuse == "avg" else [None]
        for alpha in alphas:
            for q in QUANTILES:
                if q is None:
                    for thr in THRESHOLDS:
                        combos.append({"fuse": fuse, "alpha": alpha, "quantile": None, "threshold": thr})
                else:
                    combos.append({"fuse": fuse, "alpha": alpha, "quantile": q, "threshold": None})

    pbar = tqdm(total=len(combos), desc="sweeping combos", leave=True)
    for idx, params in enumerate(combos, 1):
        score = _compute_score(base, params["fuse"], params["alpha"])
        df = base.copy()
        score_col = "bs_score" if (params["fuse"] != "none" and "nli_vague" in base.columns) else "vagueness_score"
        if score_col == "bs_score":
            df["bs_score"] = score

        # thresholding (quantile wins)
        if params["quantile"] is not None:
            thr = float(score.quantile(params["quantile"]))
        else:
            thr = float(params["threshold"])
        df["label"] = (score >= thr).astype(int)

        # metrics + streaming write
        m = _metrics(df, score_col)
        run_id = f"{params['fuse']}-a{params['alpha']}-q{params['quantile']}-t{params['threshold']}-{_hash_params(params)}"
        outrow = {
            "run_id": run_id, **params,
            **m
        }
        _append_row(RESULTS_CSV, outrow)

        # small snapshot per combo (optional)
        if SAVE_PER_COMBO_CSV:
            rundir = OUTDIR / run_id
            rundir.mkdir(parents=True, exist_ok=True)
            keep_cols = [c for c in df.columns if c in
                        ["company","domain","datatype","vagueness_score","nli_vague","bs_score",
                         "label","weasel_cnt","passive_cnt","tiny_pred","tiny_prob","text"]]
            df[keep_cols].to_csv(rundir / f"{run_id}.csv", index=False)

        # live status on the bar
        pbar.set_postfix({
            "run": idx,
            "fuse": params["fuse"],
            "q": params["quantile"] if params["quantile"] is not None else "-",
            "thr": f"{thr:.3f}",
            "bs_rate": f"{m['bs_rate']:.2%}" if not np.isnan(m["bs_rate"]) else "NA"
        })
        pbar.update(1)
    pbar.close()

    # final echo
    print(f"\nSaved: {RESULTS_CSV}")
    try:
        last = pd.read_csv(RESULTS_CSV)
        show = last[["run_id","fuse","alpha","quantile","threshold","bs_rate","score_mean"]]
        print(show.tail(10).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
