#!/usr/bin/env python3
"""
Parameter sweep, but fast: run NLI once, then sweep everything in-memory.
"""
import subprocess, sys, os, time, hashlib, json, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ===== knobs (tweak these, not the rest) =====
PYTHON = sys.executable            # use current venv python
SCRIPT = "dax_bs_detector.py"      # your main script
MAX_DOCS = 400
NLI_BATCH = 32
USE_ZEROSHOT = True                # yes, we want NLI columns
LOAD_TINY = True                   # set False if you don't have tiny_clf.joblib
TINY_CLF_PATH = "tiny_clf.joblib"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# sweep grid: if quantile is set => ignore threshold. If quantile=None => use threshold.
FUSES = ["none", "avg", "and", "or"]
ALPHAS = [0.25, 0.5, 0.75]         # only used when fuse="avg"
QUANTILES = [0.6, 0.7, 0.8, 0.9, None]
THRESHOLDS = [0.30, 0.35, 0.40, 0.50]

# artifacts
OUTDIR = Path("sweep_runs")
OUTDIR.mkdir(exist_ok=True)

BASE_CSV = OUTDIR / "base_run.csv"      # the one-and-only heavy run
BASE_HTML = OUTDIR / "base_run.html"    # kept for reference
RESULTS_CSV = Path("sweep_results.csv")

# set this True if you also want per-combo CSV snapshots
SAVE_PER_COMBO_CSV = True

# ===== helpers =====
def _run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return p.stdout

def _hash_params(d: dict) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]

def _metrics(df: pd.DataFrame, score_col: str) -> dict:
    cols = df.columns.str.lower()
    m = {
        "rows": len(df),
        "bs_rate": float(df["label"].mean()) if "label" in cols else np.nan,
        "score_mean": float(df[score_col].mean()),
        "score_std": float(df[score_col].std(ddof=0)),
    }
    if "tiny_pred" in cols:
        tiny = df["tiny_pred"].astype(int)
        m["tiny_pos_rate"] = float(tiny.mean())
        if "label" in cols:
            m["agree_rate_vs_label"] = float((tiny == df["label"].astype(int)).mean())
        if "tiny_prob" in cols:
            try:
                m["corr_tinyprob_score"] = float(np.corrcoef(df["tiny_prob"], df[score_col])[0,1])
            except Exception:
                m["corr_tinyprob_score"] = np.nan
        # calibration-ish: how tiny positives climb in high-score buckets
        for q in [0.5, 0.7, 0.8, 0.9]:
            thr = df[score_col].quantile(q)
            sub = df[df[score_col] >= thr]
            m[f"cal_q{int(q*100)}_tiny_pos_rate"] = float(sub["tiny_pred"].mean()) if len(sub) else np.nan
    return m

def _ensure_base_run():
    """Run dax_bs_detector.py once to get the heavy stuff, then stash as base_run.csv/html."""
    if BASE_CSV.exists():
        print(f"[base] using cached {BASE_CSV}")
        return

    cmd = [PYTHON, SCRIPT, "--max_docs", str(MAX_DOCS)]
    if USE_ZEROSHOT:
        cmd += ["--zeroshot", "--nli_batch", str(NLI_BATCH)]
    # use a neutral fuse (we’ll recompute anyway); quantile just to make script happy
    cmd += ["--fuse", "none", "--quantile", "0.80"]
    if LOAD_TINY and Path(TINY_CLF_PATH).exists():
        cmd += ["--load_clf", TINY_CLF_PATH, "--embed_model", EMBED_MODEL]

    _ = _run_cmd(cmd)

    # stash artifacts
    src_csv = Path("dax_vagueness_scores.csv")
    src_html = Path("preview.html")
    if not src_csv.exists():
        raise FileNotFoundError("Expected dax_vagueness_scores.csv was not produced.")
    shutil.move(src_csv, BASE_CSV)
    if src_html.exists():
        shutil.move(src_html, BASE_HTML)
    print(f"[base] cached to {BASE_CSV}")

def _compute_score(df: pd.DataFrame, fuse: str, alpha: float | None) -> pd.Series:
    cols = df.columns
    has_nli = "nli_vague" in cols
    if fuse == "none":
        return df["vagueness_score"]
    if not has_nli:
        # no NLI available, bail to rules-only
        return df["vagueness_score"]
    if fuse == "avg":
        a = 0.5 if alpha is None else float(alpha)
        return a*df["vagueness_score"] + (1-a)*df["nli_vague"]
    if fuse == "and":
        return df[["vagueness_score","nli_vague"]].min(axis=1)
    if fuse == "or":
        return df[["vagueness_score","nli_vague"]].max(axis=1)
    # default
    return df["vagueness_score"]

def main():
    _ensure_base_run()
    base = pd.read_csv(BASE_CSV)

    # sanity checks
    need_cols = ["vagueness_score", "text"]
    for c in need_cols:
        if c not in base.columns:
            raise RuntimeError(f"Base CSV misses required column: {c}")
    if USE_ZEROSHOT and "nli_vague" not in base.columns:
        print("[warn] Base file has no nli_vague; sweeping will fall back to rules-only where needed.")

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

    rows = []
    pbar = tqdm(total=len(combos), desc="sweeping combos", leave=True)
    for params in combos:
        score = _compute_score(base, params["fuse"], params["alpha"])
        df = base.copy()
        score_col = "bs_score" if params["fuse"] != "none" and "nli_vague" in base.columns else "vagueness_score"
        if score_col == "bs_score":
            df["bs_score"] = score
        # thresholding
        if params["quantile"] is not None:
            thr = float(score.quantile(params["quantile"]))
        else:
            thr = float(params["threshold"])
        df["label"] = (score >= thr).astype(int)

        # metrics
        m = _metrics(df, score_col if score_col in df.columns else "vagueness_score")

        # save per-combo csv if you want to eyeball later
        run_id = f"{params['fuse']}-a{params['alpha']}-q{params['quantile']}-t{params['threshold']}-{_hash_params(params)}"
        if SAVE_PER_COMBO_CSV:
            rundir = OUTDIR / run_id
            rundir.mkdir(parents=True, exist_ok=True)
            df_out = df.copy()
            # keep file reasonably sized
            keep_cols = [c for c in df_out.columns if c in
                         ["company","domain","datatype","vagueness_score","nli_vague","bs_score",
                          "label","weasel_cnt","passive_cnt","tiny_pred","tiny_prob","text"]]
            df_out[keep_cols].to_csv(rundir / f"{run_id}.csv", index=False)

        rows.append({
            "run_id": run_id,
            "fuse": params["fuse"],
            "alpha": params["alpha"],
            "quantile": params["quantile"],
            "threshold": params["threshold"],
            **m
        })
        pbar.update(1)
    pbar.close()

    out = pd.DataFrame(rows)
    out.sort_values(["fuse","quantile","threshold","alpha"], inplace=True, na_position="last")
    out.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved: {RESULTS_CSV}")
    print(out[["run_id","fuse","alpha","quantile","threshold","bs_rate","score_mean",
               "agree_rate_vs_label","tiny_pos_rate"]].fillna("").to_string(index=False))

if __name__ == "__main__":
    main()
