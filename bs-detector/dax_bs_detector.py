#!/usr/bin/env python3
import argparse, json, re, pathlib, sys, csv, os, signal
from dataclasses import dataclass
import pandas as pd
import kagglehub

# progress toys
from tqdm import tqdm

# ---- config knobs (keep it boring, ship it) ----
TEXT_COLUMN = "content"
KEEP_COLS = ["company","datatype","date","domain","esg_topics","symbol","title","url","internal","source"]

# ---- regex zoo: weasel/hedge/verbs/etc ----
WEASEL = r"\b(aim|aspire|strive|explore|consider|intend|plan to|promote|encourage|support|potentially|may|might|could|should|believe|likely|approximately|around|some|several|various|significant|robust|world[- ]class|best[- ]in[- ]class|commitment|vision|journey|ongoing|progress|continue|enhance)\b"
WEAK_VERB = r"\b(commit to|work towards|align with|raise awareness|foster|enable|drive|advance)\b"
STRONG_VERB = r"\b(reduce|cut|install|retrofit|phase out|electrify|switch to|invest|audit|publish|ban|replace|deploy|measure|report|verify|purchase|offset|abate)\b"
PASSIVE = r"\b(is|are|was|were|been|being)\s+\w+ed\b"
HAS_NUMBER = r"(\b\d+(\.\d+)?\b|%)"
UNITS = r"\b(tco2e|co2e|mwh|kwh|km|tonnes?|t|€|\bnz\$\b|\$)\b"
DEADLINE = r"\b(by\s+(20\d\d|Q[1-4]|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})|within\s+\d+\s+(day|week|month|year)s?|no later than|by end of)\b"
OWNER = r"\b(ceo|cfo|board|operations|sustainability\s+team|procurement|plant\s+\w+|facility\s+\w+|department|committee|supplier[s]?)\b"
ORG_TAIL = r"\b(inc|ltd|plc|ag|gmbh|corp|company)\b"

def _ratio(count, total_words, cap=0.2):
    if total_words == 0:
        return 0.0
    return min(count / total_words, cap) / cap

@dataclass
class VaguenessOutput:
    score: float
    features: dict
    label: int   # 1=bs; 0=non-bs

def compute_vagueness_score(text: str, threshold: float = 0.6) -> VaguenessOutput:
    t = text.lower()
    words = re.findall(r"\b\w[\w\-]*\b", t)
    n = len(words)

    weasel_cnt = len(re.findall(WEASEL, t))
    weak_cnt   = len(re.findall(WEAK_VERB, t))
    strong_cnt = len(re.findall(STRONG_VERB, t))
    passive_cnt= len(re.findall(PASSIVE, t))

    has_number = bool(re.search(HAS_NUMBER, t))
    has_units  = bool(research := re.search(UNITS, t))
    has_deadln = bool(re.search(DEADLINE, t))
    has_owner  = bool(re.search(OWNER, t) or re.search(r"\b[A-Z][A-Za-z&\-]+ " + ORG_TAIL, text))

    f_weasel   = _ratio(weasel_cnt, n)
    f_weak     = _ratio(weak_cnt, n)
    f_passive  = _ratio(passive_cnt, n)
    f_strong   = _ratio(strong_cnt, n)

    w = {
        "weasel":0.30, "weak":0.12, "passive":0.10,
        "miss_number":0.22, "miss_deadline":0.18, "miss_owner":0.18,
        "strong":-0.10, "measurables":-0.10
    }
    miss_number = 0.0 if has_number else 1.0
    miss_deadln = 0.0 if has_deadln else 1.0
    miss_owner  = 0.0 if has_owner else 1.0
    measurables = 1.0 if has_units else 0.0

    score = (w["weasel"]*f_weasel + w["weak"]*f_weak + w["passive"]*f_passive +
             w["miss_number"]*miss_number + w["miss_deadline"]*miss_deadln +
             w["miss_owner"]*miss_owner + w["strong"]*f_strong + w["measurables"]*measurables)
    score = max(0.0, min(1.0, score))
    label = 1 if score >= threshold else 0

    return VaguenessOutput(
        round(score,3),
        {
            "weasel_cnt":weasel_cnt, "weak_cnt":weak_cnt,
            "strong_cnt":strong_cnt, "passive_cnt":passive_cnt,
            "has_number":has_number, "has_units":has_units,
            "has_deadline":has_deadln, "has_owner":has_owner
        },
        label
    )

def highlight_weasels_html(text: str) -> str:
    out = re.sub(WEASEL, lambda m: f'<span class="weasel">{m.group(0)}</span>', text, flags=re.IGNORECASE)
    out = re.sub(WEAK_VERB, lambda m: f'<span class="weasel">{m.group(0)}</span>', out, flags=re.IGNORECASE)
    out = re.sub(PASSIVE, lambda m: f'<span class="passive">{m.group(0)}</span>', out, flags=re.IGNORECASE)
    return out

# ---- dataset plumbing ----
def get_dataset_csv_path() -> pathlib.Path:
    path = kagglehub.dataset_download("equintel/dax-esg-media-dataset")
    root = pathlib.Path(path)
    hits = list(root.rglob("esg_documents_for_dax_companies.csv"))
    if hits:
        return hits[0]
    any_csv = list(root.rglob("*.csv"))
    if not any_csv:
        raise FileNotFoundError("No CSV files found in the Kaggle dataset.")
    return any_csv[0]

def load_samples(csv_path: pathlib.Path, max_docs: int) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep="|", engine="pyarrow", dtype=str)
        print("[info] Loaded full CSV with pyarrow (pipe-delimited).")
    except Exception as e:
        print(f"[warn] pyarrow failed, fallback to python engine: {e}")
        try:
            csv.field_size_limit(min(sys.maxsize, 10**9))
        except OverflowError:
            csv.field_size_limit(10**9)
        df = pd.read_csv(csv_path, sep="|", engine="python", dtype=str, on_bad_lines="skip")

    drop_cols = [c for c in df.columns if (c is None) or (str(c).strip() == "") or str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if TEXT_COLUMN not in df.columns:
        raise RuntimeError(f"`{TEXT_COLUMN}` not found in columns: {df.columns.tolist()}")

    keep = ["company","content","datatype","date","domain","esg_topics","internal","symbol","title","url"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].rename(columns={"content":"text"}).dropna(subset=["text"])

    if len(df) > max_docs:
        df = df.sample(n=max_docs, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df

# ---- zero-shot NLI with visible progress ----
def run_zeroshot_nli(texts, model_name="facebook/bart-large-mnli", batch_size=16):
    """Return three lists: nli_vague prob, nli_specific prob, nli_label (1=bs)."""
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification", model=model_name, device=device)

    labels = ["vague and non-operational", "specific, measurable, and time-bound"]
    htemp = "This text is {}."

    n_vague, n_spec, n_lbl = [], [], []
    total = (len(texts) + batch_size - 1) // batch_size
    pbar = tqdm(total=total, desc="[NLI] batches", leave=True)
    for i in range(0, len(texts), batch_size):
        batch = [t if len(t) < 4000 else t[:4000] for t in texts[i:i+batch_size]]
        out = clf(
            batch,
            candidate_labels=labels,
            hypothesis_template=htemp,
            multi_label=False,
            truncation=True,
            max_length=512
        )
        if isinstance(out, dict):
            out = [out]
        for r in out:
            mapping = {lab: score for lab, score in zip(r["labels"], r["scores"])}
            pv = float(mapping.get(labels[0], 0.0))
            ps = float(mapping.get(labels[1], 0.0))
            n_vague.append(pv)
            n_spec.append(ps)
            n_lbl.append(1 if pv >= ps else 0)
        pbar.update(1)
    pbar.close()
    return n_vague, n_spec, n_lbl

# ---- HTML renderer ----
def build_preview_html(df_out: pd.DataFrame, threshold: float, top_n: int = 50):
    has_domain = "domain" in df_out.columns
    has_dtype  = "datatype" in df_out.columns
    has_nli    = "nli_vague" in df_out.columns
    has_fuse   = "bs_score" in df_out.columns

    parts = []
    if has_dtype:
        by_type = df_out.groupby("datatype").agg(
            mean_score=("vagueness_score","mean"),
            bs_rate=("label","mean"),
            n=("label","size"),
        ).sort_values("mean_score", ascending=False).round(3)
        parts.append("<p><b>By datatype</b></p>" + by_type.to_html(escape=False))
    if has_domain:
        by_domain = df_out.groupby("domain").agg(
            mean_score=("vagueness_score","mean"),
            bs_rate=("label","mean"),
            n=("label","size"),
        ).sort_values("mean_score", ascending=False).head(10).round(3)
        parts.append("<p><b>Top domains by score</b></p>" + by_domain.to_html(escape=False))
    summary_html = "<h3>Summary</h3>" + ("".join(parts) if parts else "<p>(no grouping columns found)</p>")

    headers = ["#", "company"]
    if has_domain: headers.append("domain")
    if has_dtype:  headers.append("datatype")
    headers += ["vagueness_score"]
    if has_nli:   headers += ["nli_vague","nli_label"]
    if has_fuse:  headers += ["bs_score"]
    headers += ["label","weasel_cnt","passive_cnt","text (highlighted)"]

    head = df_out.head(top_n)
    rows_html = []
    for i, row in head.iterrows():
        html_txt = highlight_weasels_html(str(row.get("text","")))
        tds = [str(i), str(row.get("company",""))]
        if has_domain: tds.append(str(row.get("domain","")))
        if has_dtype:  tds.append(str(row.get("datatype","")))
        tds.append(f"{row['vagueness_score']:.3f}")
        if has_nli:
            tds += [f"{row['nli_vague']:.3f}", str(row.get("nli_label",""))]
        if has_fuse:
            tds.append(f"{row['bs_score']:.3f}")
        tds += [str(row.get("label","")), str(row.get("weasel_cnt","")), str(row.get("passive_cnt","")), f'<div class="txt">{html_txt}</div>']
        rows_html.append("<tr>" + "".join(f"<td>{x}</td>" for x in tds) + "</tr>")

    table_html = f"""
    <table>
      <tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr>
      {''.join(rows_html)}
    </table>
    """

    template = f"""
    <html><head>
      <meta charset="utf-8"/>
      <style>
        body {{ font-family:-apple-system, Segoe UI, Roboto, Arial; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
        th {{ background:#f6f6f6; position: sticky; top: 0; }}
        .weasel {{ background:#ffd5d5; color:#900; font-weight:600; }}
        .passive {{ background:#ffe9c9; }}
        .txt {{ max-width: 1000px; }}
        .meta {{ color:#555; }}
      </style>
    </head><body>
      <h2>DAX ESG — Bullshit/Vagueness Preview</h2>
      <p class="meta">Threshold = {threshold:.2f} (label 1 = bullshit) | Rows = {len(df_out)}</p>
      {summary_html}
      {table_html}
    </body></html>
    """
    with open("preview.html","w",encoding="utf-8") as f:
        f.write(template)
    print("Saved: preview.html")

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_docs", type=int, default=300, help="how many docs to sample")
    ap.add_argument("--threshold", type=float, default=0.6, help=">= threshold => label=1 (bs)")
    ap.add_argument("--quantile", type=float, default=None, help="auto threshold via quantile")
    ap.add_argument("--reports_only", action="store_true", help="filter datatype=report only")

    # zero-shot knobs
    ap.add_argument("--zeroshot", action="store_true", help="run zero-shot NLI baseline")
    ap.add_argument("--nli_model", type=str, default="facebook/bart-large-mnli")
    ap.add_argument("--nli_batch", type=int, default=16)
    ap.add_argument("--fuse", choices=["none","avg","and","or"], default="none")
    ap.add_argument("--alpha", type=float, default=0.5)

    # tiny supervised layer
    ap.add_argument("--supervised", action="store_true", help="train tiny layer from seeds")
    ap.add_argument("--seed_csv", type=str, default=None)
    ap.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--train_ratio", type=float, default=0.85)
    ap.add_argument("--save_clf", type=str, default=None)
    ap.add_argument("--load_clf", type=str, default=None)

    args = ap.parse_args()

    # graceful partial-save on Ctrl-C
    interrupted = {"flag": False}
    def _sigint_handler(signum, frame):
        interrupted["flag"] = True
        print("\n[warn] SIGINT received; will save partial artifacts…")
    signal.signal(signal.SIGINT, _sigint_handler)

    csv_path = get_dataset_csv_path()
    print("CSV:", csv_path)
    df = load_samples(csv_path, max_docs=args.max_docs)
    if args.reports_only and "datatype" in df.columns:
        df = df[df["datatype"].str.lower().eq("report")].reset_index(drop=True)

    # rules pass with progress
    scores, labels, feats = [], [], []
    for txt in tqdm(df["text"].astype(str), desc="[rules] scoring", leave=False):
        r = compute_vagueness_score(txt, threshold=args.threshold)
        scores.append(r.score); labels.append(r.label); feats.append(r.features)
        if interrupted["flag"]:
            break

    parts = [
        df.iloc[:len(scores)].reset_index(drop=True),
        pd.Series(scores, name="vagueness_score"),
        pd.DataFrame(feats),
    ]

    # zero-shot pass
    if args.zeroshot and not interrupted["flag"]:
        print(f"[nli] running zero-shot on {args.nli_model} ...")
        texts_list = df["text"].astype(str).tolist()[:len(scores)]
        nli_vague, nli_spec, nli_label = run_zeroshot_nli(
            texts_list, model_name=args.nli_model, batch_size=args.nli_batch
        )
        parts += [
            pd.Series(nli_vague, name="nli_vague"),
            pd.Series(nli_spec,  name="nli_specific"),
            pd.Series(nli_label, name="nli_label"),
        ]

    out = pd.concat(parts, axis=1)

    # tiny supervised layer (optional)
    if args.supervised and not interrupted["flag"]:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        import joblib
        import torch

        if args.seed_csv is None:
            raise RuntimeError("--supervised requires --seed_csv")

        # --- robust seed loader: handle comma/pipe + y/label ---
        # Use Python engine to allow sep=None (sniffer) and weird quotes.
        seeds = pd.read_csv(args.seed_csv, sep=None, engine="python", dtype=str)
        # unify column names
        cols = {c.strip().lower(): c for c in seeds.columns}
        if "label" in cols:
            labcol = cols["label"]
        elif "y" in cols:
            labcol = cols["y"]
            seeds = seeds.rename(columns={labcol: "label"})
            labcol = "label"
        else:
            raise RuntimeError(f"seeds CSV must have a 'label' or 'y' column, got: {list(seeds.columns)}")

        if "text" not in seeds.columns:
            # try a few common names
            cand = [c for c in seeds.columns if c.strip().lower() in ("text","sentence","content")]
            if not cand:
                raise RuntimeError(f"seeds CSV must have a 'text' column, got: {list(seeds.columns)}")
            seeds = seeds.rename(columns={cand[0]: "text"})

        seeds = seeds.dropna(subset=["text", "label"]).copy()
        # coerce to 0/1 ints
        seeds["label"] = seeds["label"].astype(str).str.extract(r"(\d)").astype(int)
        # sanity prints
        print(f"[seeds] loaded {len(seeds)} rows "
              f"(pos={int((seeds['label']==1).sum())}, neg={int((seeds['label']==0).sum())})")

        # --- embeddings device ---
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embed] loading {args.embed_model} on {device_str} ...")
        enc = SentenceTransformer(args.embed_model, device=device_str)

        X_texts = seeds["text"].astype(str).tolist()
        X_emb = enc.encode(
            X_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )

        # cheap rule features to concatenate
        cheap = []
        for t in tqdm(X_texts, desc="[tiny] rules feats for seeds", leave=False):
            v = compute_vagueness_score(t, threshold=args.threshold)
            cheap.append([
                v.score,
                int(v.features["has_number"]),
                int(v.features["has_deadline"]),
                int(v.features["has_owner"]),
            ])

        import numpy as np
        X = np.hstack([X_emb, np.array(cheap, dtype=float)])
        y = seeds["label"].values

        Xtr, Xva, ytr, yva = train_test_split(
            X, y, train_size=args.train_ratio, stratify=y, random_state=42
        )
        clf = LogisticRegression(max_iter=1000, n_jobs=1)
        clf.fit(Xtr, ytr)
        print("[tiny] trained logistic regression.")
        ypred = clf.predict(Xva)
        print("[tiny] validation report:\n", classification_report(yva, ypred, digits=3))

        if args.save_clf:
            joblib.dump({"clf": clf, "embed": args.embed_model, "feat_dim": X.shape[1]}, args.save_clf)
            print(f"[tiny] saved model to {args.save_clf}")

    if args.load_clf and not interrupted["flag"]:
        import joblib, numpy as np, torch
        from sentence_transformers import SentenceTransformer

        bundle = joblib.load(args.load_clf)
        clf = bundle["clf"]; embed_name = bundle.get("embed", args.embed_model)
        expect_dim = bundle.get("feat_dim", None)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        enc = SentenceTransformer(embed_name, device=device_str)

        texts_eval = out["text"].astype(str).tolist()
        X_emb = enc.encode(texts_eval, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

        cheap = []
        for t in tqdm(texts_eval, desc="[tiny] rules feats for eval", leave=False):
            v = compute_vagueness_score(t, threshold=args.threshold)
            cheap.append([
                v.score,
                int(v.features["has_number"]),
                int(v.features["has_deadline"]),
                int(v.features["has_owner"]),
            ])
        X = np.hstack([X_emb, np.array(cheap, dtype=float)])

        if expect_dim is not None and X.shape[1] != expect_dim:
            raise ValueError(f"[tiny] feature dim mismatch: got {X.shape[1]}, expect {expect_dim}. "
                             f"Did you change embed model or cheap features?")

        probs = clf.predict_proba(X)[:, 1]
        out["tiny_prob"] = probs
        out["tiny_pred"] = (probs >= 0.5).astype(int)
        print("[tiny] attached tiny layer outputs to dataframe.")
    # fusion + thresholding
    score_col = "vagueness_score"
    if args.zeroshot and "nli_vague" in out.columns and args.fuse != "none":
        if args.fuse == "avg":
            out["bs_score"] = args.alpha*out["vagueness_score"] + (1-args.alpha)*out["nli_vague"]
        elif args.fuse == "and":
            out["bs_score"] = out[["vagueness_score","nli_vague"]].min(axis=1)
        elif args.fuse == "or":
            out["bs_score"] = out[["vagueness_score","nli_vague"]].max(axis=1)
        score_col = "bs_score"

    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)

    thr = args.threshold
    if args.quantile is not None and len(out):
        thr = float(out[score_col].quantile(args.quantile))
        print(f"[auto] threshold(from quantile={args.quantile}) = {thr:.3f}")
    out["label"] = (out[score_col] >= thr).astype(int)

    # write asap so you can peek while it runs
    tmp_csv = "dax_vagueness_scores.partial.csv"
    out.to_csv(tmp_csv, index=False, encoding="utf-8")
    os.replace(tmp_csv, "dax_vagueness_scores.csv")
    print(f"Saved: dax_vagueness_scores.csv (rows={len(out)})")

    build_preview_html(out, threshold=thr, top_n=50)
    print(f"BS rate: {out['label'].mean():.1%} | mean {score_col}: {out[score_col].mean():.3f} | threshold: {thr:.2f}")

if __name__ == "__main__":
    main()
