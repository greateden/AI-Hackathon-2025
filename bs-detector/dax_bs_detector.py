import argparse, json, re, pathlib, sys, csv
from dataclasses import dataclass
import pandas as pd
import kagglehub

# --- extra deps for the tiny supervised layer (kept optional) ---
# we import lazily later as well, but having them here helps type checkers chill
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
except Exception:
    LogisticRegression = None
try:
    import joblib
except Exception:
    joblib = None

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
    # normalize word-level counts into a cute 0..1 with a cap
    if total_words == 0:
        return 0.0
    return min(count / total_words, cap) / cap

@dataclass
class VaguenessOutput:
    score: float
    features: dict
    label: int   # 1=bs; 0=non-bs

def compute_vagueness_score(text: str, threshold: float = 0.6) -> VaguenessOutput:
    # rules-first, no drama
    t = text.lower()
    words = re.findall(r"\b\w[\w\-]*\b", t)
    n = len(words)

    weasel_cnt = len(re.findall(WEASEL, t))
    weak_cnt   = len(re.findall(WEAK_VERB, t))
    strong_cnt = len(re.findall(STRONG_VERB, t))
    passive_cnt= len(re.findall(PASSIVE, t))

    has_number = bool(re.search(HAS_NUMBER, t))
    has_units  = bool(re.search(UNITS, t))
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
    # paint it red; name and shame
    out = re.sub(WEASEL, lambda m: f'<span class="weasel">{m.group(0)}</span>', text, flags=re.IGNORECASE)
    out = re.sub(WEAK_VERB, lambda m: f'<span class="weasel">{m.group(0)}</span>', out, flags=re.IGNORECASE)
    out = re.sub(PASSIVE, lambda m: f'<span class="passive">{m.group(0)}</span>', out, flags=re.IGNORECASE)
    return out

# ---- dataset plumbing: locate + load the big CSV ----
def get_dataset_csv_path() -> pathlib.Path:
    path = kagglehub.dataset_download("equintel/dax-esg-media-dataset")
    root = pathlib.Path(path)
    hits = list(root.rglob("esg_documents_for_dax_companies.csv"))
    if hits:
        return hits[0]
    # fallback: any CSV if the expected one moves
    any_csv = list(root.rglob("*.csv"))
    if not any_csv:
        raise FileNotFoundError("No CSV files found in the Kaggle dataset.")
    return any_csv[0]

def load_samples(csv_path: pathlib.Path, max_docs: int) -> pd.DataFrame:
    # try pyarrow first (robust to long, multi-line cells)
    try:
        df = pd.read_csv(csv_path, sep="|", engine="pyarrow", dtype=str)
        print("[info] Loaded full CSV with pyarrow (pipe-delimited).")
    except Exception as e:
        print(f"[warn] pyarrow failed, fallback to python engine: {e}")
        # crank up csv field limit, some cells are thicc
        try:
            csv.field_size_limit(min(sys.maxsize, 10**9))
        except OverflowError:
            csv.field_size_limit(10**9)
        df = pd.read_csv(csv_path, sep="|", engine="python", dtype=str, on_bad_lines="skip")

    # drop the phantom first column if present
    drop_cols = [c for c in df.columns if (c is None) or (str(c).strip() == "") or str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if TEXT_COLUMN not in df.columns:
        raise RuntimeError(f"`{TEXT_COLUMN}` not found in columns: {df.columns.tolist()}")

    keep = ["company","content","datatype","date","domain","esg_topics","internal","symbol","title","url"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].rename(columns={"content":"text"}).dropna(subset=["text"])

    # sample for speed, we’re prototyping not doing a census
    if len(df) > max_docs:
        df = df.sample(n=max_docs, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df

# ---- zero-shot NLI on GPU (because, well, you have one) ----
def run_zeroshot_nli(texts, model_name="facebook/bart-large-mnli", batch_size=8):
    """Return three lists: nli_vague prob, nli_specific prob, nli_label (1=bs)."""
    from transformers import pipeline
    import torch

    # use GPU 0 if available; otherwise meh, CPU
    device = 0 if torch.cuda.is_available() else -1

    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    labels = ["vague and non-operational", "specific, measurable, and time-bound"]
    htemp = "This text is {}."

    n_vague, n_spec, n_lbl = [], [], []
    # process in chunks to avoid eating all VRAM at once
    for i in range(0, len(texts), batch_size):
        batch = [t if len(t) < 4000 else t[:4000] for t in texts[i:i+batch_size]]  # quick-n-dirty truncation
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
    return n_vague, n_spec, n_lbl

# ---- tiny supervised layer (embeddings + LR) ----
def _need(pkg, name):
    if pkg is None:
        raise RuntimeError(f"{name} not installed. Try: pip install -U {name}")

def _load_embedder(model_name: str):
    _need(SentenceTransformer, "sentence-transformers")
    print(f"[embed] loading {model_name} ...")
    return SentenceTransformer(model_name)

def _embed(embedder, texts, batch_size=64):
    # L2-normalized embeddings; LR likes it tidy
    embs = embedder.encode(
        list(texts), batch_size=batch_size,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return embs.astype("float32")

def _tiny_feats_from_row(row, rule_score: float, nli_vague: float | None):
    # cheap-but-effective tiny features
    f_has_number  = 1.0 if row.get("has_number", False) else 0.0
    f_has_deadln  = 1.0 if row.get("has_deadline", False) else 0.0
    f_has_owner   = 1.0 if row.get("has_owner", False) else 0.0
    f_nli         = float(nli_vague) if nli_vague is not None else 0.0
    f_rule        = float(rule_score)
    return np.array([f_rule, f_nli, f_has_number, f_has_deadln, f_has_owner], dtype="float32")

def train_tiny_supervised(seed_csv: str, embed_model: str, train_ratio: float, save_path: str = "tiny_clf.joblib"):
    _need(LogisticRegression, "scikit-learn"); _need(joblib, "joblib")
    seeds = pd.read_csv(seed_csv)
    if not {"text","y"}.issubset(seeds.columns):
        raise ValueError("seeds.csv must have columns: text,y (y in {0,1})")
    seeds["text"] = seeds["text"].astype(str)
    seeds["y"] = seeds["y"].astype(int)

    embedder = _load_embedder(embed_model)
    X_embed = _embed(embedder, seeds["text"].tolist())

    # compute rule/tiny feats on seeds; keep nli=0.0 here (fast path)
    tiny = []
    for t in seeds["text"]:
        r = compute_vagueness_score(t, threshold=0.5)
        tiny.append(_tiny_feats_from_row(r.features, r.score, nli_vague=0.0))
    X_tiny = np.vstack(tiny)
    X = np.hstack([X_embed, X_tiny])
    y = seeds["y"].values

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, train_size=train_ratio, random_state=42, stratify=y)
    clf = LogisticRegression(
        solver="saga", penalty="l2", C=1.0, max_iter=2000,
        class_weight="balanced", n_jobs=-1, verbose=0
    )
    clf.fit(X_tr, y_tr)
    print("[tiny] trained logistic regression.")
    y_hat = clf.predict(X_va)
    print("[tiny] validation report:\n", classification_report(y_va, y_hat, digits=3))

    artifact = {"clf": clf, "embed_model": embed_model, "embed_dim": X_embed.shape[1]}
    joblib.dump(artifact, save_path)
    print(f"[tiny] saved model to {save_path}")

def predict_with_tiny(texts, model_path: str, embed_model: str | None):
    _need(joblib, "joblib")
    art = joblib.load(model_path)
    use_model = embed_model or art.get("embed_model")
    if use_model is None:
        raise ValueError("missing embed_model; pass --embed_model or retrain.")
    embedder = _load_embedder(use_model)
    X_embed = _embed(embedder, texts)

    # build tiny rule features on the fly (nli left as 0.0 unless you wire it in)
    tiny = []
    for t in texts:
        r = compute_vagueness_score(t, threshold=0.5)
        tiny.append(_tiny_feats_from_row(r.features, r.score, nli_vague=0.0))
    X_tiny = np.vstack(tiny)
    X = np.hstack([X_embed, X_tiny])

    clf = art["clf"]
    proba = clf.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return proba, preds

# ---- HTML renderer: sortable top hits + cute summary ----
def build_preview_html(df_out: pd.DataFrame, threshold: float, top_n: int = 50):
    has_domain = "domain" in df_out.columns
    has_dtype  = "datatype" in df_out.columns
    has_nli    = "nli_vague" in df_out.columns
    has_fuse   = "bs_score" in df_out.columns
    has_tiny   = "tiny_prob" in df_out.columns

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
    if has_tiny:  headers += ["tiny_prob","tiny_pred"]
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
            tds += [f"{row.get('nli_vague',0.0):.3f}", str(row.get("nli_label",""))]
        if has_fuse:
            tds.append(f"{row.get('bs_score',0.0):.3f}")
        if has_tiny:
            tds += [f"{row.get('tiny_prob',0.0):.3f}", str(row.get("tiny_pred",""))]
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

# ---- main – bring the chaos together ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_docs", type=int, default=300, help="how many docs to sample")
    ap.add_argument("--threshold", type=float, default=0.6, help=">= threshold => label=1 (bs)")
    ap.add_argument("--quantile", type=float, default=None, help="auto threshold via quantile, e.g., 0.8 => top20% is 1")
    ap.add_argument("--reports_only", action="store_true", help="filter datatype=report only")

    # zero-shot knobs
    ap.add_argument("--zeroshot", action="store_true", help="run zero-shot NLI baseline")
    ap.add_argument("--nli_model", type=str, default="facebook/bart-large-mnli",
                    help="e.g., facebook/bart-large-mnli or MoritzLaurer/mDeBERTa-v3-base-xnli-multinli")
    ap.add_argument("--nli_batch", type=int, default=16, help="batch size for NLI (GPU loves bigger)")
    ap.add_argument("--fuse", choices=["none","avg","and","or"], default="none",
                    help="combine rules+NLI: none/avg/and/or")
    ap.add_argument("--alpha", type=float, default=0.5, help="when fuse=avg: alpha*rules + (1-alpha)*NLI")

    # tiny supervised layer knobs
    ap.add_argument("--supervised", action="store_true", help="train the tiny logistic layer from seed labels")
    ap.add_argument("--seed_csv", type=str, default="seeds.csv", help="CSV with columns: text,y (y in {0,1})")
    ap.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5",
                    help="Sentence embedding model, e.g., BAAI/bge-small-en-v1.5 or intfloat/e5-base-v2")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="train split ratio for seed labels")
    ap.add_argument("--save_clf", type=str, default="tiny_clf.joblib", help="where to save the trained tiny classifier")
    ap.add_argument("--load_clf", type=str, default=None, help="load a saved tiny classifier for inference")

    # single-text quick classify (rules/NLI/tiny)
    ap.add_argument("--text", type=str, default=None, help="classify a single text instead of dataset")

    args = ap.parse_args()

    # single-text path: no dataset IO, just run and print
    if args.text is not None:
        t = args.text.strip()
        r = compute_vagueness_score(t, threshold=args.threshold)
        res = {"vagueness_score": r.score, "features": r.features, "rule_label": r.label}
        # optional NLI
        if args.zeroshot:
            n_v, n_s, n_l = run_zeroshot_nli([t], model_name=args.nli_model, batch_size=1)
            res.update({"nli_vague": n_v[0], "nli_label": n_l[0]})
        # optional tiny
        if args.load_clf:
            probs, preds = predict_with_tiny([t], model_path=args.load_clf, embed_model=args.embed_model)
            res.update({"tiny_prob": float(probs[0]), "tiny_pred": int(preds[0])})
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # dataset path
    csv_path = get_dataset_csv_path()
    print("CSV:", csv_path)
    df = load_samples(csv_path, max_docs=args.max_docs)
    if args.reports_only and "datatype" in df.columns:
        df = df[df["datatype"].str.lower().eq("report")]

    # rules pass
    scores, feats = [], []
    for txt in df["text"].astype(str):
        r = compute_vagueness_score(txt, threshold=args.threshold)
        scores.append(r.score); feats.append(r.features)

    # zero-shot pass (optional)
    nli_vague = nli_spec = nli_label = []
    if args.zeroshot:
        print(f"[nli] running zero-shot on {args.nli_model} ...")
        texts_list = df["text"].astype(str).tolist()
        nli_vague, nli_spec, nli_label = run_zeroshot_nli(
            texts_list, model_name=args.nli_model, batch_size=args.nli_batch
        )

    # merge columns
    parts = [
        df.reset_index(drop=True),
        pd.Series(scores, name="vagueness_score"),
        pd.DataFrame(feats),
    ]
    if args.zeroshot:
        parts += [
            pd.Series(nli_vague, name="nli_vague"),
            pd.Series(nli_spec,  name="nli_specific"),
            pd.Series(nli_label, name="nli_label"),
        ]
    out = pd.concat(parts, axis=1)

    # fusion (rules + NLI)
    if args.zeroshot and len(out) and args.fuse != "none":
        if args.fuse == "avg":
            out["bs_score"] = args.alpha*out["vagueness_score"] + (1-args.alpha)*out["nli_vague"]
        elif args.fuse == "and":
            out["bs_score"] = out[["vagueness_score","nli_vague"]].min(axis=1)
        elif args.fuse == "or":
            out["bs_score"] = out[["vagueness_score","nli_vague"]].max(axis=1)
        score_col = "bs_score"
    else:
        score_col = "vagueness_score"

    # order by dodginess
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)

    # thresholding (quantile wins if provided)
    thr = args.threshold
    if args.quantile is not None:
        thr = float(out[score_col].quantile(args.quantile))
        print(f"[auto] threshold(from quantile={args.quantile}) = {thr:.3f}")
    out["label"] = (out[score_col] >= thr).astype(int)

    # === tiny supervised: train or predict, attach to out ===
    if args.supervised:
        try:
            train_tiny_supervised(seed_csv=args.seed_csv, embed_model=args.embed_model,
                                  train_ratio=args.train_ratio, save_path=args.save_clf)
        except Exception as e:
            print(f"[tiny] training failed: {e}")

    if args.load_clf:
        try:
            probs, preds = predict_with_tiny(out["text"].astype(str).tolist(),
                                             model_path=args.load_clf, embed_model=args.embed_model)
            out["tiny_prob"] = probs
            out["tiny_pred"] = preds
            print("[tiny] attached tiny layer outputs to dataframe.")
        except Exception as e:
            print(f"[tiny] inference failed: {e}")

    # dump files + pretty page
    out.to_csv("dax_vagueness_scores.csv", index=False, encoding="utf-8")
    print(f"Saved: dax_vagueness_scores.csv (rows={len(out)})")
    build_preview_html(out, threshold=thr, top_n=50)
    metric_col = "bs_score" if "bs_score" in out.columns else "vagueness_score"
    print(f"BS rate: {out['label'].mean():.1%} | mean {metric_col}: {out[metric_col].mean():.3f} | threshold: {thr:.2f}")

if __name__ == "__main__":
    main()