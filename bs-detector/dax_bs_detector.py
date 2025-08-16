import argparse, json, re, pathlib
from dataclasses import dataclass
import pandas as pd
import kagglehub

# ========= 配置 =========
TEXT_COLUMN = "content"   # 明确指定
KEEP_COLS = ["company", "source", "datatype", "url", "date"]

# ========= 规则与打分 =========
WEASEL = r"\b(aim|aspire|strive|explore|consider|intend|plan to|promote|encourage|support|potentially|may|might|could|should|believe|likely|approximately|around|some|several|various|significant|robust|world[- ]class|best[- ]in[- ]class)\b"
WEAK_VERB = r"\b(commit to|work towards|align with|raise awareness|foster|enable|drive|advance)\b"
STRONG_VERB = r"\b(reduce|cut|install|retrofit|phase out|electrify|switch to|invest|audit|publish|ban|replace|deploy|measure|report|verify|purchase|offset|abate)\b"
PASSIVE = r"\b(is|are|was|were|been|being)\s+\w+ed\b"
HAS_NUMBER = r"(\b\d+(\.\d+)?\b|%)"
UNITS = r"\b(tco2e|co2e|mwh|kwh|km|tonnes?|t|€|\bnz\$\b|\$)\b"
DEADLINE = r"\b(by\s+(20\d\d|Q[1-4]|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})|within\s+\d+\s+(day|week|month|year)s?|no later than|by end of)\b"
OWNER = r"\b(ceo|cfo|board|operations|sustainability\s+team|procurement|plant\s+\w+|facility\s+\w+|department|committee|supplier[s]?)\b"
ORG_TAIL = r"\b(inc|ltd|plc|ag|gmbh|corp|company)\b"

def _ratio(count, total_words, cap=0.2):
    if total_words == 0: return 0.0
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
    has_units  = bool(re.search(UNITS, t))
    has_deadln = bool(re.search(DEADLINE, t))
    has_owner  = bool(re.search(OWNER, t) or re.search(r"\b[A-Z][A-Za-z&\-]+ " + ORG_TAIL, text))
    f_weasel   = _ratio(weasel_cnt, n)
    f_weak     = _ratio(weak_cnt, n)
    f_passive  = _ratio(passive_cnt, n)
    f_strong   = _ratio(strong_cnt, n)

    w = {"weasel":0.25,"weak":0.10,"passive":0.10,"miss_number":0.20,"miss_deadline":0.15,"miss_owner":0.15,"strong":-0.10,"measurables":-0.10}
    miss_number = 0.0 if has_number else 1.0
    miss_deadln = 0.0 if has_deadln else 1.0
    miss_owner  = 0.0 if has_owner else 1.0
    measurables = 1.0 if has_units else 0.0

    score = (w["weasel"]*f_weasel + w["weak"]*f_weak + w["passive"]*f_passive +
             w["miss_number"]*miss_number + w["miss_deadline"]*miss_deadln +
             w["miss_owner"]*miss_owner + w["strong"]*f_strong + w["measurables"]*measurables)
    score = max(0.0, min(1.0, score))
    label = 1 if score >= threshold else 0
    return VaguenessOutput(round(score,3),
        {"weasel_cnt":weasel_cnt,"weak_cnt":weak_cnt,"strong_cnt":strong_cnt,"passive_cnt":passive_cnt,
         "has_number":has_number,"has_units":has_units,"has_deadline":has_deadln,"has_owner":has_owner}, label)

def highlight_weasels_html(text: str) -> str:
    out = re.sub(WEASEL, lambda m: f'<span class="weasel">{m.group(0)}</span>', text, flags=re.IGNORECASE)
    out = re.sub(WEAK_VERB, lambda m: f'<span class="weasel">{m.group(0)}</span>', out, flags=re.IGNORECASE)
    out = re.sub(PASSIVE, lambda m: f'<span class="passive">{m.group(0)}</span>', out, flags=re.IGNORECASE)
    return out

# ========= 数据：只读 esg_documents_for_dax_companies.csv 的 content =========
def get_dataset_csv_path() -> pathlib.Path:
    path = kagglehub.dataset_download("equintel/dax-esg-media-dataset")
    root = pathlib.Path(path)
    hits = list(root.rglob("esg_documents_for_dax_companies.csv"))
    if hits: return hits[0]
    # 兜底：找任意包含“dax_companies.csv”的文件
    any_csv = list(root.rglob("*.csv"))
    if not any_csv:
        raise FileNotFoundError("未找到 CSV 文件。")
    return any_csv[0]

def load_samples(csv_path: pathlib.Path, max_docs: int) -> pd.DataFrame:
    usecols = [c for c in [TEXT_COLUMN,*KEEP_COLS] if c]  # 只取必要列
    acc = []
    for chunk in pd.read_csv(csv_path, usecols=lambda c: c in usecols, dtype=str,
                             on_bad_lines="skip", encoding="utf-8", chunksize=5000, engine="pyarrow"):
        chunk = chunk.dropna(subset=[TEXT_COLUMN])
        if chunk.empty: 
            continue
        acc.append(chunk)
        if sum(len(x) for x in acc) >= max_docs:
            break
    if not acc:
        raise RuntimeError(f"在 {csv_path.name} 中未找到可用文本列 `{TEXT_COLUMN}`")
    df = pd.concat(acc, ignore_index=True).head(max_docs)
    df = df.rename(columns={TEXT_COLUMN:"text"})
    return df

def build_preview_html(df_out: pd.DataFrame, threshold: float):
    rows = []
    for i, row in df_out.head(50).iterrows():
        html = highlight_weasels_html(str(row["text"]))
        rows.append(f"""
        <tr>
          <td>{i}</td><td>{row.get('company','')}</td><td>{row.get('source','')}</td>
          <td>{row['vagueness_score']:.3f}</td><td>{row['label']}</td>
          <td class="txt">{html}</td>
        </tr>""")
    html = f"""<html><head><meta charset="utf-8"/><style>
      body{{font-family:-apple-system,Segoe UI,Roboto,Arial}} table{{border-collapse:collapse;width:100%}}
      th,td{{border:1px solid #ddd;padding:8px;vertical-align:top}} th{{background:#f6f6f6;position:sticky;top:0}}
      .weasel{{background:#ffd5d5;color:#900;font-weight:600}} .passive{{background:#ffe9c9}} .txt{{max-width:900px}}
    </style></head><body>
      <h2>DAX ESG — Bullshit/Vagueness Preview</h2>
      <p>Threshold = {threshold}（label 1 = bullshit）</p>
      <table><tr><th>#</th><th>company</th><th>source</th><th>vagueness_score</th><th>label</th><th>text</th></tr>
      {''.join(rows)}
      </table></body></html>"""
    pathlib.Path("preview.html").write_text(html, encoding="utf-8")
    print("Saved: preview.html")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_docs", type=int, default=300, help="抽样文档数")
    ap.add_argument("--threshold", type=float, default=0.6, help=">= 阈值判为 1=BS")
    args = ap.parse_args()

    csv_path = get_dataset_csv_path()
    print("CSV:", csv_path)
    df = load_samples(csv_path, max_docs=args.max_docs)

    scores, labels, feats = [], [], []
    for txt in df["text"].astype(str):
        r = compute_vagueness_score(txt, threshold=args.threshold)
        scores.append(r.score); labels.append(r.label); feats.append(r.features)

    out = pd.concat(
        [df.reset_index(drop=True),
         pd.Series(scores, name="vagueness_score"),
         pd.Series(labels, name="label"),
         pd.DataFrame(feats)],
        axis=1
    )
    out.to_csv("dax_vagueness_scores.csv", index=False, encoding="utf-8")
    print(f"Saved: dax_vagueness_scores.csv (rows={len(out)})")

    build_preview_html(out, args.threshold)
    print(f"BS rate: {out['label'].mean():.1%} | mean score: {out['vagueness_score'].mean():.3f}")

if __name__ == "__main__":
    main()
