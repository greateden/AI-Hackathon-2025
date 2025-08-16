import argparse, json, re, pathlib
from dataclasses import dataclass
import pandas as pd
import kagglehub
import sys, csv  # 放在文件顶部（若已导入则忽略）

# ========= 配置 =========
TEXT_COLUMN = "content"   # 明确指定
KEEP_COLS = ["company","datatype","date","domain","esg_topics","symbol","title","url","internal","source"]

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
    # 1) 先尝试用 pyarrow 整表读取（最稳，能处理长文本/多行）
    try:
        df = pd.read_csv(
            csv_path,
            sep="|",
            engine="pyarrow",
            dtype=str
        )
        print("[info] Loaded full CSV with pyarrow (pipe-delimited).")
    except Exception as e:
        print(f"[warn] pyarrow failed, fallback to python engine: {e}")
        # 2) 回退到 python 引擎，并把单字段大小上限拉高
        try:
            csv.field_size_limit(min(sys.maxsize, 10**9))
        except OverflowError:
            csv.field_size_limit(10**9)
        df = pd.read_csv(
            csv_path,
            sep="|",
            engine="python",
            dtype=str,
            on_bad_lines="skip"
        )

    # 丢掉开头的“空表头/Unnamed”列（管道分隔导致的占位列）
    drop_cols = [c for c in df.columns if (c is None) or (str(c).strip() == "") or str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 确认 content 存在
    if "content" not in df.columns:
        raise RuntimeError(f"`content` 不在列中：{df.columns.tolist()}")

    # 只保留我们关心的列
    keep = ["company","content","datatype","date","domain","esg_topics","internal","symbol","title","url"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].rename(columns={"content": "text"}).dropna(subset=["text"])

    # 抽样
    if len(df) > max_docs:
        df = df.sample(n=max_docs, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df

def build_preview_html(df_out: pd.DataFrame, threshold: float, top_n: int = 50):
    """渲染预览页：按分数降序展示 top_n 条，并附 Summary（按 datatype / domain 的均分&命中率）"""

    # 取需要展示的列是否存在
    has_domain = "domain" in df_out.columns
    has_dtype  = "datatype" in df_out.columns

    # Summary：按 datatype / domain 聚合
    summ_parts = []
    if has_dtype:
        by_type = df_out.groupby("datatype").agg(
            mean_score=("vagueness_score","mean"),
            bs_rate=("label","mean"),
            n=("label","size"),
        ).sort_values("mean_score", ascending=False).round(3)
        summ_parts.append("<p><b>By datatype</b></p>" + by_type.to_html(escape=False))

    if has_domain:
        by_domain = df_out.groupby("domain").agg(
            mean_score=("vagueness_score","mean"),
            bs_rate=("label","mean"),
            n=("label","size"),
        ).sort_values("mean_score", ascending=False).head(10).round(3)
        summ_parts.append("<p><b>Top domains by score</b></p>" + by_domain.to_html(escape=False))

    summary_html = "<h3>Summary</h3>" + ("".join(summ_parts) if summ_parts else "<p>(no grouping columns found)</p>")

    # 表头
    cols_header = ["#", "company"]
    if has_domain: cols_header.append("domain")
    if has_dtype:  cols_header.append("datatype")
    cols_header += ["vagueness_score", "label", "weasel_cnt", "passive_cnt", "text (highlighted)"]

    # 构建行（取前 top_n）
    head = df_out.head(top_n)
    rows_html = []
    for i, row in head.iterrows():
        html_txt = highlight_weasels_html(str(row.get("text","")))
        tds = [
            str(i),
            str(row.get("company","")),
        ]
        if has_domain: tds.append(str(row.get("domain","")))
        if has_dtype:  tds.append(str(row.get("datatype","")))
        tds += [
            f"{row['vagueness_score']:.3f}",
            str(row.get("label","")),
            str(row.get("weasel_cnt","")),
            str(row.get("passive_cnt","")),
            f'<div class="txt">{html_txt}</div>',
        ]
        rows_html.append("<tr>" + "".join(f"<td>{x}</td>" for x in tds) + "</tr>")

    table_html = f"""
    <table>
      <tr>{"".join(f"<th>{h}</th>" for h in cols_header)}</tr>
      {''.join(rows_html)}
    </table>
    """

    # 页面模板
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
      <p class="meta">Threshold = {threshold:.2f}（label 1 = bullshit） | Rows = {len(df_out)}</p>
      {summary_html}
      {table_html}
    </body></html>
    """

    with open("preview.html","w",encoding="utf-8") as f:
        f.write(template)
    print("Saved: preview.html")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_docs", type=int, default=300, help="抽样文档数")
    ap.add_argument("--threshold", type=float, default=0.6, help=">= 阈值判为 1=BS")
# argparse 增加两个开关
    ap.add_argument("--quantile", type=float, default=None, help="用分位数自动设阈值，如 0.8 表示取 top20% 为 1")
    ap.add_argument("--reports_only", action="store_true", help="仅评估 datatype=report 的文本")

    args = ap.parse_args()

    csv_path = get_dataset_csv_path()
    print("CSV:", csv_path)
# 读取后可选过滤
    df = load_samples(csv_path, max_docs=args.max_docs)
    if args.reports_only and "datatype" in df.columns:
        df = df[df["datatype"].str.lower().eq("report")]

    scores, labels, feats = [], [], []
    for txt in df["text"].astype(str):
        r = compute_vagueness_score(txt, threshold=args.threshold)
        scores.append(r.score); labels.append(r.label); feats.append(r.features)

# === 合并结果 ===
    out = pd.concat(
        [
            df.reset_index(drop=True),
            pd.Series(scores, name="vagueness_score"),
            pd.Series(labels, name="label"),
            pd.DataFrame(feats),
        ],
        axis=1,
    )

# === 排序：高分在前，方便预览锁定“最水”的 ===
    out = out.sort_values("vagueness_score", ascending=False).reset_index(drop=True)

# === 阈值：支持分位数自动阈值（若你之前在 argparse 里加了 --quantile） ===
# 没加也没事，这里会自动忽略，使用你传入的 --threshold
    thr = args.threshold
    if hasattr(args, "quantile") and args.quantile is not None:
        thr = float(out["vagueness_score"].quantile(args.quantile))
        out["label"] = (out["vagueness_score"] >= thr).astype(int)
        print(f"[auto] threshold(from quantile={args.quantile}) = {thr:.3f}, BS rate = {out['label'].mean():.1%}")

# === 导出 CSV ===
    out.to_csv("dax_vagueness_scores.csv", index=False, encoding="utf-8")
    print(f"Saved: dax_vagueness_scores.csv (rows={len(out)})")

# === 生成 HTML 预览（前 50 条）===
    build_preview_html(out, threshold=thr, top_n=50)

# === 简要统计 ===
    print(f"BS rate: {out['label'].mean():.1%} | mean score: {out['vagueness_score'].mean():.3f} | threshold: {thr:.2f}")

if __name__ == "__main__":
    main()
