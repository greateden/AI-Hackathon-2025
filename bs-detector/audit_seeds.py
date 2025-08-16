import pandas as pd
import pathlib, re

def _read_seeds(seed_csv: str, min_chars: int = 20):
    path = pathlib.Path(seed_csv)
    if not path.exists():
        raise FileNotFoundError(f"seed csv not found: {seed_csv}")

    # auto-detect separator
    with open(path, "rb") as fh:
        sniff = fh.read(4096)
    text_head = sniff.decode("utf-8", errors="ignore")
    sep = "|" if text_head.count("|") > text_head.count(",") else ","

    # try utf-8-sig first to survive BOM
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    print(f"[seed] raw shape = {df.shape}, sep='{sep}'")

    # canonicalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    # try to locate text/label with a few aliases
    text_col = None
    for k in ["text","content","sentence","snippet"]:
        if k in cols: text_col = cols[k]; break
    label_col = None
    for k in ["label","y","is_bs","class","target"]:
        if k in cols: label_col = cols[k]; break
    if text_col is None or label_col is None:
        raise RuntimeError(f"[seed] need columns 'text' and 'label' (found: {list(df.columns)})")

    df = df.rename(columns={text_col:"text", label_col:"label"})
    print(f"[seed] after rename: {df.columns.tolist()}")

    # normalize label to 0/1
    def norm_label(x):
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"1","true","yes","y","bs","vague","bad"}: return 1
            if s in {"0","false","no","n","ok","good","clear"}: return 0
        return int(x)
    df["label"] = df["label"].apply(norm_label)

    # basic cleaning
    df["text"] = df["text"].astype(str).str.strip()
    before = len(df)
    df = df.dropna(subset=["text","label"])
    print(f"[seed] dropna: {before} -> {len(df)}")

    # kill ultra-short zombies
    before = len(df)
    df = df[df["text"].str.len() >= min_chars]
    print(f"[seed] min_chars>={min_chars}: {before} -> {len(df)}")

    # dedup
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"[seed] dedup: {before} -> {len(df)}")

    # class balance snapshot
    print("[seed] class counts:\n", df["label"].value_counts(dropna=False).to_string())

    if df["label"].nunique() < 2:
        raise RuntimeError("[seed] only one class left after cleaning — add more seeds or check label mapping.")

    return df
