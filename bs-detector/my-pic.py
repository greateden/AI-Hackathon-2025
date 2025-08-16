import pandas as pd
import matplotlib.pyplot as plt

# 读取用户上传的 CSV 文件
df = pd.read_csv("bs-detector/dax_vagueness_scores.csv")

# 设置统一的字体大小，适合演讲
plt.rcParams.update({
    "font.size": 16,      # 基础字体
    "axes.titlesize": 20, # 标题
    "axes.labelsize": 18, # 轴标签
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# 1. vagueness_score vs bs_score 散点图
plt.figure(figsize=(8,6))
plt.scatter(df["vagueness_score"], df["bs_score"], alpha=0.5)
plt.xlabel("Vagueness Score")
plt.ylabel("BS Score")
plt.title("Vagueness vs BS Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("vagueness_vs_bs.png")
plt.close()

# 2. BS Score 分布直方图
plt.figure(figsize=(8,6))
plt.hist(df["bs_score"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("BS Score")
plt.ylabel("Frequency")
plt.title("Distribution of BS Scores")
plt.grid(True)
plt.tight_layout()
plt.savefig("bs_score_hist.png")
plt.close()

# 3. 按 domain 的 top10 平均 BS Score
if "domain" in df.columns:
    domain_scores = df.groupby("domain")["bs_score"].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    domain_scores.plot(kind="bar", color="orange", edgecolor="black")
    plt.ylabel("Mean BS Score")
    plt.title("Top 10 Domains by Mean BS Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("domain_bs_scores.png")
    plt.close()

