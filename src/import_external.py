"""把公开标注语料并入训练集 —— 用真实数据补合成数据的盲区。

动机（来自实际踩坑）：
  用合成数据训出来的情感模型，在真实酒店评论 ChnSentiCorp 上只有 79.3% 准确率，
  其中「真实好评被判负面」的错误率高达 25%。原因是合成数据里的正面样本
  基本是"谢谢/好评/很满意"这类**显式感谢语**，而真实好评是
  "环境安静、早餐丰富、位置方便"这类**中性描述**——模型没见过。

数据来源与许可：
  - ChnSentiCorp（谭松波中文酒店评论语料）
    7,766 条，二分类（正/负），来源：https://github.com/pengming617/bert_classification
    学术研究常用语料，引用：Tan & Zhang, 2008
  - weibo_senti_100k（微博情感语料）
    119,988 条，二分类，正负各半，来源：https://github.com/SophonPlus/ChineseNlpCorpus

接入策略：
  只把**二分类**的语料用于情感任务的补充，不碰意图任务
  （公开语料没有意图标注）。
  情感原任务是三分类，这里把公开语料的极性映射为 正面/负面，
  中性仍然只来自合成数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXT = DATA / "external"
OUT = DATA / "external_sentiment.parquet"


def load_chnsenticorp() -> pd.DataFrame:
    f = EXT / "chnsenticorp_htl.csv"
    if not f.exists():
        raise FileNotFoundError(f"缺少 {f}，请先下载 ChnSentiCorp")
    d = pd.read_csv(f).dropna(subset=["review"])
    d = d.rename(columns={"review": "text"})
    d["source"] = "chnsenticorp"
    return d[["text", "label", "source"]]


def load_weibo(limit: int = 40000, seed: int = 42) -> pd.DataFrame:
    f = EXT / "weibo_senti_100k.csv"
    if not f.exists():
        raise FileNotFoundError(f"缺少 {f}，请先下载 weibo_senti_100k")
    d = pd.read_csv(f).dropna(subset=["review"]).rename(columns={"review": "text"})
    d["source"] = "weibo"
    d = d[["text", "label", "source"]]
    # 微博语料很大，默认只抽 4 万条，避免训练时间失控
    if limit and len(d) > limit:
        d = d.sample(limit, random_state=seed)
    return d


def clean_text(t: str) -> str:
    """微博语料带 [表情] 占位符，这类 token 对情感没有稳定含义，去掉。"""
    import re

    t = re.sub(r"\[[\u4e00-\u9fff]{1,6}\]", "", str(t))
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def main(weibo_limit: int, min_len: int, seed: int) -> None:
    frames = [load_chnsenticorp()]
    if weibo_limit != 0:
        try:
            frames.append(load_weibo(weibo_limit, seed))
        except FileNotFoundError as e:
            print(f"跳过微博语料：{e}")

    d = pd.concat(frames, ignore_index=True)
    d["text"] = d.text.map(clean_text)
    n0 = len(d)
    d = d[d.text.str.len() >= min_len].drop_duplicates("text")

    # label 1 -> 正面, 0 -> 负面
    d["sentiment"] = d.label.map({1: "正面", 0: "负面"})
    d = d.dropna(subset=["sentiment"])

    print(f"原始 {n0:,} 条 -> 清洗后 {len(d):,} 条（去掉 {n0 - len(d):,} 条过短/重复/无效）")
    print("来源分布:")
    print(d.source.value_counts().to_string())
    print("\n情感分布:")
    print(d.sentiment.value_counts().to_string())

    d[["text", "sentiment", "source"]].to_parquet(OUT, index=False)
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weibo-limit", type=int, default=40000,
                    help="微博语料最多取多少条，0 表示不用")
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.weibo_limit, a.min_len, a.seed)