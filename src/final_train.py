"""最终训练：合并 train+val 训练，在独立 test 上做一次性评估。

为什么这么改（上一版的教训）：
  val 只有 382 条，早停信号噪声大 —— 意图任务 val macro-F1 在 3 个 epoch 里
  是 0.677→0.681→0.675，几乎是一条平线，等于随机选 epoch。
  真实情况是 epoch 1 之后就过拟合了（loss 从 0.46 掉到 0.0002）。

  更稳的做法：先在小规模上确认 epoch 数（参考上一版结果，情感/意图都在
  第 1 个 epoch 达到验证集最优），合并 train+val 后固定训练，
  只在从未参与任何选择的 test 上评估一次。这样 test 才是干净的一次性度量。

用法：
  python src/final_train.py --epochs 2
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from model import ROOT, TransformerCLS, load_ood, load_split

REPORTS = ROOT / "reports"
DB_PATH = ROOT / "db" / "customer_voice.sqlite"
TASKS = {"intent": "意图分类", "sentiment": "情感分析"}


def write_preds(version: str, df: pd.DataFrame, task: str, preds) -> None:
    con = sqlite3.connect(DB_PATH)
    # 先删同版本旧预测，保证可重复运行
    con.execute(
        "DELETE FROM predictions WHERE model_version=? AND task=?", (version, task)
    )
    con.executemany(
        "INSERT INTO predictions (msg_id, model_version, task, label, confidence) "
        "VALUES (?,?,?,?,?)",
        [(int(m), version, task, p.label, p.confidence) for m, p in zip(df.msg_id, preds)],
    )
    con.commit()
    con.close()


def write_eval(version: str, task: str, res: dict) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "DELETE FROM evaluations WHERE model_version=? AND task=?", (version, task)
    )
    con.executemany(
        "INSERT INTO evaluations (model_version, task, metric, value) VALUES (?,?,?,?)",
        [(version, task, "accuracy", res["accuracy"]),
         (version, task, "macro_f1", res["macro_f1"]),
         (version, task, "ood_accuracy", res["ood_accuracy"])],
    )
    con.commit()
    con.close()


def main(epochs: int, seed: int) -> None:
    REPORTS.mkdir(exist_ok=True)

    # 固定随机种子：不加的话两次运行结果会有 3 个点左右的波动
    # （分类头初始化 + 数据打乱顺序都不确定），报告数字就不可复现。
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tr, te, va = load_split()
    ood = load_ood()

    # 合并 train+val 用于最终训练；test 全程不参与任何决策
    tr_full = pd.concat([tr, va], ignore_index=True)
    print(f"合成训练集(合并 train+val) {len(tr_full)} / 测试(一次性) {len(te)} / 分布外 {len(ood)}")

    # 公开标注语料（真实评论）只补情感任务，且**不参与测试集**
    # —— 两者来源不同（酒店评论/微博 vs 客服对话），不存在泄漏。
    ext = None
    ext_path = ROOT / "data" / "external_sentiment.parquet"
    if ext_path.exists():
        ext = pd.read_parquet(ext_path)
        print(f"外部真实语料 {len(ext):,} 条（来源: {dict(ext.source.value_counts())}）")

    summary = []
    for task, cn in TASKS.items():
        print(f"\n{'=' * 60}\n任务：{cn} ({task})   epochs={epochs}\n{'=' * 60}")

        # max_len 按任务区分：客服短句 p99 只有 32 字，192 反而让模型
        # 在大量 padding 上浪费容量（实测意图分类从 86.7% 掉到 79.3%）；
        # 外部评论语料 p90=139 字，才需要更长窗口。
        # 160 是对照实验里效果与速度的折中（192 慢 50%，指标没更好）
        max_len = 160 if task == "sentiment" else 128
        batch = 32 if task == "sentiment" else 16
        m = TransformerCLS(task, name=f"bert-{task}", epochs=epochs,
                           max_len=max_len, batch=batch)

        # 情感任务：合成数据 + 外部真实语料一起训
        if task == "sentiment" and ext is not None:
            # 取舍：ChnSentiCorp 是商品/服务评价类语料，与客服场景同域，全量用；
            # 微博语料题材杂、噪声大，只抽少量做多样性补充。
            # 全量 4.5 万条在 CPU 上要训几十分钟，性价比不高。
            # 数据量用对照实验定过，不是拍脑袋：
            #   纯合成 → 客服 87.7% / 真实酒店评论 79.3%
            #   +3000 条 ChnSentiCorp → 客服 86.9% / 真实 89.8%
            # 真实场景 +10.5pt，客服场景只掉 0.8pt，性价比最高。
            # 早期的 1.46 万条版本训练要几十分钟，收益不成比例，已放弃。
            ext_chn = ext[ext.source == "chnsenticorp"]
            chn = ext_chn.sample(min(3000, len(ext_chn)), random_state=42)
            ext_use = chn
            train_texts = tr_full.text.tolist() + ext_use.text.tolist()
            train_labels = tr_full[task].tolist() + ext_use.sentiment.tolist()
            print(f"  训练集 = 合成 {len(tr_full)} + 真实 {len(ext_use)} "
                  f"= {len(train_texts)}")
        else:
            train_texts = tr_full.text.tolist()
            train_labels = tr_full[task].tolist()

        m.fit(train_texts, train_labels)

        preds = m.predict(te.text.tolist())
        y_true = te[task].tolist()
        y_pred = [p.label for p in preds]

        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # 负类召回：客服场景下最关键的单项指标
        labels = sorted(set(y_true))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        neg_recall = None
        if task == "sentiment" and "负面" in labels:
            i = labels.index("负面")
            neg_recall = float(cm[i, i] / cm[i].sum())

        # 真实公开语料上的表现（holdout，未参与训练）
        real_acc = None
        if task == "sentiment" and ext is not None:
            chn_all = ext[ext.source == "chnsenticorp"]
            used = set(chn_all.sample(min(3000, len(chn_all)), random_state=42).text)
            hold = chn_all[~chn_all.text.isin(used)].sample(600, random_state=7)
            hp = m.predict(hold.text.tolist())
            real_acc = float(np.mean([p.label == t for p, t in zip(hp, hold.sentiment)]))
            print(f"  真实酒店评论 acc={real_acc:.4f} (n={len(hold)})")

        ood_preds = m.predict(ood.text.tolist())
        ood_acc = float(np.mean([p.label == t for p, t in zip(ood_preds, ood[task])]))

        print(f"  test acc={acc:.4f}  macroF1={mf1:.4f}  ood acc={ood_acc:.4f}")
        if neg_recall is not None:
            print(f"  负面召回={neg_recall:.4f}")
        print("\n" + classification_report(y_true, y_pred, digits=4, zero_division=0))

        version = f"bert-{task}"
        write_preds(version, te, task, preds)
        write_eval(version, task, {"accuracy": acc, "macro_f1": mf1, "ood_accuracy": ood_acc})

        # 持久化模型，供看板的在线分析接口直接加载
        ckpt = ROOT / "models" / version
        m.save(ckpt)
        print(f"  模型已保存: {ckpt}")

        summary.append({
            "task": task, "model": version, "epochs": epochs,
            "n_train": len(tr_full), "n_test": len(te), "n_ood": len(ood),
            "accuracy": acc, "macro_f1": mf1, "ood_accuracy": ood_acc,
            "real_corpus_accuracy": real_acc,
            "negative_recall": neg_recall,
            "labels": labels, "confusion_matrix": cm.tolist(),
        })

    (REPORTS / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n最终报告: {REPORTS / 'final_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.epochs, a.seed)