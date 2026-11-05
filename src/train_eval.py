"""训练 + 评测 + 入库。

流程：
  1. 按会话切分训练/测试集
  2. 两个任务（意图 / 情感）各训两个模型（baseline / 微调）
  3. 计算指标，做显著性检验（scipy）
  4. 所有预测写入 predictions 表，指标写入 evaluations 表
  5. 输出报告：混淆矩阵、难样本分析

用法：
  python src/train_eval.py --fast     # 只跑 baseline（无 torch 也能跑）
  python src/train_eval.py            # baseline + 微调
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from model import (
    BaselineTFIDF,
    MODELS,
    ROOT,
    clean,
    load_ood,
    load_split,
)

DB_PATH = ROOT / "db" / "customer_voice.sqlite"
REPORTS = ROOT / "reports"
TASKS = {"intent": "意图分类", "sentiment": "情感分析"}


def evaluate(y_true, y_pred, model, task, split="test"):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    return {
        "model": model,
        "task": task,
        "split": split,
        "accuracy": acc,
        "macro_f1": mf1,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": sorted(set(y_true)),
    }


def mcnemar(y_true, pred_a, pred_b) -> float:
    """McNemar 检验：比较两个分类器在同一测试集上的差异是否显著。

    b = A 对 B 错的样本数；c = A 错 B 对的样本数。
    用精确二项检验更适合小样本（我们的测试集只有 190 条）。
    """
    correct_a = np.array([p == t for p, t in zip(pred_a, y_true)])
    correct_b = np.array([p == t for p, t in zip(pred_b, y_true)])
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0
    res = stats.binomtest(b, b + c, 0.5, alternative="two-sided")
    return float(res.pvalue)


def write_predictions(model_version: str, df: pd.DataFrame, task: str, preds) -> None:
    con = sqlite3.connect(DB_PATH)
    con.executemany(
        """INSERT INTO predictions (msg_id, model_version, task, label, confidence)
           VALUES (?,?,?,?,?)""",
        [(int(mid), model_version, task, p.label, p.confidence)
         for mid, p in zip(df.msg_id, preds)],
    )
    con.commit()
    con.close()


def write_eval(model_version: str, res: dict) -> None:
    con = sqlite3.connect(DB_PATH)
    rows = [
        (model_version, res["task"], "accuracy", res["accuracy"]),
        (model_version, res["task"], "macro_f1", res["macro_f1"]),
    ]
    con.executemany(
        "INSERT INTO evaluations (model_version, task, metric, value) VALUES (?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


def main(fast: bool, seed: int) -> None:
    REPORTS.mkdir(exist_ok=True)
    tr, te, va = load_split(seed=seed)
    ood = load_ood()
    summary = []

    print(f"训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)} / 分布外 {len(ood)}")

    for task, cn in TASKS.items():
        print(f"\n{'=' * 60}\n任务：{cn} ({task})\n{'=' * 60}")
        ytr, yte = tr[task].tolist(), te[task].tolist()

        results = {}

        # --- baseline
        print("[baseline] TF-IDF + LinearSVC")
        base = BaselineTFIDF(task, name=f"baseline-{task}")
        base.fit(tr.text.tolist(), ytr)
        p_base = base.predict(te.text.tolist())
        pred_base = [p.label for p in p_base]
        r_base = evaluate(yte, pred_base, f"baseline-{task}", task)
        results["baseline"] = (r_base, pred_base, p_base)
        print(f"  分布内 acc={r_base['accuracy']:.4f}  macroF1={r_base['macro_f1']:.4f}")

        # baseline 在分布外测试集上的表现（这是最能说明问题的指标）
        ood_pred_base = [p.label for p in base.predict(ood.text.tolist())]
        ood_acc_base = np.mean([p == t for p, t in zip(ood_pred_base, ood[task])])
        print(f"  分布外 acc={ood_acc_base:.4f}")

        # --- 微调
        if not fast:
            print("[finetuned] bert-base-chinese")
            try:
                from model import TransformerCLS

                tr_m = TransformerCLS(task, name=f"finetuned-{task}")
                tr_m.fit(tr.text.tolist(), ytr,
                         val_texts=va.text.tolist(), val_labels=va[task].tolist())
                p_ft = tr_m.predict(te.text.tolist())
                pred_ft = [p.label for p in p_ft]
                r_ft = evaluate(yte, pred_ft, f"finetuned-{task}", task)
                results["finetuned"] = (r_ft, pred_ft, p_ft)
                print(f"  分布内 acc={r_ft['accuracy']:.4f}  macroF1={r_ft['macro_f1']:.4f}")
                ood_pred_ft = [p.label for p in tr_m.predict(ood.text.tolist())]
                ood_acc_ft = np.mean([p == t for p, t in zip(ood_pred_ft, ood[task])])
                print(f"  分布外 acc={ood_acc_ft:.4f}")
            except Exception as e:
                print(f"  微调跳过（{type(e).__name__}: {e}）")

        # --- 显著性检验 & 难样本
        hard_cases = []
        if "finetuned" in results:
            pv = mcnemar(yte, pred_base, results["finetuned"][1])
            print(f"  McNemar 精确检验 p = {pv:.4f} "
                  f"({'显著' if pv < 0.05 else '不显著'}，α=0.05)")
            both_wrong = [
                (te.iloc[i].text, yte[i], pred_base[i], results["finetuned"][1][i])
                for i in range(len(yte))
                if pred_base[i] != yte[i] and results["finetuned"][1][i] != yte[i]
            ]
            hard_cases = both_wrong
            print(f"  两个模型都错的样本: {len(both_wrong)} 条")

        # --- 入库 + 汇总
        ood_scores = {"baseline": ood_acc_base}
        if "finetuned" in results:
            ood_scores["finetuned"] = ood_acc_ft

        for key, (res, labels, preds) in results.items():
            ver = f"{key}-{task}"
            write_predictions(ver, te, task, preds)
            write_eval(ver, res)

        best = max(results, key=lambda k: results[k][0]["macro_f1"])
        summary.append({
            "task": task,
            "n_train": len(tr), "n_test": len(te), "n_ood": len(ood),
            "results": {
                k: {
                    "accuracy": v[0]["accuracy"],
                    "macro_f1": v[0]["macro_f1"],
                    "ood_accuracy": ood_scores.get(k),
                }
                for k, v in results.items()
            },
            "confusion": {k: v[0]["confusion_matrix"] for k, v in results.items()},
            "labels": sorted(set(yte)),
            "hard_cases": hard_cases[:15],
        })

        print("\n" + results[best][0]["report"])

    # 对比图数据
    (REPORTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {"task": s["task"], "model": k, **v}
        for s in summary for k, v in s["results"].items()
    ]).to_csv(REPORTS / "metrics.csv", index=False, encoding="utf-8")

    print("\n报告已写入 reports/summary.json 与 reports/metrics.csv")
    print(f"预测已入库：{DB_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="只跑 baseline")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.fast, a.seed)