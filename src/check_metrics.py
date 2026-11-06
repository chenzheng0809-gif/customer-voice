"""校验评估指标是否在"合理区间"，防止数据泄漏或代码回归悄悄发生。

这个脚本的存在理由：
  这个项目最大的教训就是「指标突然变好看 = 大概率出问题」。
  baseline 意图分类一度跑到 100%，根因是训练/测试同源。
  把预期区间写成断言进 CI，下次再有人改动数据生成逻辑导致泄漏，
  CI 会直接红掉，而不是等你事后发现。

判定原则（都是保守下界/上界，不追求精确）：
  - baseline 意图分类不应超过 85%：TF-IDF 的合理上限，超过就该怀疑泄漏
  - baseline 与 BERT 都不应为 100%
  - 训练集与测试集不得有句子重合（硬性条件）

用法：
  python src/check_metrics.py
  python src/check_metrics.py --allow-no-bert   # 只跑了 baseline 时
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "db" / "customer_voice.sqlite"
REPORTS = ROOT / "reports"

# (任务, 指标, 下界, 上界, 说明)
RULES: list[tuple[str, str, float, float, str]] = [
    ("intent", "baseline", 0.40, 0.85,
     "TF-IDF baseline 应明显低于上限，过高多半是训练/测试同源"),
    ("intent", "best", 0.60, 0.98,
     "微调后应显著高于 baseline，但不应接近 100%"),
    ("sentiment", "baseline", 0.30, 0.90,
     "情感 baseline 因程度副词样本通常偏低"),
    ("sentiment", "best", 0.60, 0.98,
     "微调后情感应 >= 0.60，且不应接近 100%"),
]


def load_metrics() -> dict[str, dict[str, float]]:
    """从 evaluations 表读每个任务的最优与 baseline 指标。"""
    con = sqlite3.connect(DB)
    rows = con.execute("""
        SELECT model_version, task, metric, value FROM evaluations
        WHERE metric IN ('accuracy','macro_f1','ood_accuracy','real_corpus_accuracy')
    """).fetchall()
    con.close()

    by_task: dict[str, dict[str, dict[str, float]]] = {}
    for ver, task, metric, val in rows:
        by_task.setdefault(task, {}).setdefault(ver, {})[metric] = val

    out: dict[str, dict[str, float]] = {}
    for task, versions in by_task.items():
        accs = {v: m.get("accuracy") for v, m in versions.items() if m.get("accuracy") is not None}
        if not accs:
            continue
        base = next((v for v in accs if v.startswith("baseline")), None)
        best = max(accs, key=accs.get)
        out[task] = {
            "baseline": accs[base] if base else None,
            "best": accs[best],
            "best_version": best,
            **{k: versions[best].get(k) for k in
               ("macro_f1", "ood_accuracy", "real_corpus_accuracy")},
        }
    return out


def check_leakage() -> list[str]:
    """硬性检查：训练集与测试集不得有完全相同的句子。"""
    errs: list[str] = []
    frames = ROOT / "data" / "frames.csv"
    if not frames.exists():
        return ["data/frames.csv 不存在，跳过泄漏检查"]

    df = pd.read_csv(frames)
    cust = df[df.role == "customer"]
    tr = set(cust[cust.split == "train"].text.str.strip())
    te = set(cust[cust.split == "test"].text.str.strip())
    overlap = tr & te
    if overlap:
        errs.append(f"训练集与测试集有 {len(overlap)} 句重合，疑似数据泄漏")
        for s in list(overlap)[:3]:
            errs.append(f"    样例: {s[:50]}")
    return errs


def main(allow_no_bert: bool) -> int:
    if not DB.exists():
        print(f"数据库不存在：{DB}")
        return 1

    metrics = load_metrics()
    errors: list[str] = []
    warnings: list[str] = []

    print("=" * 64)
    print("指标合理性校验")
    print("=" * 64)
    for task, m in metrics.items():
        print(f"\n[{task}] 最佳版本 = {m['best_version']}")
        for k in ("baseline", "best", "macro_f1", "ood_accuracy", "real_corpus_accuracy"):
            if m.get(k) is not None:
                print(f"  {k:22s} {m[k]:.4f}")

    for task, kind, lo, hi, note in RULES:
        m = metrics.get(task)
        if not m or m.get(kind) is None:
            if kind == "best" and allow_no_bert:
                warnings.append(f"{task}/{kind} 缺失（--allow-no-bert 已放行）")
                continue
            errors.append(f"{task}/{kind} 缺失指标，无法校验")
            continue
        val = m[kind]
        label = f"{task}/{kind} = {val:.4f}"
        if val > hi:
            errors.append(f"{label} 超过上限 {hi} —— {note}")
        elif val < lo:
            errors.append(f"{label} 低于下限 {lo} —— {note}")

    print()
    leak = check_leakage()
    if not leak or leak[0].startswith("data/frames.csv 不存在"):
        print("泄漏检查：通过（train ∩ test = 0）")
    errors.extend(leak)

    for w in warnings:
        print(f"  [WARN] {w}")
    if errors:
        print("\n校验失败：")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print("\n全部通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-no-bert", action="store_true",
                    help="只跑了 baseline 时允许缺少微调指标")
    sys.exit(main(ap.parse_args().allow_no_bert))