"""情感分析回归测试：专门盯"程度副词 + 负面形容词"这类系统性错误。

来源：用户反馈"客服态度好差"被判成正面（置信度 0.97）暴露的问题。
中文里"好"既可作形容词（服务态度好）也可作程度副词（态度好差），
模型如果只学字面搭配，遇到后者会自信地判错。

这个文件的作用是**回归防护**：以后每次改数据或换模型都跑一遍，
防止修好的问题被改回去。

用法：
  python src/test_sentiment_regression.py
  python src/test_sentiment_regression.py --model baseline-sentiment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# (文本, 期望情感, 说明)
CASES: list[tuple[str, str, str]] = [
    # ---- 用户直接反馈的句子
    ("客服态度好差", "负面", "用户原始报障句"),
    ("客服态度好烂", "负面", "同结构换形容词"),
    ("服务态度好差", "负面", "加'服务'前缀"),
    ("态度真的好差", "负面", "插入'真的'"),
    ("体验好差", "负面", "换主语"),
    ("质量好差", "负面", "换主语"),
    ("包装好差", "负面", "换主语"),
    ("物流好差", "负面", "换主语"),
    # ---- 其他程度副词
    ("客服态度太差", "负面", "太+形容词（对照组，原本就对）"),
    ("客服态度很差", "负面", "很+形容词（对照组）"),
    ("客服态度特别差", "负面", "特别+形容词"),
    ("客服态度超级差", "负面", "超级+形容词"),
    # ---- 其他"好+负面形容词"
    ("这个价格好贵", "负面", "好贵"),
    ("快递好慢啊", "负面", "好慢"),
    ("这东西好难用", "负面", "好难用"),
    ("等了好久都没人回", "负面", "好久"),
    ("这次真的好气人", "负面", "好气人"),
    ("买得好后悔", "负面", "好后悔"),
    # ---- 反例：必须保持判对的正面句子（防止矫枉过正）
    ("服务态度好，很满意", "正面", "反例：真的好"),
    ("客服态度很好，感谢", "正面", "反例：很好"),
    ("东西不错，物流也快", "正面", "反例：正面评价"),
    ("质量好，用着很舒服", "正面", "反例：质量好"),
]


def load_model(version: str):
    from model import BaselineTFIDF, TransformerCLS

    if version.startswith("baseline"):
        task = version.split("-", 1)[1]
        return BaselineTFIDF.load(task, ROOT / "models" / f"{version}.joblib")
    m = TransformerCLS("sentiment", name=version)
    return m.load(ROOT / "models" / version)


def main(version: str, verbose: bool) -> int:
    m = load_model(version)
    fails: list[tuple[str, str, str, str, float]] = []

    by_group: dict[str, list[int]] = {"负面": [0, 0], "正面": [0, 0]}  # [对, 总]
    for text, expect, note in CASES:
        p = m.predict([text])[0]
        ok = p.label == expect
        by_group[expect][1] += 1
        by_group[expect][0] += int(ok)
        if not ok:
            fails.append((text, expect, p.label, note, p.confidence))
        if verbose:
            mark = "✅" if ok else "❌"
            print(f"  {mark} {text:16s} 期望={expect} 实际={p.label} "
                  f"({p.confidence:.3f})  {note}")

    total = len(CASES)
    passed = total - len(fails)
    print(f"\n{'=' * 56}")
    print(f"模型 {version}: {passed}/{total} 通过 ({passed / total:.1%})")
    for label, (ok, n) in by_group.items():
        if n:
            print(f"  {label}样本: {ok}/{n}")

    if fails:
        print(f"\n失败 {len(fails)} 条:")
        for text, expect, got, note, conf in fails:
            print(f"  ❌ {text:16s} 期望 {expect} → 实际 {got} ({conf:.3f})  [{note}]")
        return 1

    print("\n全部通过")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-sentiment")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.model, a.verbose))