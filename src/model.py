"""模型层：baseline 与 transformer 两套实现，输出统一接口。

统一接口的意义（面试可讲）：
  下游的建库/看板/评测都只依赖 predict() 的返回格式，
  所以换模型不需要改任何下游代码 —— 这就是"策略模式"在 ML 工程里的用法。

两个模型：
  BaselineTFIDF : TF-IDF + LinearSVC，秒级训练，指标是"及格线"
  TransformerCLS: 预训练中文模型微调，指标是"上限"

两者都必须输出 (label, confidence)，才能统一入库并比较。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

from schema import INTENT_NAMES, SENTIMENT_NAMES  # noqa: E402


# ---------------------------------------------------------------- 统一接口
@dataclass
class Pred:
    label: str
    confidence: float


class Classifier(Protocol):
    name: str
    task: str

    def fit(self, texts: list[str], labels: list[str]) -> "Classifier": ...
    def predict(self, texts: list[str]) -> list[Pred]: ...


# ---------------------------------------------------------------- 预处理
# 中文口语化的清洗规则。注意：不要过度清洗，否则会把情感信号（！、？）洗掉。
_PUNCT_KEEP = re.compile(r"[^\w\s\u4e00-\u9fff！？。，]")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """保留情绪标点（！？。，），去掉其他符号与多余空白。

    注意：这里**不做**程度副词归一化。归一化只在推理时生效（见 predict 层），
    训练数据保持原始表达，避免把"好差"改写成"很差"后丢掉真实的语言分布。
    """
    t = _PUNCT_KEEP.sub(" ", str(text))
    return _WS.sub(" ", t).strip()


# ---------------------------------------------------------------- baseline
class BaselineTFIDF:
    """TF-IDF 字符 n-gram + 线性分类器。

    选字符级 n-gram 而非分词，是因为中文分词会引入第三方词典，
    字符 n-gram 对口语化/错别字更鲁棒，且无需额外依赖。
    """

    def __init__(self, task: str, name: str = "baseline-tfidf"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import Pipeline
        from sklearn.calibration import CalibratedClassifierCV

        self.task = task
        self.name = name
        self._vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),   # 捕获单字到三字短语
            min_df=1,
            max_features=8000,
            sublinear_tf=True,
        )
        # LinearSVC 的 decision_function 不是概率，用 CalibratedClassifierCV 转成概率
        # 这样两个模型都有可信的 confidence，可以统一入库
        base = Pipeline([("vec", self._vec), ("clf", LinearSVC(max_iter=5000))])
        self.model = CalibratedClassifierCV(base, cv=3)

    def fit(self, texts, labels):
        self.model.fit([clean(t) for t in texts], labels)
        return self

    def predict(self, texts) -> list[Pred]:
        # 与 TransformerCLS 保持同一套预处理，两个模型的判定口径才可比
        def prep(t: str) -> str:
            s = clean(t)
            if self.task == "sentiment":
                from normalize import normalize

                s = normalize(s)
            return s

        X = [prep(t) for t in texts]
        proba = self.model.predict_proba(X)
        classes = self.model.classes_
        idx = proba.argmax(axis=1)
        return [
            Pred(label=str(classes[i]), confidence=float(proba[j, i]))
            for j, i in enumerate(idx)
        ]

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vec": self._vec, "model": self.model}, path)

    @classmethod
    def load(cls, task: str, path: Path) -> "BaselineTFIDF":
        import joblib

        obj = joblib.load(path)
        inst = cls.__new__(cls)
        inst.task = task
        inst.name = "baseline-tfidf"
        inst._vec = obj["vec"]
        inst.model = obj["model"]
        return inst


# ---------------------------------------------------------------- transformer
class TransformerCLS:
    """预训练中文模型 + 线性头微调。

    默认底座用 hfl/chinese-roberta-wwm-ext 或 bert-base-chinese，
    小数据集下 3 epoch + lr 2e-5 就够；这是 BERT 微调的标准起点。
    """

    def __init__(
        self,
        task: str,
        backbone: str = "bert-base-chinese",
        name: str = "finetuned-v1",
        max_len: int = 128,
        epochs: int = 3,
        lr: float = 2e-5,
        batch: int = 16,
        dropout: float = 0.3,
    ):
        self.task = task
        self.backbone = backbone
        self.name = name
        self.max_len = max_len
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.dropout = dropout
        self.labels: list[str] | None = None
        self.tok = None
        self.model = None
        self.head = None

    def _build(self):
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.backbone)
        n = len(self.labels)
        self.model = AutoModel.from_pretrained(self.backbone)
        # 分类头带 dropout：小数据上（1500 条）不加的话第二个 epoch 就把训练集背下来，
        # 表现为 loss 掉到 0.0005、测试集置信度全是 1.0 —— 过拟合的典型信号。
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.model.config.hidden_size, n),
        )

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        """训练。若给 val_*，则按验证集 macro-F1 保留最优 epoch（早停），
        这样不用手工猜 epoch 数，也不至于把训练集背下来。
        """
        import copy

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset

        self.labels = sorted(set(labels))
        lab2id = {l: i for i, l in enumerate(self.labels)}
        self._build()
        tok, model, head, max_len = self.tok, self.model, self.head, self.max_len

        class DS(Dataset):
            def __init__(self, xs, ys):
                self.enc = tok(
                    xs, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt",
                )
                self.ys = torch.tensor([lab2id[y] for y in ys])

            def __len__(self):
                return len(self.ys)

            def __getitem__(self, i):
                return {k: v[i] for k, v in self.enc.items()}, self.ys[i]

        ds = DS([clean(t) for t in texts], labels)
        # 用独立 generator 控制打乱顺序，配合 torch.manual_seed 保证可复现
        g = torch.Generator().manual_seed(getattr(self, "seed", 42))
        dl = DataLoader(ds, batch_size=self.batch, shuffle=True, generator=g)

        # 分层学习率：BERT 本体用小的 lr（保护预训练知识），
        # 新初始化的分类头用大 10 倍的 lr（它需要从零学起）。
        # 统一用一个 lr 时，小数据上 head 几乎学不动，置信度会挤在 0.4~0.5。
        opt = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": self.lr},
                {"params": head.parameters(), "lr": self.lr * 10},
            ],
            weight_decay=0.01,
        )
        lossf = nn.CrossEntropyLoss()
        n_steps = len(dl) * self.epochs

        # 线性 warmup + 衰减：前 10% 步数逐步升到目标 lr，避免早期梯度震荡
        def lr_lambda(step: int) -> float:
            warm = max(1, int(n_steps * 0.1))
            if step < warm:
                return step / warm
            return max(0.0, 1.0 - (step - warm) / max(1, n_steps - warm))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        model.train()
        best_f1, best_state = -1.0, None
        for ep in range(self.epochs):
            tot, n = 0.0, 0
            for batch, y in dl:
                out = model(**batch).last_hidden_state[:, 0]  # [CLS]
                logits = head(out)
                loss = lossf(logits, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()), 1.0
                )
                opt.step()
                sched.step()
                tot += loss.item() * len(y)
                n += len(y)

            line = f"  epoch {ep + 1}/{self.epochs} loss={tot / n:.4f}"
            if val_texts is not None and val_labels is not None:
                from sklearn.metrics import f1_score

                vp = [p.label for p in self.predict(val_texts)]
                vf1 = f1_score(val_labels, vp, average="macro", zero_division=0)
                line += f"  val_macroF1={vf1:.4f}"
                if vf1 > best_f1:
                    best_f1 = vf1
                    best_state = copy.deepcopy(
                        {"model": model.state_dict(), "head": head.state_dict()}
                    )
            print(line)

        # 回滚到验证集最优的那个 epoch
        if best_state is not None:
            model.load_state_dict(best_state["model"])
            head.load_state_dict(best_state["head"])
            print(f"  已回滚到最优 epoch（val macroF1={best_f1:.4f}）")
        return self

    def predict(self, texts) -> list[Pred]:
        import torch

        self.model.eval()
        out_all: list[Pred] = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                chunk = [self._prep(t) for t in texts[i : i + 32]]
                enc = self.tok(
                    chunk, truncation=True, padding=True,
                    max_length=self.max_len, return_tensors="pt",
                )
                logits = self.head(self.model(**enc).last_hidden_state[:, 0])
                proba = torch.softmax(logits, dim=-1).numpy()
                for row in proba:
                    j = int(row.argmax())
                    out_all.append(Pred(self.labels[j], float(row[j])))
        return out_all

    def _prep(self, text: str) -> str:
        """推理前的预处理：清洗 + （情感任务）程度副词归一化。

        归一化只在**推理**时生效，训练数据保持原始表达，
        否则会把"好差"改写成"很差"，丢掉真实的语言分布。
        """
        t = clean(text)
        if self.task == "sentiment":
            from normalize import normalize

            t = normalize(t)
        return t

    # ---- 持久化：模型权重 + 分类头 + 标签表 + 超参
    def save(self, path: Path) -> None:
        """保存到目录：backbone 权重/分词器/head/标签。

        不直接存整个 model 对象，是因为 transformers 的类结构跨版本会变，
        存 state_dict + 配置更稳。
        """
        import torch

        path = Path(path)
        (path / "backbone").mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "backbone")
        self.tok.save_pretrained(path / "backbone")
        torch.save(self.head.state_dict(), path / "head.pt")
        (path / "meta.json").write_text(
            json.dumps({
                "task": self.task, "name": self.name, "labels": self.labels,
                "backbone": self.backbone, "max_len": self.max_len,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, path: Path) -> "TransformerCLS":
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.task = meta["task"]
        self.name = meta["name"]
        self.labels = meta["labels"]
        self.max_len = meta["max_len"]
        self.backbone = meta["backbone"]

        self.tok = AutoTokenizer.from_pretrained(path / "backbone")
        self.model = AutoModel.from_pretrained(path / "backbone")
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.model.config.hidden_size, len(self.labels)),
        )
        self.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
        self.model.eval()
        self.head.eval()
        return self


# ---------------------------------------------------------------- 数据切分
def load_split(seed: int = 42, test_size: float = 0.2):
    """读取已经按"主体句级别"划好的 train / test 切分。

    为什么不用 train_test_split 现场切：同一句主体会被多次复用，
    现场按会话随机切仍可能让同一句同时进 train 和 test，
    指标会虚高。数据生成阶段已经保证两个集合无句子重合。
    """
    import sqlite3

    df = pd.read_csv(DATA / "frames.csv")

    con = sqlite3.connect(ROOT / "db" / "customer_voice.sqlite")
    ids = pd.read_sql("SELECT msg_id, session_id, turn FROM messages", con)
    con.close()
    df = df.merge(ids, on=["session_id", "turn"], how="left")
    assert df.msg_id.notna().all(), "有消息没匹配到 msg_id"

    cust = df[df.role == "customer"].reset_index(drop=True)
    tr = cust[cust.split == "train"].reset_index(drop=True)
    te = cust[cust.split == "test"].reset_index(drop=True)
    va = cust[cust.split == "val"].reset_index(drop=True)
    return tr, te, va


def load_ood() -> pd.DataFrame:
    """分布外测试集：手工构造，不在任何片段池中。"""
    return pd.read_json(DATA / "test_hard.jsonl", lines=True)