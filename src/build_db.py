"""SQLite 建库与导入脚本。

表设计遵循 3NF：会话 / 消息 / 预测分表，避免把模型预测写回标注表。
这样同一份数据可以挂多个模型版本的结果，做 A/B 对比。

表结构：
  sessions     一次会话             session_id PK
  messages     会话内每条消息         msg_id PK, session_id FK
  predictions  模型对某条消息的预测    pred_id PK, msg_id FK, model_version
  evaluations  某模型的整体指标        eval_id PK, model_version

索引策略（面试会问）：
  - messages.session_id 上建索引，因为看板按会话聚合查询最频繁
  - predictions(model_version, msg_id) 联合索引，支持按模型过滤后 JOIN
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "customer_voice.sqlite"
DATA = ROOT / "data"

DDL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS evaluations;

CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    channel     TEXT NOT NULL CHECK (channel IN ('text','image','review')),
    created_at  TEXT NOT NULL
);

CREATE TABLE messages (
    msg_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    turn        INTEGER NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('customer','agent')),
    text        TEXT NOT NULL,
    intent      TEXT,          -- 仅 customer 有金标
    sentiment   TEXT,          -- 同上
    split       TEXT NOT NULL CHECK (split IN ('train','val','test')),
    UNIQUE (session_id, turn)
);

CREATE TABLE predictions (
    pred_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_id         INTEGER NOT NULL REFERENCES messages(msg_id) ON DELETE CASCADE,
    model_version  TEXT NOT NULL,
    task           TEXT NOT NULL CHECK (task IN ('intent','sentiment')),
    label          TEXT NOT NULL,
    confidence     REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1)
);

CREATE TABLE evaluations (
    eval_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version  TEXT NOT NULL,
    task           TEXT NOT NULL,
    metric         TEXT NOT NULL,
    value          REAL NOT NULL
);

CREATE INDEX idx_messages_session  ON messages(session_id);
CREATE INDEX idx_messages_role     ON messages(role);
CREATE INDEX idx_messages_split    ON messages(split);
CREATE INDEX idx_pred_ver_msg      ON predictions(model_version, msg_id);
CREATE INDEX idx_eval_version      ON evaluations(model_version, task);
"""


def build(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(DATA / "frames.csv")

    con = sqlite3.connect(db_path)
    con.executescript(DDL)

    # ---- sessions
    sessions = (
        frames[["session_id", "channel"]]
        .drop_duplicates("session_id")
        .assign(created_at="2025-01-01")
    )
    sessions.to_sql("sessions", con, if_exists="append", index=False)

    # ---- messages（客服的 intent/sentiment 存 NULL 而不是空串）
    msgs = frames[["session_id", "turn", "role", "text", "intent", "sentiment", "split"]].copy()
    msgs.loc[msgs.role == "agent", ["intent", "sentiment"]] = None
    msgs["intent"] = msgs["intent"].replace({float("nan"): None})
    msgs["sentiment"] = msgs["sentiment"].replace({float("nan"): None})
    msgs.to_sql("messages", con, if_exists="append", index=False)

    con.commit()

    # 自检
    n_sess = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    n_msg = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    n_cust = con.execute("SELECT COUNT(*) FROM messages WHERE role='customer'").fetchone()[0]
    print(f"入库: {n_sess} 会话, {n_msg} 消息（客户 {n_cust}）")
    print(f"DB: {db_path}")
    con.close()


if __name__ == "__main__":
    build()