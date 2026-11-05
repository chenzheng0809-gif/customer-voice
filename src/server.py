"""Web 后端：只读 SQLite，用标准库 http.server，零额外依赖。

为什么不用 Flask/FastAPI：
  这个项目要展示的是"能用 SQL 取数、能起服务"，不是框架熟练度。
  标准库 server 反而让面试官看到你清楚 HTTP 是怎么转的。
  如果后续要加异步/鉴权，再把 handler 换成 FastAPI 即可，路由逻辑不变。

接口：
  GET /api/overview          总览：会话数、消息数、情感分布
  GET /api/sessions          会话列表（支持 ?risk=1 只看高风险）
  GET /api/session/<id>      单会话明细（消息 + 预测）
  GET /api/confusion         混淆矩阵
  GET /api/badcases          模型预测错误的样本
  GET /api/models            模型指标对比
  POST /api/analyze          在线分析一条文本（调 baseline 模型）
"""

from __future__ import annotations

import json
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "db" / "customer_voice.sqlite"
WEB = ROOT / "web"
sys.path.insert(0, str(ROOT / "src"))

PORT = 8765


def q(sql: str, args=()) -> list[dict]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(sql, args).fetchall()]
    con.close()
    return rows


def api_overview() -> dict:
    return {
        "counts": q("""
            SELECT
              (SELECT COUNT(*) FROM sessions) AS sessions,
              (SELECT COUNT(*) FROM messages) AS messages,
              (SELECT COUNT(*) FROM messages WHERE role='customer') AS customer_msgs,
              (SELECT COUNT(DISTINCT model_version) FROM predictions) AS models
        """)[0],
        "sentiment": q("""
            SELECT sentiment AS label, COUNT(*) AS n
            FROM messages WHERE role='customer'
            GROUP BY sentiment ORDER BY n DESC
        """),
        "intent": q("""
            SELECT intent AS label, COUNT(*) AS n
            FROM messages WHERE role='customer'
            GROUP BY intent ORDER BY n DESC
        """),
        "cross": q("""
            SELECT intent, sentiment, COUNT(*) AS n
            FROM messages WHERE role='customer'
            GROUP BY intent, sentiment
        """),
    }


def api_sessions(risk_only: bool) -> list[dict]:
    sql = """
        SELECT s.session_id, s.channel,
               COUNT(m.msg_id) AS msgs,
               SUM(CASE WHEN m.sentiment='负面' THEN 1 ELSE 0 END) AS neg,
               ROUND(SUM(CASE WHEN m.sentiment='负面' THEN 1 ELSE 0 END) * 1.0
                     / COUNT(m.msg_id), 3) AS neg_ratio
        FROM sessions s
        JOIN messages m ON m.session_id = s.session_id AND m.role='customer'
        GROUP BY s.session_id, s.channel
    """
    if risk_only:
        sql += " HAVING neg_ratio >= 0.5 AND msgs >= 2"
    sql += " ORDER BY neg_ratio DESC, msgs DESC LIMIT 60"
    return q(sql)


def api_session(sid: str) -> dict:
    msgs = q("""
        SELECT msg_id, turn, role, text, intent, sentiment
        FROM messages WHERE session_id=? ORDER BY turn
    """, (sid,))
    preds = q("""
        SELECT msg_id, task, label, confidence, model_version
        FROM predictions WHERE msg_id IN (
            SELECT msg_id FROM messages WHERE session_id=?
        ) ORDER BY msg_id
    """, (sid,))
    by_msg: dict[int, dict] = {}
    for p in preds:
        by_msg.setdefault(p["msg_id"], {})[p["task"]] = p
    return {"session_id": sid, "messages": msgs, "predictions": by_msg}


def _best_versions() -> dict[str, str]:
    """按 evaluations 表里的 macro_f1 选每个任务的最优模型版本。

    不把版本名写死在 SQL 里，否则重新训练换了版本名就得改代码。
    """
    rows = q("""
        SELECT task, model_version, value
        FROM evaluations WHERE metric='macro_f1'
        ORDER BY task, value DESC
    """)
    best: dict[str, str] = {}
    for r in rows:
        best.setdefault(r["task"], r["model_version"])
    return best


def api_badcases() -> list[dict]:
    """模型预测错误且置信度高的样本 —— 最值得人工复查的那批。"""
    v = _best_versions()
    vi = v.get("intent", "baseline-intent")
    vs = v.get("sentiment", "baseline-sentiment")
    return q("""
        SELECT m.msg_id, m.text, m.intent AS gold_intent, m.sentiment AS gold_sent,
               p_intent.label AS pred_intent, p_intent.confidence AS intent_conf,
               p_sent.label AS pred_sent, p_sent.confidence AS sent_conf,
               p_intent.model_version AS intent_model,
               p_sent.model_version  AS sent_model
        FROM messages m
        JOIN predictions p_intent
          ON p_intent.msg_id=m.msg_id AND p_intent.task='intent'
         AND p_intent.model_version=?
        JOIN predictions p_sent
          ON p_sent.msg_id=m.msg_id AND p_sent.task='sentiment'
         AND p_sent.model_version=?
        WHERE m.role='customer' AND m.split='test'
          AND (p_intent.label <> m.intent OR p_sent.label <> m.sentiment)
        ORDER BY (p_intent.confidence + p_sent.confidence) DESC
        LIMIT 40
    """, (vi, vs))


def api_models() -> list[dict]:
    return q("""
        SELECT model_version, task, metric, value
        FROM evaluations ORDER BY task, model_version, metric
    """)


def api_confusion() -> list[dict]:
    """用 SQL 现算混淆矩阵，而不是读 json —— 展示 GROUP BY 能直接出矩阵。"""
    vi = _best_versions().get("intent", "baseline-intent")
    return q("""
        SELECT m.intent AS gold, p.label AS pred, COUNT(*) AS n
        FROM messages m
        JOIN predictions p ON p.msg_id=m.msg_id
                          AND p.task='intent'
                          AND p.model_version=?
        WHERE m.role='customer' AND m.split='test'
        GROUP BY gold, pred
    """, (vi,))


_MODELS: dict = {}


def api_analyze(text: str) -> dict:
    """在线分析：懒加载最优模型，避免每次请求都重训。

    优先用评估表里 macro-F1 最高的 bert 模型；它需要 torch + 权重文件，
    没有的话回退到 baseline（TF-IDF），保证在没有 GPU/模型的机器上也能演示。
    """
    if not _MODELS:
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from model import BaselineTFIDF, load_split

        for task in ("intent", "sentiment"):
            ckpt = ROOT / "models" / f"bert-{task}"
            if ckpt.exists():
                try:
                    from model import TransformerCLS

                    m = TransformerCLS(task, name=f"bert-{task}")
                    m.load(ckpt)
                    _MODELS[task] = m
                    continue
                except Exception:
                    pass  # 回退 baseline

            base_path = ROOT / "models" / f"baseline-{task}.joblib"
            if base_path.exists():
                _MODELS[task] = BaselineTFIDF.load(task, base_path)
            else:
                tr, _, _ = load_split()
                m = BaselineTFIDF(task, name=f"baseline-{task}")
                m.fit(tr.text.tolist(), tr[task].tolist())
                m.save(base_path)
                _MODELS[task] = m

    out = {"text": text, "models": {t: m.name for t, m in _MODELS.items()}}
    for task in ("intent", "sentiment"):
        p = _MODELS[task].predict([text])[0]
        out[task] = {"label": p.label, "confidence": round(p.confidence, 4)}
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音
        pass

    def _send(self, obj, status=200, ctype="application/json; charset=utf-8"):
        body = obj if isinstance(obj, bytes) else json.dumps(
            obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        try:
            if p == "/api/overview":
                return self._send(api_overview())
            if p == "/api/sessions":
                risk = parse_qs(u.query).get("risk", ["0"])[0] == "1"
                return self._send(api_sessions(risk))
            if p.startswith("/api/session/"):
                return self._send(api_session(p.rsplit("/", 1)[-1]))
            if p == "/api/badcases":
                return self._send(api_badcases())
            if p == "/api/models":
                return self._send(api_models())
            if p == "/api/confusion":
                return self._send(api_confusion())
            # 静态文件
            fname = "index.html" if p in ("/", "") else p.lstrip("/")
            fpath = (WEB / fname).resolve()
            if not str(fpath).startswith(str(WEB.resolve())) or not fpath.exists():
                return self._send({"error": "not found"}, 404)
            ctype = {"html": "text/html; charset=utf-8",
                     "js": "application/javascript; charset=utf-8",
                     "css": "text/css; charset=utf-8"}.get(
                fpath.suffix.lstrip("."), "application/octet-stream")
            return self._send(fpath.read_bytes(), ctype=ctype)
        except Exception as e:  # 开发期把错误返回前端，方便调试
            return self._send({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/analyze":
            return self._send({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        text = (payload.get("text") or "").strip()
        if not text:
            return self._send({"error": "empty text"}, 400)
        try:
            return self._send(api_analyze(text))
        except Exception as e:
            return self._send({"error": f"{type(e).__name__}: {e}"}, 500)


if __name__ == "__main__":
    print(f"看板地址: http://127.0.0.1:{PORT}/")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()