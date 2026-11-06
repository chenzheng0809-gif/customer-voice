"""校验 db/queries.sql 里的每条查询都能在真实库上执行。

用途：CI 里跑，防止改表结构后 SQL 悄悄腐烂（语法对但列名/表名失效）。

用法：
  python src/check_sql.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "db" / "customer_voice.sqlite"
SQL_FILE = ROOT / "db" / "queries.sql"


def split_statements(sql: str) -> list[str]:
    """按 sqlite3.complete_statement 切分，正确处理字符串里的分号。"""
    stmts: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        # 跳过纯注释行，但保留行内注释
        if line.strip().startswith("--"):
            continue
        buf.append(line)
        cur = "\n".join(buf)
        if sqlite3.complete_statement(cur):
            stmts.append(cur.strip())
            buf = []
    return stmts


def main() -> int:
    if not DB.exists():
        print(f"数据库不存在：{DB}\n请先执行 src/build_db.py")
        return 1

    con = sqlite3.connect(DB)
    stmts = split_statements(SQL_FILE.read_text(encoding="utf-8"))

    ok = skipped = failed = 0
    for s in stmts:
        head = next((ln for ln in s.splitlines() if ln.strip()), "")[:52]
        upper = head.upper()
        # EXPLAIN 和图例性语句不进 CI 校验（依赖运行环境差异）
        if upper.startswith(("EXPLAIN", "BEGIN", "COMMIT", "ROLLBACK")):
            skipped += 1
            continue
        try:
            rows = con.execute(s).fetchall()
            print(f"  OK  {len(rows):>5} 行  {head}")
            ok += 1
        except sqlite3.Error as e:
            print(f"  ERR {e}\n      <<{head}>>")
            failed += 1

    con.close()
    print(f"\n通过 {ok} / 跳过 {skipped} / 失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())