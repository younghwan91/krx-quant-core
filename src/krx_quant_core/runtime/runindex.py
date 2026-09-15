"""실행 기록 Postgres 색인(``kqc_runs``) — 세 레포를 가로질러 조회하려는 용도.

정본은 레포의 ``RUNS.jsonl`` 이다. 여기는 **실패해도 된다**: ``KR_QUANT_DB`` 가 없거나, psycopg 가
안 깔렸거나(``[db]`` extra), 연결이 안 되면 경고만 남기고 ``False`` 를 돌려준다. 빠진 행은
:func:`sync` 가 JSONL 에서 다시 채운다.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

__all__ = ["DDL", "sync", "upsert"]

log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS kqc_runs (
    run_id      text PRIMARY KEY,
    label       text NOT NULL,
    repo        text,
    git_sha     text,
    host        text,
    status      text,
    fingerprint text,
    started_at  timestamptz,
    ended_at    timestamptz,
    result      jsonb,
    start_row   jsonb,
    end_row     jsonb
)
"""

_UPSERT_START = """
INSERT INTO kqc_runs (run_id, label, repo, git_sha, host, fingerprint, started_at, start_row)
VALUES (%(run_id)s, %(label)s, %(repo)s, %(git_sha)s, %(host)s, %(fingerprint)s, %(ts)s, %(row)s)
ON CONFLICT (run_id) DO UPDATE SET label = EXCLUDED.label, repo = EXCLUDED.repo,
    git_sha = EXCLUDED.git_sha, host = EXCLUDED.host, fingerprint = EXCLUDED.fingerprint,
    started_at = EXCLUDED.started_at, start_row = EXCLUDED.start_row
"""

_UPSERT_END = """
INSERT INTO kqc_runs (run_id, label, status, ended_at, result, end_row)
VALUES (%(run_id)s, %(label)s, %(status)s, %(ts)s, %(result)s, %(row)s)
ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status, ended_at = EXCLUDED.ended_at,
    result = EXCLUDED.result, end_row = EXCLUDED.end_row
"""


def _params(row: dict[str, Any]) -> dict[str, Any]:
    p = {k: row.get(k) for k in ("run_id", "label", "repo", "git_sha", "host", "fingerprint",
                                 "ts", "status")}  # fmt: skip
    p["result"] = json.dumps(row.get("result"), ensure_ascii=False, default=str)
    p["row"] = json.dumps(row, ensure_ascii=False, default=str)
    return p


def _connect() -> Any:
    dsn = os.environ.get("KR_QUANT_DB")
    if not dsn:
        return None
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError:
        log.debug("psycopg not installed — skipping kqc_runs index")
        return None
    return psycopg.connect(dsn, connect_timeout=3)


def _write(conn: Any, rows: list[dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
        for row in rows:
            sql = _UPSERT_START if row.get("event") == "start" else _UPSERT_END
            cur.execute(sql, _params(row))
    conn.commit()


def upsert(row: dict[str, Any]) -> bool:
    """행 하나를 색인에 반영. 성공이면 ``True``, 건너뛰거나 실패하면 ``False``(경고 로그)."""
    try:
        conn = _connect()
        if conn is None:
            return False
        with conn:
            _write(conn, [row])
        return True
    except Exception as exc:  # 색인은 실행을 막지 않는다
        log.warning("kqc_runs index upsert failed (JSONL is still written): %s", exc)
        return False


def sync(repo_root: Path | str) -> int:
    """``research/runs/*/RUNS.jsonl`` 전체를 색인에 다시 쓴다. 반영한 행 수. DB 없으면 예외."""
    from .gitstate import runs_root

    rows: list[dict[str, Any]] = []
    for path in sorted(runs_root(repo_root).glob("*/RUNS.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    conn = _connect()
    if conn is None:
        raise RuntimeError("KR_QUANT_DB unset or psycopg missing — install krx-quant-core[db]")
    with conn:
        _write(conn, rows)
    return len(rows)
