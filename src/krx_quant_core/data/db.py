"""공용 DB(TimescaleDB ``kr_quant``) 접속과 빠른 읽기.

세 소비 레포가 ``KR_QUANT_DB`` 를 각자 찾고(.env 파싱 3벌), ``pd.read_sql`` 로 행 튜플을 만든다.
여기서는 Postgres 면 ``COPY (SELECT …) TO STDOUT (FORMAT csv)`` 로 받아 pandas C 파서에 바로
넘긴다 — 하루치 틱(190만 행) 기준 ``read_sql`` 7.6s → 4.5s, 행 튜플·object 열을 만들지 않는다.

드라이버는 psycopg(3) 가 있으면 그것, 없으면 psycopg2(소비 레포 셋 다 psycopg2). 그 밖의 DB-API
연결(sqlite — 테스트용)은 커서로 읽는다. SQL 은 이 패키지 안에서만 만들고 값은
:func:`sql_literal` 로 검증해 넣는다(``COPY`` 는 서버측 파라미터를 받지 않는다).
"""

from __future__ import annotations

import io
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["code_list", "connect", "fetch_frame", "is_postgres", "resolve_dsn", "sql_literal"]

#: DSN 을 담는 환경 변수(세 소비 레포·quant-airflow 공통).
DSN_ENV = "KR_QUANT_DB"
#: DSN 을 찾을 .env 파일을 직접 지정하는 환경 변수.
ENV_FILE_ENV = "KQC_ENV_FILE"

_DSN_LINE = re.compile(r"^\s*(?:export\s+)?KR_QUANT_DB\s*=\s*(.+?)\s*$", re.M)
_CODE = re.compile(r"^[0-9A-Z]{6}$")


def _dsn_from_env_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _DSN_LINE.search(text)
    if not m:
        return None
    return m.group(1).strip().strip("'\"") or None


def resolve_dsn(dsn: str | None = None, *, search_from: Path | str | None = None) -> str:
    """DSN 을 정한다: 인자 → ``$KR_QUANT_DB`` → ``$KQC_ENV_FILE`` → ``search_from``(기본 cwd)부터
    위로 올라가며 ``.env`` → 형제 ``quant-airflow/.env``.

    Raises:
        RuntimeError: 어디에도 없을 때(어디를 봤는지 메시지에 적는다).
    """
    if dsn:
        return dsn
    env = os.environ.get(DSN_ENV)
    if env:
        return env
    tried: list[str] = [f"${DSN_ENV}"]
    explicit = os.environ.get(ENV_FILE_ENV)
    candidates: list[Path] = [Path(explicit).expanduser()] if explicit else []
    start = Path(search_from) if search_from is not None else Path.cwd()
    for d in [start, *start.resolve().parents]:
        candidates.append(d / ".env")
        candidates.append(d / "quant-airflow" / ".env")
    for path in candidates:
        found = _dsn_from_env_file(path)
        if found:
            return found
        tried.append(str(path))
    raise RuntimeError(
        f"no {DSN_ENV}: set the env var or {ENV_FILE_ENV}, or put it in a .env "
        f"(looked at {', '.join(tried[:4])}, …)"
    )


def connect(dsn: str | None = None, **kwargs: Any) -> Any:
    """Postgres 연결 — psycopg(3) 가 있으면 그것, 없으면 psycopg2. DSN 은 :func:`resolve_dsn`."""
    target = resolve_dsn(dsn)
    try:
        import psycopg  # type: ignore[import-not-found]

        return psycopg.connect(target, **kwargs)
    except ImportError:
        pass
    try:
        import psycopg2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "no Postgres driver: pip install 'krx-quant-core[db]' (psycopg) or psycopg2"
        ) from exc
    return psycopg2.connect(target, **kwargs)


def is_postgres(conn: Any) -> bool:
    """psycopg(3)·psycopg2 연결인가."""
    mod = type(conn).__module__
    return mod.startswith("psycopg2") or mod.startswith("psycopg")


def sql_literal(value: Any) -> str:
    """``COPY`` 문에 넣을 리터럴. 날짜·시각·숫자·6자리 종목코드만 받는다(그 밖은 ``ValueError``)."""
    if isinstance(value, bool):
        raise ValueError("bool is not a SQL literal here")
    if isinstance(value, datetime):
        return f"'{value.isoformat(sep=' ')}'"
    if isinstance(value, date):
        return f"'{value.isoformat()}'"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"refusing to inline non-finite {value!r}")
        return repr(v)
    if isinstance(value, str) and _CODE.match(value):
        return f"'{value}'"
    raise ValueError(f"refusing to inline {value!r} into SQL")


def code_list(codes: Sequence[str]) -> str:
    """``('005930','000660')`` — 종목코드 IN 목록(각각 :func:`sql_literal` 검증)."""
    if not codes:
        raise ValueError("codes must be non-empty")
    return "(" + ",".join(sql_literal(c) for c in codes) + ")"


def _copy_bytes(conn: Any, sql: str) -> io.BytesIO:
    buf = io.BytesIO()
    copy_sql = f"COPY ({sql}) TO STDOUT WITH (FORMAT csv)"
    if type(conn).__module__.startswith("psycopg2"):
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buf)
    else:
        with conn.cursor() as cur, cur.copy(copy_sql) as cp:
            for chunk in cp:
                buf.write(chunk)
    buf.seek(0)
    return buf


def fetch_frame(
    conn: Any,
    sql: str,
    columns: Sequence[str],
    *,
    dtypes: Mapping[str, Any] | None = None,
    timestamps: Sequence[str] = (),
) -> pd.DataFrame:
    """``SELECT`` 결과를 DataFrame 으로 — Postgres 는 COPY CSV, 그 밖은 커서.

    Args:
        sql: ``columns`` 순서로 열을 내는 ``SELECT`` 문(끝 세미콜론 없이).
        dtypes: 열별 numpy 자료형(결측이 있을 수 있는 정수 열은 float 로 줄 것).
        timestamps: ``datetime64`` 로 파싱할 열(날짜 열 포함).
    """
    cols = list(columns)
    dt = dict(dtypes or {})
    if is_postgres(conn):
        buf = _copy_bytes(conn, sql)
        if buf.getbuffer().nbytes == 0:
            df = pd.DataFrame({c: pd.Series(dtype=dt.get(c, object)) for c in cols})
        else:
            df = pd.read_csv(
                buf, header=None, names=cols, dtype={c: dt[c] for c in cols if c in dt},
                engine="c", keep_default_na=False, na_values=[""],
            )  # fmt: skip
    else:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
        finally:
            cur.close()
        df = pd.DataFrame(rows, columns=cols)
        for c, t in dt.items():
            if c in df.columns and c not in timestamps:
                df[c] = df[c].astype(t)
    for c in timestamps:
        # Postgres 는 소수 초 끝 0 을 지워 "…:02"·"…:02.5" 가 섞여 나온다 — 형식 추론 대신 ISO8601.
        df[c] = pd.to_datetime(df[c], format="ISO8601")
    return df
