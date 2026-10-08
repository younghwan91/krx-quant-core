"""공용 데이터 층 — DB 읽기(COPY), 열 단위 mmap 캐시, 하루치 장중 CSR, 일봉 패널, 거래일 달력.

세 소비 레포가 각자 들고 있던 로더(scalp-it 9벌·daytrade-it 4벌·swing-it 전 스크립트)를 대신한다.
옵트인이다 — 기존 어댑터를 깨지 않고 하나씩 갈아끼운다. Postgres 드라이버는
``pip install "krx-quant-core[db]"``(psycopg) 또는 이미 깔린 psycopg2.

캐시 위치는 ``$KQC_DATA`` 또는 ``~/.kqc/data``. 장중 데이터는 닫힌 날만, 일봉은 지문이 같을 때만
캐시를 쓴다.
"""

from .daily import DAILY_COLUMNS, DailyPanel, daily_panel, load_daily_bars, trading_calendar
from .db import connect, fetch_frame, is_postgres, resolve_dsn, sql_literal
from .intraday import (
    MINUTE_BARS,
    QUOTES,
    TICKS,
    CodeDay,
    DatasetSpec,
    is_closed_day,
    load_day,
    load_minute_bars,
    load_quotes,
    load_ticks,
)
from .store import ColumnStore, default_root

__all__ = [
    "DAILY_COLUMNS",
    "MINUTE_BARS",
    "QUOTES",
    "TICKS",
    "CodeDay",
    "ColumnStore",
    "DailyPanel",
    "DatasetSpec",
    "connect",
    "daily_panel",
    "default_root",
    "fetch_frame",
    "is_closed_day",
    "is_postgres",
    "load_daily_bars",
    "load_day",
    "load_minute_bars",
    "load_quotes",
    "load_ticks",
    "resolve_dsn",
    "sql_literal",
    "trading_calendar",
]
