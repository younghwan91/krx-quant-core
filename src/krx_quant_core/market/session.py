"""KRX 정규장 세션 시각과 KST 시계.

daytrade-it 은 ``brokers/base.py`` 와 ``quant_airflow_db_client.py`` 에 ``KST`` 를
두 번 정의했고, scalp-it 은 naive 로컬시각에 KST 를 가정했다. 여기가 정본이다.

시각 문자열 헬퍼(``past_cutoff``·``in_window``)는 scalp-it ``realtime/leader_scan.py``
에서 옮겼다 — 설정 오타 하나로 거래일을 날리지 않도록 **형식 오류는 막지 않는
쪽**으로 동작하는 규약을 그대로 유지한다.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

__all__ = [
    "CLOSING_AUCTION_START",
    "KST",
    "PRE_OPEN_AUCTION_START",
    "REGULAR_CLOSE",
    "REGULAR_OPEN",
    "in_regular_session",
    "in_window",
    "now_kst",
    "parse_hhmm",
    "past_cutoff",
]

KST = ZoneInfo("Asia/Seoul")

#: 장전 동시호가 시작(08:30). 08:30~09:00 호가 접수, 09:00 단일가 체결.
PRE_OPEN_AUCTION_START = time(8, 30)
#: 정규장 시작.
REGULAR_OPEN = time(9, 0)
#: 장마감 동시호가 시작(15:20). 이후 15:30 까지 접수만, 15:30 단일가 체결.
CLOSING_AUCTION_START = time(15, 20)
#: 정규장 종료(단일가 체결 시각).
REGULAR_CLOSE = time(15, 30)


def now_kst() -> datetime:
    """현재 KST 시각(tz-aware)."""
    return datetime.now(KST)


def parse_hhmm(text: str) -> time:
    """``"0930"``·``"09:30"`` → ``time(9, 30)``. 형식이 틀리면 ``ValueError``."""
    digits = text.strip().replace(":", "")
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"expected HHMM or HH:MM, got {text!r}")
    return time(int(digits[:2]), int(digits[2:]))


def past_cutoff(now_hhmm: str, cutoff: str) -> bool:
    """``now_hhmm`` 이 ``cutoff``(예 ``"1000"``)을 **지났나**.

    빈 값이나 형식 오류면 ``False`` — 막지 않는다(오타로 하루를 날리지 않는다).
    """
    cut = cutoff.strip().replace(":", "")
    if not cut or not cut.isdigit():
        return False
    return now_hhmm.replace(":", "") > cut


def in_window(now_hhmm: str, window: str) -> bool:
    """``now_hhmm`` 이 ``window``(예 ``"0900-0930"``, 양끝 포함) 안인가.

    빈 문자열이면 항상 ``True``(창 없음). 형식 오류도 ``True`` — 막지 않는다.
    """
    if not window.strip():
        return True
    try:
        start, end = (p.strip().replace(":", "") for p in window.split("-", 1))
        return start <= now_hhmm.replace(":", "") <= end
    except ValueError:
        return True


def in_regular_session(ts: datetime) -> bool:
    """``ts`` 가 정규장(09:00 이상 15:30 미만, KST) 안인가. 휴장일은 보지 않는다.

    naive datetime 은 KST 로 간주한다.
    """
    local = ts.astimezone(KST) if ts.tzinfo else ts
    t = local.time()
    return local.weekday() < 5 and REGULAR_OPEN <= t < REGULAR_CLOSE
