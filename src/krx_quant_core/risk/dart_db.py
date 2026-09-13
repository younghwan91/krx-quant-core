"""진입 리스크 게이트용 최근 DART 공시 — scalp-it 의 ``data/dart.db`` 에서 읽는다.

daytrade-it ``infrastructure/external/dart_disclosure_db.py`` 를 동작 그대로 옮겼다.
바뀐 것은 import 뿐이다: ``KST``·``now_kst`` 는 코어 ``market.session`` 에서,
분류기는 코어 :mod:`krx_quant_core.risk.disclosure` 에서 가져온다.

왜 이 소스인가: quant-airflow ``kr_quant.disclosures`` 는 하루 두 번 수집이라 당일
공시가 없다. scalp-it cron 은 평일 08:40(7일치)과 09:00~15:50 10분마다 OpenDART 로
``data/dart.db`` 를 갱신한다 — 같은 호스트에서 당일 공시를 가진 유일한 데이터다.

**fail-closed 가 설계다.** 아무도 갱신하지 않는 파일 위의 게이트는 배선된 것처럼
보이면서 모든 진입을 통과시킨다. 파일이 없거나·못 읽거나·오래됐으면
:class:`DisclosureDataUnavailable` 을 던지고, 호출부는 **진입**을 거부한다. 청산은
이 모듈을 보지 않는다.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Callable
from pathlib import Path

from ..market.session import KST, now_kst
from .disclosure import DisclosureSeverity, classify_disclosure_severity

__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_MAX_STALENESS",
    "DartDisclosureDB",
    "DisclosureDataUnavailable",
]

#: 장중 갱신 주기가 10분이다. 30분이면 느리거나 한 번 건너뛴 실행까지 버틴다.
#: 08:40 실행이 첫 진입(09:00:30)을 이 창 안에 둔다.
DEFAULT_MAX_STALENESS = dt.timedelta(minutes=30)

#: 오늘 포함 달력일. 주초의 상장적격성 심사·횡령 공시도 신규 진입엔 여전히 반대 근거다.
DEFAULT_LOOKBACK_DAYS = 7


class DisclosureDataUnavailable(RuntimeError):
    """공시 소스가 지금 이 종목을 보증할 수 없다."""


class DartDisclosureDB:
    def __init__(
        self,
        path: Path,
        *,
        max_staleness: dt.timedelta = DEFAULT_MAX_STALENESS,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        clock: Callable[[], dt.datetime] = now_kst,
    ) -> None:
        self._path = Path(path)
        self._max_staleness = max_staleness
        self._lookback_days = lookback_days
        self._clock = clock

    def hard_disclosure(self, ticker: str) -> str | None:
        """lookback 안에서 ``ticker`` 의 가장 최근 HARD 공시 제목, 없으면 None."""
        now = self._clock()
        if not self._path.exists():
            raise DisclosureDataUnavailable(f"dart.db not found: {self._path}")
        refreshed = dt.datetime.fromtimestamp(self._path.stat().st_mtime, KST)
        age = now - refreshed
        if age > self._max_staleness:
            raise DisclosureDataUnavailable(
                f"dart.db stale: last refreshed {refreshed:%Y-%m-%d %H:%M} KST "
                f"({int(age.total_seconds() // 60)} min ago, limit "
                f"{int(self._max_staleness.total_seconds() // 60)} min)"
            )
        since = (now.date() - dt.timedelta(days=self._lookback_days)).strftime("%Y%m%d")
        try:
            con = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
            try:
                rows = con.execute(
                    "SELECT report_nm FROM disclosures WHERE code = ? AND rcept_dt >= ? "
                    "ORDER BY rcept_dt DESC, rcept_no DESC",
                    (ticker, since),
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error as exc:
            raise DisclosureDataUnavailable(f"dart.db unreadable: {exc}") from exc
        for (report_nm,) in rows:
            if classify_disclosure_severity(report_nm) is DisclosureSeverity.HARD:
                return str(report_nm)
        return None
