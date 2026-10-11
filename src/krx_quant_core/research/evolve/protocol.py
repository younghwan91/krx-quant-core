"""세대 프로토콜 — 결과 전 등록 원장·잡음 규칙·짝 레이싱·봉인·결정성·출처.

swing-it ``research/logs/sepa_explore/META_EVOLUTION.md`` 의 절차를 일반화했다.

- **결과 전 등록.** 후보를 돌리기 전에 :meth:`EvoLedger.register` 로 적고, 끝나면
  :meth:`EvoLedger.result`. 결과 없는 등록(죽은 실행)도 DSR N 에 센다 — N = **등록 수**(결과 줄은
  세지 않는다).
- **잡음 규칙(호환).** :func:`distinguish` = swing-it 규칙: 평균 차 > ``k`` × 두 표준편차(ddof=1) 중
  큰 값. 표준편차는 시드 집합 수 n 으로 줄지 않으므로 이 규칙은 n 을 늘려도 검정력이 늘지 않는다.
- **짝 레이싱.** :func:`race` 는 모든 후보가 **같은 시드 집합**을 돌린다는 점을 써서 짝 차이의
  표준오차로 가른다: 리더와의 짝 차 평균 + ``k``·SE < 0 이면 탈락. n 이 늘면 SE 가 √n 으로 준다.

  공개할 편향(보고서에 같이 적는다): ① 살아남은 후보의 평균은 승자의 저주·선택적 정지로 위로
  치우친다 — 판정은 봉인 구간에서만. ② 작은 n 의 탈락은 되돌리지 않는다(min_sets 3 이면 짝 차 2
  자유도). ③ n 이 다른 평균끼리 비교하지 않는다 — 보고는 같은 시드 집합 위 짝 차이로만.
  ④ ``k``·``min_sets``·``max_sets``·시드 순서는 결과 전에 등록한다(``race(ledger=)``).
- **봉인.** :class:`Seal` — 경계 행 이후를 자르고, 청산 행이 경계 이상이면 :class:`SealBreach`.
- **결정성.** :func:`digest` — 같은 후보를 두 번 돌려 같은 digest 인지 본다.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from krx_quant_core.runtime.gitstate import git_dirty, git_head
from krx_quant_core.stats.trials import read_trials, record_trial

__all__ = [
    "EvoLedger",
    "RaceResult",
    "Seal",
    "SealBreach",
    "digest",
    "distinguish",
    "paired_diff",
    "provenance",
    "race",
    "summarize",
]


# ---------------------------------------------------------------------------
# 원장
# ---------------------------------------------------------------------------

class EvoLedger:
    """``stats.trials`` 원장 위 진화 기록 — 줄마다 ``role`` 이 ``"pre"``(결과 전) 또는 ``"result"``.

    ``register`` 가 돌려준 지문을 ``result(reg)`` 에 넘긴다 — 결과 줄은 그 등록을 지문으로 가리킨다
    (같은 이름·시드로 메뉴만 바꾼 두 등록이 서로의 결과로 덮이지 않게)."""

    _RESERVED = frozenset({"role", "candidate", "menu", "seeds", "rationale", "reg", "metrics"})

    def __init__(self, label: str, *, logs_dir: Path | str):
        self.label = label
        self.logs_dir = Path(logs_dir)

    def _check(self, extra: dict) -> None:
        bad = self._RESERVED & set(extra)
        if bad:
            raise ValueError(f"예약된 키: {sorted(bad)}")

    def register(self, candidate: str, *, menu: Any, seeds: Sequence[int], rationale: str = "",
                 **extra: Any) -> str:
        """결과 전 등록(돌리기 **전에** 부른다). 반환 = 지문(:meth:`result` 에 넘긴다)."""
        self._check(extra)
        cfg = {**extra, "role": "pre", "candidate": candidate, "menu": menu,
               "seeds": [int(s) for s in seeds], "rationale": rationale}
        return record_trial(self.label, cfg, logs_dir=self.logs_dir,
                            note=f"진화 후보 {candidate}(결과 전)")

    def result(self, reg: str, *, metrics: dict, **extra: Any) -> str:
        """``reg`` = 그 후보 :meth:`register` 의 지문. 원장에 없는 지문이면 ``KeyError``."""
        self._check(extra)
        pre = {c["fingerprint"]: c for c in self.registrations()}
        if reg not in pre:
            raise KeyError(f"등록 지문 {reg} 이 원장에 없다")
        cfg = {**extra, "role": "result", "reg": reg, "candidate": pre[reg]["candidate"],
               "seeds": pre[reg]["seeds"], "metrics": metrics}
        return record_trial(self.label, cfg, logs_dir=self.logs_dir,
                            note=f"진화 후보 {pre[reg]['candidate']} 결과")

    def rows(self) -> list[dict]:
        return read_trials(self.label, logs_dir=self.logs_dir)

    def registrations(self) -> list[dict]:
        """등록 줄의 config(``fingerprint`` 키를 붙여서)."""
        return [{**r["config"], "fingerprint": r["fingerprint"]} for r in self.rows()
                if r.get("config", {}).get("role") == "pre"]

    def pending(self) -> list[dict]:
        """결과 줄이 가리키지 않는 등록 — 죽은 실행. DSR N 에 그대로 센다."""
        done = {r["config"].get("reg") for r in self.rows()
                if r.get("config", {}).get("role") == "result"}
        return [c for c in self.registrations() if c["fingerprint"] not in done]

    def n_trials(self) -> int:
        """DSR N = 등록 수(완료 + 미완)."""
        return len(self.registrations())


# ---------------------------------------------------------------------------
# 잡음 규칙
# ---------------------------------------------------------------------------

def summarize(vals: Sequence[float]) -> dict:
    """평균·표준편차(ddof=1, n < 2 면 NaN)·n."""
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()) if len(a) else math.nan,
            "sd": float(a.std(ddof=1)) if len(a) >= 2 else math.nan, "n": int(len(a))}


def distinguish(a: Sequence[float], b: Sequence[float], k: float = 2.0) -> str | None:
    """swing-it 잡음 규칙 — 평균 차 > ``k`` × max(sd_a, sd_b) 면 큰 쪽(``"a"``/``"b"``), 아니면
    ``None``."""
    sa, sb = summarize(a), summarize(b)
    thr = k * max(sa["sd"], sb["sd"])
    d = sa["mean"] - sb["mean"]
    if not math.isfinite(thr) or abs(d) <= thr:
        return None
    return "a" if d > 0 else "b"


def paired_diff(a: dict, b: dict) -> dict:
    """같은 시드 집합 키 위 짝 차이 a − b → 평균·SE(ddof=1)·n(공통 키만)."""
    ks = sorted(set(a) & set(b))
    d = np.array([a[x] - b[x] for x in ks], float)
    n = len(d)
    se = float(d.std(ddof=1) / math.sqrt(n)) if n >= 2 else math.nan
    return {"mean": float(d.mean()) if n else math.nan, "se": se, "n": n}


@dataclass
class RaceResult:
    scores: dict[str, dict[int, float]]          # 후보 → {시드 오프셋: 메타 적합도}
    alive: list[str]
    eliminated: dict[str, int] = field(default_factory=dict)   # 후보 → 탈락한 라운드(집합 수)
    log: list[dict] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def leader(self) -> str:
        """살아 있는 후보 중 공통 시드 집합 평균 최대(동률 = 앞 순서)."""
        def key(c: str) -> tuple:
            return (np.mean(list(self.scores[c].values())), -self.alive.index(c))
        return max(self.alive, key=key)


def race(candidates: Sequence[str], seed_offsets: Sequence[int], *,
         run: Callable[[str, int], float] | None = None,
         run_round: Callable[[list[tuple[str, int]]], list[float]] | None = None,
         min_sets: int = 5, max_sets: int | None = None, k: float = 3.0,
         ledger: EvoLedger | None = None, name: str = "race") -> RaceResult:
    """짝 SE 레이싱(모듈 설명). 라운드마다 살아 있는 후보 모두에 다음 시드 오프셋 하나.

    ``run_round(pairs) -> values`` 를 주면 한 라운드를 한 번에(풀에서 병렬) 돌린다. 아니면
    ``run(c, off)``. ``ledger`` 를 주면 첫 라운드 **전에** 레이싱 인자(후보·시드 순서·k·
    min/max_sets)를 결과 전 줄로 적는다.

    기본값(k=3, min_sets=5)의 근거 — 모의(후보 6, 집합 10, 집합 공통 잡음 sd 1 + 후보별 잡음
    sd 1, 3,000회):

    - k=2·min_sets=3: 모두 같은 후보 5개 중 평균 2.4개를 잘못 떨군다. +0.5 sd 나은 후보도
      15% 떨군다.
    - k=3·min_sets=5: 0.7개·1.6%.

    라운드마다 다시 검사하고 리더를 최대로 뽑으므로 k·SE 의 명목 수준보다 훨씬 자주 떨군다.
    기본값을 바꾸려면 같은 모의로 오류율을 재서 사전등록에 싣는다."""
    if len(set(candidates)) != len(candidates):
        raise ValueError("후보 이름 중복")
    if min_sets < 2:
        raise ValueError("min_sets ≥ 2(짝 SE 를 내려면)")
    offs = list(seed_offsets)
    max_sets = len(offs) if max_sets is None else min(max_sets, len(offs))
    if max_sets < min_sets:
        raise ValueError("max_sets ≥ min_sets 여야 한다")
    if run_round is None:
        if run is None:
            raise ValueError("run 또는 run_round 가 필요하다")

        def run_round(pairs):
            return [run(c, o) for c, o in pairs]

    res = RaceResult({c: {} for c in candidates}, list(candidates))
    res.params = {"candidates": list(candidates), "seed_offsets": offs[:max_sets], "k": k,
                  "min_sets": min_sets, "max_sets": max_sets}
    if ledger is not None:
        ledger.register(name, menu=None, seeds=offs[:max_sets], rationale="레이싱 인자(결과 전)",
                        race=res.params)
    for r in range(max_sets):
        off = offs[r]
        pairs = [(c, off) for c in res.alive]
        vals = run_round(pairs)
        for (c, o), v in zip(pairs, vals, strict=True):
            res.scores[c][o] = float(v)
        n = r + 1
        if n >= min_sets and len(res.alive) > 1:
            lead = res.leader
            out = []
            for c in res.alive:
                if c == lead:
                    continue
                pdf = paired_diff(res.scores[c], res.scores[lead])
                if pdf["mean"] + k * pdf["se"] < 0:
                    out.append(c)
                res.log.append({"sets": n, "cand": c, "leader": lead, **pdf, "out": c in out})
            for c in out:
                res.alive.remove(c)
                res.eliminated[c] = n
        if len(res.alive) == 1:
            break
    return res


# ---------------------------------------------------------------------------
# 봉인
# ---------------------------------------------------------------------------

class SealBreach(RuntimeError):
    """청산·평가 경로가 봉인 경계를 넘었다."""


class Seal:
    """봉인 경계 — ``days`` (거래일 순 날짜 배열) 에서 ``seal_date`` 의 첫 행(이 행부터 봉인).

    swing-it: ``E.LIM = min(LIM, seal)``·``end_row = seal − 1``·청산 행 ≥ seal 이면 중단. 학습창
    재평가는
    자르기 **전에** 끝낸다(호출자 몫)."""

    def __init__(self, days: Sequence, seal_date: Any):
        import pandas as pd  # noqa: PLC0415

        idx = pd.DatetimeIndex(days)
        if not idx.is_monotonic_increasing:
            raise ValueError("days 가 오름차순이 아니다")
        self.date = pd.Timestamp(seal_date)
        if len(idx) == 0 or not (idx[0] < self.date <= idx[-1]):
            raise ValueError(f"봉인 날짜 {self.date.date()} 가 자료 범위 (첫날, 끝날] 밖이다")
        self.row = int(idx.searchsorted(self.date))

    @property
    def end_row(self) -> int:
        """봉인 전 마지막 행."""
        return self.row - 1

    def clip(self, rows: np.ndarray) -> np.ndarray:
        """행 번호 배열(예: 상장 폐지·데이터 끝 행)을 경계로 자른다."""
        return np.minimum(np.asarray(rows), self.row).astype(np.int64)

    def check(self, exit_rows: Sequence[int]) -> None:
        a = np.asarray(exit_rows)
        if a.size and int(np.max(a)) >= self.row:
            raise SealBreach(f"청산 행 {int(np.max(a))} ≥ 봉인 경계 {self.row}({self.date.date()})")


# ---------------------------------------------------------------------------
# 결정성 · 출처
# ---------------------------------------------------------------------------

def _jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"digest: {type(o).__name__} 는 JSON 으로 못 쓴다")


def digest(obj: Any) -> str:
    """정렬된 JSON(sha256) — 같은 후보 두 실행의 결과가 바이트까지 같은지 대조할 때.

    JSON 표기 그대로 비교한다: −0.0 과 0.0, 1 과 1.0 은 다른 digest 다(바이트 동일 확인용이라
    의도)."""
    s = json.dumps(obj, sort_keys=True, default=_jsonable, ensure_ascii=False, allow_nan=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def provenance(repo: Path | str | None = None, modules: Sequence[Any] = ()) -> dict:
    """git HEAD·dirty·모듈 소스 해시·numpy/numba/cmaes 버전."""
    out: dict[str, Any] = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("numba", "cmaes", "pandas"):
        m = sys.modules.get(name)
        if m is None:
            try:
                m = __import__(name)
            except ImportError:
                m = None
        out[name] = getattr(m, "__version__", None) if m else None
    if repo is not None:
        try:
            out["git_head"] = git_head(repo)
            out["git_dirty"] = git_dirty(repo)
        except Exception as e:  # noqa: BLE001 — 출처 기록이 실행을 막지 않는다
            out["git_error"] = str(e)
    if modules:
        h = hashlib.sha256()
        for m in modules:
            h.update(inspect.getsource(m).encode("utf-8"))
        out["code_hash"] = h.hexdigest()[:16]
    return out
