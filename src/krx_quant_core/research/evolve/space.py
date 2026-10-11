"""탐색 공간 — 연속(Real)·범주형(Choice)·순서형(Ordinal) 유전자의 부호화·복호화·키.

swing-it ``pattern_ga_run.py`` 의 ``CONT``·``NOM``·``DISC`` + ``decode``·``encode``·``random_x``·
메뉴 덮어쓰기(``PATTERN_GA_MENU``)를 일반화했다.

**배치(legacy layout)** — 선언 순서와 무관하게 ``[Real…][Choice 로짓…][Ordinal 단계…]``. 각 묶음
안은 선언 순서. 최적화기(CMAwM)는 이 벡터를 본다: Real ∈ [0, 1], Choice 는 수준마다 로짓
∈ [0, 1](argmax, 동률 = 앞), Ordinal 은 수준 번호 ∈ [0, k−1](정수 축).

**유전자 행렬 G** — ``decode(X)`` 는 ``(n, n_genes)`` float64 를 낸다(열 = 켜진 유전자, 선언 순서).

- Real: 실제 값 ``round(lo + u·(hi−lo), digits)`` (파이썬 ``round`` — 정확한 십진 반올림,
  ``np.round`` 와 동점에서 다르다). 끔 끝(``off="lo"``/``"hi"``)에서 ``off_zone`` 안이면
  NaN(= 끔/``None``).
- Choice·Ordinal: **어휘 코드**. 어휘 = 수준 + ``extra`` (같은 값은 한 코드로 접는다 — 한 수준
  덮어쓰기 복제 ``(v, v)`` 가 같은 키가 되게), ``None`` 수준·값은 코드 −1.

``keys(G)`` 는 행마다 int64 키(−0 → 0, NaN 정규화). 같은 표현형 = 같은 키. 소비자 ``canonical``
은 G 를 제자리에서 고쳐 조건부로 평평한 유전자를 NaN/−1 로 만든다(같은 표현형 = 같은 키).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

__all__ = ["Choice", "Gene", "Ordinal", "Real", "Space"]

OFF_ZONE = 0.02


@dataclass(frozen=True)
class Real:
    """연속 유전자. ``off`` = 끔 끝("lo"/"hi"/None). ``optional`` = 덮어쓰기에 이름이 있을 때만
    켜짐."""

    name: str
    lo: float
    hi: float
    off: str | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if self.off not in (None, "lo", "hi"):
            raise ValueError(f"{self.name}: off 는 None/'lo'/'hi'")
        if not self.hi > self.lo:
            raise ValueError(f"{self.name}: hi > lo 여야 한다")


@dataclass(frozen=True)
class Choice:
    """범주형(명목) 유전자 — 최적화기에는 수준마다 연속 로짓 하나."""

    name: str
    levels: tuple
    optional: bool = False
    extra: tuple = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "extra", tuple(self.extra))
        if len(self.levels) < 1:
            raise ValueError(f"{self.name}: 수준이 없다")


@dataclass(frozen=True)
class Ordinal:
    """순서형(이산) 유전자 — 최적화기에는 정수 축 하나(수준 ≥ 2, 한 수준은 덮어쓰기에서 복제)."""

    name: str
    levels: tuple
    optional: bool = False
    extra: tuple = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "extra", tuple(self.extra))
        if len(self.levels) < 2:
            raise ValueError(f"{self.name}: 순서형 수준은 2개 이상(한 수준이면 (v, v) 로 복제)")


Gene = Real | Choice | Ordinal


def _same(a: Any, b: Any) -> bool:
    """수준 값 동치 — 파이썬 ``==`` (1 == 1.0, inf == inf). NaN 은 NaN 끼리 같다고 본다."""
    if a is None or b is None:
        return a is None and b is None
    try:
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return bool(a == b)
    except Exception:  # noqa: BLE001 — 비교 불가 타입은 다르다
        return False


def _vocab(levels: tuple, extra: tuple) -> tuple[tuple, np.ndarray]:
    """(어휘, 수준 번호 → 코드). ``None`` → −1, 같은 값은 첫 코드."""
    vocab: list = []
    codes = np.empty(len(levels), np.int64)
    for i, v in enumerate((*levels, *extra)):
        if v is None:
            c = -1
        else:
            c = next((j for j, w in enumerate(vocab) if _same(v, w)), -2)
            if c == -2:
                vocab.append(v)
                c = len(vocab) - 1
        if i < len(levels):
            codes[i] = c
    return tuple(vocab), codes


def _lv(v: Any) -> Any:
    return math.inf if v == "inf" else v


_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def _mix(h: np.ndarray) -> np.ndarray:
    h = h ^ (h >> np.uint64(30))
    h = h * _M1
    h = h ^ (h >> np.uint64(27))
    h = h * _M2
    return h ^ (h >> np.uint64(31))


class Space:
    """켜진 유전자 목록 + 부호화. 불변 — :meth:`override` 는 새 Space 를 낸다."""

    def __init__(self, genes: Sequence[Gene], *, off_zone: float = OFF_ZONE, digits: int = 4,
                 active: Iterable[str] | None = None):
        names = [g.name for g in genes]
        if len(set(names)) != len(names):
            raise ValueError("유전자 이름 중복")
        self.all_genes: tuple[Gene, ...] = tuple(genes)
        on = set(active) if active is not None else set()
        self.genes: tuple[Gene, ...] = tuple(g for g in genes if not g.optional or g.name in on)
        self.off_zone = float(off_zone)
        self.digits = int(digits)
        self.reals = tuple(g for g in self.genes if isinstance(g, Real))
        self.choices = tuple(g for g in self.genes if isinstance(g, Choice))
        self.ordinals = tuple(g for g in self.genes if isinstance(g, Ordinal))
        self.names: tuple[str, ...] = tuple(g.name for g in self.genes)
        self.col = {n: i for i, n in enumerate(self.names)}
        self.n_logit = sum(len(g.levels) for g in self.choices)
        self.n_cont = len(self.reals) + self.n_logit
        self.disc_sizes: tuple[int, ...] = tuple(len(g.levels) for g in self.ordinals)
        self.dim = self.n_cont + len(self.ordinals)
        self._vocab: dict[str, tuple] = {}
        self._codes: dict[str, np.ndarray] = {}
        for g in (*self.choices, *self.ordinals):
            self._vocab[g.name], self._codes[g.name] = _vocab(g.levels, g.extra)
        self._real_cols = np.array([self.col[g.name] for g in self.reals], np.int64)
        self._cat_cols = np.array([self.col[g.name] for g in (*self.choices, *self.ordinals)],
                                  np.int64)
        self._real_lo = np.array([g.lo for g in self.reals], float)
        self._real_span = np.array([g.hi - g.lo for g in self.reals], float)
        self._off_lo = np.array([g.off == "lo" for g in self.reals], bool)
        self._off_hi = np.array([g.off == "hi" for g in self.reals], bool)

    # --- 메타 ---
    def gene(self, name: str) -> Gene:
        return self.genes[self.col[name]]

    def vocab(self, name: str) -> tuple:
        """범주형·순서형 유전자의 어휘(코드 → 값). 코드 −1 = ``None``."""
        return self._vocab[name]

    def code(self, name: str, value: Any) -> int:
        """값 → 어휘 코드(``None`` → −1). 어휘에 없으면 ``KeyError`` — ``extra=`` 에 선언한다."""
        if value is None:
            return -1
        for j, w in enumerate(self._vocab[name]):
            if _same(value, w):
                return j
        raise KeyError(f"{name}: {value!r} 가 어휘에 없다(extra= 로 선언)")

    def level_values(self, name: str) -> np.ndarray:
        """코드 → 숫자 값 표(코드 −1 은 마지막 칸 NaN 으로 ``arr[code]`` 가 되게). 숫자가 아니면
        ``TypeError``."""
        vals = [float(v) for v in self._vocab[name]]
        return np.array([*vals, math.nan])

    @property
    def bounds(self) -> np.ndarray:
        """CMAwM 경계(배치 순서): 연속·로짓 [0, 1], 순서형 [0, k−1]."""
        return np.array([[0.0, 1.0]] * self.n_cont + [[0.0, float(k - 1)] for k in self.disc_sizes])

    @property
    def steps(self) -> np.ndarray:
        return np.array([0.0] * self.n_cont + [1.0] * len(self.disc_sizes))

    # --- 덮어쓰기 ---
    def override(self, menu: dict | None) -> Space:
        """swing-it ``PATTERN_GA_MENU`` 형식 ``{"nom": {이름: [수준…]}, "disc": {…}, "cont": {이름:
        [lo, hi]}}``.

        ``"inf"`` → ∞, ``None`` → 끔 수준. 순서형 한 수준은 ``(v, v)`` 로 복제. ``optional``
        유전자는 이름이
        있을 때만 켜진다(같은 종류 안 선언 순서). 모르는 이름·종류가 안 맞는 이름은 ``KeyError``."""
        menu = menu or {}
        no, do, co = menu.get("nom", {}), menu.get("disc", {}), menu.get("cont", {})
        by = {g.name: g for g in self.all_genes}
        for kind, d, cls in (("nom", no, Choice), ("disc", do, Ordinal), ("cont", co, Real)):
            bad = [n for n in d if n not in by or not isinstance(by[n], cls)]
            if bad:
                raise KeyError(f"덮어쓰기 {kind} 모르는 유전자: {sorted(bad)}")
        new: list[Gene] = []
        for g in self.all_genes:
            if isinstance(g, Choice) and g.name in no:
                g = replace(g, levels=tuple(_lv(x) for x in no[g.name]))
            elif isinstance(g, Ordinal) and g.name in do:
                t = tuple(_lv(x) for x in do[g.name])
                g = replace(g, levels=t * 2 if len(t) == 1 else t)
            elif isinstance(g, Real) and g.name in co:
                c = co[g.name]
                off = c[2] if len(c) >= 3 else g.off          # [lo, hi, off]: null = 끔 없음(필수)
                g = replace(g, lo=float(c[0]), hi=float(c[1]), off=off)
            new.append(g)
        on = {g.name for g in self.genes} | set(no) | set(do) | set(co)
        return Space(new, off_zone=self.off_zone, digits=self.digits, active=on)

    # --- 부호화 ---
    def decode(self, X: np.ndarray) -> np.ndarray:
        """``(n, dim)`` 배치 벡터 → ``(n, n_genes)`` 유전자 행렬(모듈 설명). 1차원이면 한 행."""
        X = np.asarray(X, float)
        one = X.ndim == 1
        if one:
            X = X[None, :]
        n = X.shape[0]
        G = np.empty((n, len(self.genes)))
        nr = len(self.reals)
        if nr:
            U = np.clip(X[:, :nr], 0.0, 1.0)
            off = ((self._off_lo & (U <= self.off_zone))
                   | (self._off_hi & (U >= 1.0 - self.off_zone)))
            V = self._real_lo + U * self._real_span
            d = self.digits
            R = np.array([round(v, d) for v in V.ravel().tolist()]).reshape(V.shape)
            R[off] = np.nan
            G[:, self._real_cols] = R
        j = nr
        for g in self.choices:
            k = len(g.levels)
            G[:, self.col[g.name]] = self._codes[g.name][np.argmax(X[:, j:j + k], axis=1)]
            j += k
        for q, g in enumerate(self.ordinals):
            k = len(g.levels)
            idx = np.clip(np.rint(X[:, self.n_cont + q]), 0, k - 1).astype(np.int64)
            G[:, self.col[g.name]] = self._codes[g.name][idx]
        return G[0] if one else G

    def keys(self, G: np.ndarray) -> np.ndarray:
        """행마다 int64 키 — −0 → 0, NaN 하나로. 같은 행 = 같은 키
        (64비트 섞기, 충돌 확률 ~n²/2⁶⁵)."""
        G = np.atleast_2d(np.asarray(G, float))
        A = G + 0.0                                      # −0.0 → +0.0
        A[np.isnan(A)] = np.nan                          # NaN 비트 하나로
        B = np.ascontiguousarray(A).view(np.uint64)
        h = np.full(G.shape[0], np.uint64(0x9E3779B97F4A7C15))
        for c in range(B.shape[1]):
            h = _mix(h ^ _mix(B[:, c] + np.uint64(c + 1)))
        return h.view(np.int64)

    def to_dict(self, row: np.ndarray) -> dict:
        """G 한 행 → {이름: 값} (끔·−1 = ``None``, Real 은 float — −0.0 보존)."""
        out: dict = {}
        for g in self.genes:
            v = float(row[self.col[g.name]])
            if isinstance(g, Real):
                out[g.name] = None if math.isnan(v) else v
            else:
                c = int(v)
                out[g.name] = None if c < 0 else self._vocab[g.name][c]
        return out

    def row(self, g: dict) -> np.ndarray:
        """{이름: 값} → G 한 행(값이 어휘에 없으면 ``KeyError``, 빠진 이름은 ``None``)."""
        r = np.empty(len(self.genes))
        for gene in self.genes:
            v = g.get(gene.name)
            if isinstance(gene, Real):
                r[self.col[gene.name]] = math.nan if v is None else float(v)
            else:
                r[self.col[gene.name]] = self.code(gene.name, v)
        return r

    def encode(self, g: dict) -> np.ndarray:
        """{이름: 값} → 배치 벡터. swing-it ``encode`` 와 같다: Real 끔 = 끔 끝(끔 없으면 0.5),
        Choice = 그 수준 원핫(없으면 ``ValueError``), Ordinal = 수준 번호(없으면 0)."""
        v: list[float] = []
        for gene in self.reals:
            val = g.get(gene.name)
            if val is None:
                v.append(0.0 if gene.off == "lo" else (1.0 if gene.off == "hi" else 0.5))
            else:
                v.append((val - gene.lo) / (gene.hi - gene.lo))
        for gene in self.choices:
            oh = [0.0] * len(gene.levels)
            oh[gene.levels.index(g[gene.name])] = 1.0
            v += oh
        for gene in self.ordinals:
            val = g.get(gene.name)
            v.append(float(gene.levels.index(val)) if val in gene.levels else 0.0)
        return np.array(v)

    def random(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """균등 추출 — swing-it ``random_x`` 와 같은 난수 순서(``rng.random((n, dim))`` 뒤 순서형
        열마다
        ``rng.integers(0, k, size=n)``)."""
        X = rng.random((n, self.dim))
        for q, k in enumerate(self.disc_sizes):
            X[:, self.n_cont + q] = rng.integers(0, k, size=n)
        return X
