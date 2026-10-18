"""README 히어로 이미지: KRX 상한가·하한가는 대칭이 아니다.

레포 루트에서:  uv run --with matplotlib --with numpy python scripts/make_hero.py
→ docs/images/hero-light.png, docs/images/hero-dark.png

데이터는 전부 라이브러리 함수로 계산한다(난수 없음, 재실행해도 같은 그림):
- market.limits.limit_up_price / limit_down_price  (기준가 → 상·하한가)
- market.ticks.tick_size_int / _BOUNDS              (호가단위 밴드 경계)
- costs.tax.TAX_SCHEDULE                            (시행일별 증권거래세)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from hero_style import caption, render_both  # noqa: E402

from krx_quant_core.costs.tax import TAX_SCHEDULE  # noqa: E402
from krx_quant_core.market.limits import limit_down_price, limit_up_price  # noqa: E402
from krx_quant_core.market.ticks import _BOUNDS, tick_size_int  # noqa: E402

MADE_ON = "2026-10-19"
LO, HI = 1_000, 1_000_000
N_BINS = 300
Y_LO, Y_HI = 29.55, 30.14


def _all_valid_ticks(lo: int, hi: int) -> np.ndarray:
    """``lo`` 이상 ``hi`` 이하의 모든 유효 호가(정수)."""
    out = []
    p = lo
    while p <= hi:
        out.append(p)
        p += tick_size_int(p)
    return np.asarray(out, dtype=np.float64)


def compute():
    base = _all_valid_ticks(LO, HI)
    up = np.array([limit_up_price(b) for b in base])
    dn = np.array([limit_down_price(b) for b in base])
    up_pct = (up / base - 1.0) * 100.0          # +29.7 ~ +30.0
    dn_pct = (1.0 - dn / base) * 100.0          # 하락폭 크기, 29.7 ~ 30.0

    # 로그 간격 구간마다 "가장 짧은 폭"(틱 절사 손실이 최대인 경우)을 대표값으로.
    edges = np.geomspace(LO, HI, N_BINS + 1)
    idx = np.clip(np.searchsorted(edges, base, side="right") - 1, 0, N_BINS - 1)
    x = np.sqrt(edges[:-1] * edges[1:])
    up_min = np.full(N_BINS, np.nan)
    dn_min = np.full(N_BINS, np.nan)
    for i in range(N_BINS):
        m = idx == i
        if m.any():
            up_min[i] = up_pct[m].min()
            dn_min[i] = dn_pct[m].min()
    ok = ~np.isnan(up_min)
    x, up_min, dn_min = x[ok], up_min[ok], dn_min[ok]

    # 어긋남이 가장 큰 기준가 하나(개별 호가 기준) → 직접 라벨용.
    gap = dn_pct - up_pct
    # 현실적인 가격대(5만~20만) 안에서 고른다.
    band = (base >= 50_000) & (base < 200_000)
    j = np.flatnonzero(band)[np.argmax(gap[band])]
    example = dict(base=base[j], up=up[j], dn=dn[j], up_pct=up_pct[j], dn_pct=dn_pct[j])
    return x, up_min, dn_min, example


def draw(fig, axes, p):
    ax, ax2 = axes
    x, up_min, dn_min, ex = compute()

    # 30% 기준선과 호가단위 경계
    ax.axhline(30.0, color=p["ink3"], lw=1, ls="--", zorder=1)
    for b in (LO, *_BOUNDS):
        if b != LO:
            ax.axvline(b, color=p["grid"], lw=1, zorder=0)
        ax.text(b * 1.04, Y_LO + 0.012, f"tick {tick_size_int(b):,}", color=p["ink3"], fontsize=8.5,
                ha="left", va="bottom")

    ax.fill_between(x, up_min, 30.0, color=p["blue_wash"], lw=0, zorder=2)
    ax.fill_between(x, dn_min, 30.0, color=p["amber_wash"], lw=0, zorder=2)
    ax.plot(x, up_min, color=p["blue"], zorder=4,
            label="Upper limit  (+30%, floored to the tick of the target price)")
    ax.plot(x, dn_min, color=p["amber"], zorder=3,
            label="Lower limit  (−30%, floored to the tick of the base price; magnitude)")

    ax.set_xscale("log")
    ax.set_xlim(LO, HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks([1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
                   1_000_000])
    ax.set_xticklabels(["1k", "2k", "5k", "10k", "20k", "50k", "100k", "200k", "500k", "1M"])
    ax.minorticks_off()
    ax.set_yticks([29.7, 29.8, 29.9, 30.0])
    ax.set_yticklabels(["29.7%", "29.8%", "29.9%", "30.0%"])
    ax.set_xlabel("Base price (KRW, log scale)")
    ax.set_ylabel("Worst-case limit move vs. base price")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.6, borderaxespad=0.3)

    # 직접 라벨: 어긋남이 가장 큰 예 하나
    txt = (f"base {ex['base']:,.0f}: upper {ex['up']:,.0f} (+{ex['up_pct']:.2f}%)\n"
           f"vs. lower {ex['dn']:,.0f} (−{ex['dn_pct']:.2f}%)")
    ax.annotate(txt, xy=(ex["base"], ex["up_pct"]), xytext=(HI * 0.97, Y_LO + 0.07),
                color=p["ink"], fontsize=10, ha="right", va="center",
                bbox=dict(boxstyle="square,pad=0.25", fc=p["surface"], ec="none"),
                arrowprops=dict(arrowstyle="-", color=p["ink2"], lw=1, shrinkA=2, shrinkB=5),
                zorder=6)
    ax.plot([ex["base"]], [ex["up_pct"]], "o", color=p["blue"], zorder=7)

    ax.set_title("KRX price limits are not symmetric — tick rounding bends ±30%", pad=22)

    # 두 번째 패널: 시행일별 증권거래세(매도)
    rows = [(t.effective, float(t.kospi.total) * 100, float(t.kosdaq.total) * 100)
            for t in TAX_SCHEDULE if t.effective >= date(2019, 1, 1)]
    xs = [r[0] for r in rows] + [date(2026, 12, 31)]
    ks = [r[1] for r in rows] + [rows[-1][1]]
    ax2.step(xs, ks, where="post", color=p["blue"], zorder=3)
    for d, v, _ in rows:
        ax2.plot([d], [v], "o", color=p["blue"], zorder=4)
    ax2.set_ylim(0.10, 0.30)
    ax2.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30])
    ax2.set_yticklabels(["0.10%", "0.15%", "0.20%", "0.25%", "0.30%"])
    ax2.set_xlim(date(2019, 1, 1), date(2026, 12, 31))
    ax2.set_xticks([date(y, 1, 1) for y in (2019, 2021, 2023, 2025)])
    ax2.set_xticklabels(["2019", "2021", "2023", "2025"])
    ax2.set_title("Sell-side transaction tax by effective date", fontsize=12, pad=22)
    ax2.text(0.0, 1.03, "KOSPI (securities + rural special); KOSDAQ totals are identical",
             transform=ax2.transAxes, color=p["ink2"], fontsize=9, ha="left", va="bottom")

    fig.subplots_adjust(wspace=0.28)
    caption(fig, "Source: krx_quant_core.market.limits.limit_up_price / limit_down_price over "
                 "every valid tick, 1,000\u20131,000,000 KRW\n(line = shortest move per log bin, "
                 "shade = range up to 30%); tax from krx_quant_core.costs.tax.TAX_SCHEDULE. "
                 f"Generated {MADE_ON}.", p)


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    paths = render_both(draw, out_dir="docs/images", name="hero", figsize=(12, 5), ncols=2,
                        gridspec_kw={"width_ratios": [2.4, 1]})
    for pth in paths:
        print(pth)
