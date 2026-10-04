# 판정 축 v0.6.0 Implementation Plan

**Spec:** `docs/superpowers/specs/2026-10-04-judgment-axis-design.md`

## Global Constraints
- ruff 0.16.2, line-length 100, `E,F,I,W,UP,B`. 기존 공개 API·수치 불변. 새 런타임 의존성 없음(scipy 금지).
- docstring 한국어, "왜"를 적는다. 통계 함수는 판정하지 않는다.
- 테스트는 simnode(`kqc run`). 커밋 끝 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Tasks
- [ ] 1. `runtime/nightly.py` PATH 해석 + `kqc nightly` FAILED 줄 + `kqc nightly status` (`tests/runtime/test_nightly.py`, `test_cli.py`)
- [ ] 2. `runtime/pins.py` + `kqc pins` (`tests/runtime/test_pins.py`)
- [ ] 3. `stats/selection.py` (`tests/stats/test_selection.py`, 골든)
- [ ] 4. `backtest/lob/ceiling.py` (`tests/backtest/lob/test_lob_ceiling_golden.py`, numba 폴백)
- [ ] 5. `backtest/drift.py` (`tests/backtest/test_drift.py`)
- [ ] 6. `stats/matched_null.py` (`tests/stats/test_matched_null.py`)
- [ ] 7. `__init__` 내보내기, README, 버전 0.6.0, simnode 전체 테스트, PR → main → 태그
- [ ] 8. 소비 레포 핀 PR(daytrade-it·scalp-it·swing-it), 다음날 `kqc nightly status` 로 rc 확인
