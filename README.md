# krx-quant-core

**한국 주식(KOSPI·KOSDAQ) 퀀트 공통 코어.** 호가단위·가격제한폭·세션 규칙, 일자별
증권거래세 비용모델, 키움 주문 가드, DART 중대공시 분류, 킬스위치, 체결 시뮬레이션,
Deflated Sharpe·purged CV 검증 통계를 한 패키지로 묶었다.

전략은 여기 없다. 전략을 가진 레포들이 **같은 시장 규칙과 같은 비용 숫자**를 쓰게 하는
것이 이 패키지의 존재 이유다.

| 소비자 | 매매 스타일 | 실행 | 이 패키지에서 쓰는 것 |
|---|---|---|---|
| [scalp-it](https://github.com/younghwan91) (비공개) | 스캘핑 | 자동 실주문 | 호가·상한가·주문 가드·호가 스윕·체결 규칙·호스트 가드 |
| [daytrade-it](https://github.com/younghwan91) (비공개) | 데이트레이딩 | 자동 실주문 | 비용모델·공시 리스크 게이트·DART DB·코드 정규화·호스트 가드 |
| [swing-it](https://github.com/younghwan91/swing-it) | 스윙 | 리서치 + 수동 체결 | DSR·purged CV·부트스트랩·취약성 진단·횡단면 시뮬 |

## 왜 따로 뺐나

세 레포가 같은 규칙을 각자 들고 있었고, 실제로 어긋나 있었다.

- `normalize_code` 가 두 레포에 **복붙**돼 있었다. 한쪽 옛 구현은 `0155E0` 을 `001550`
  (완전히 다른 회사)으로 바꾸던 버그를 안고 있었다.
- 거래비용이 레포마다 `0.0023`·`0.0034`·`0.0064`·`50bp` 로 흩어져 있었고, 거래세가
  2023→2024→2025→2026 에 네 번 바뀌었는데 일자별로 적용하는 코드가 **어디에도 없었다**.
- 체결 판정(`through`/`touch`)이 같은 레포 안에서도 알 수 없는 값을 받으면 한 곳은
  through, 다른 곳은 touch 로 **반대로** 읽었다.
- 하한가 계산은 없었다. 대칭식으로 짜면 틀린다 — 실측 대조 결과는 아래.

## 설치

```bash
pip install "krx-quant-core @ git+https://github.com/younghwan91/krx-quant-core@v0.2.0"
# 초 격자 호가 리플레이(backtest.lob)를 numba 로 가속하려면 extra 로: "krx-quant-core[fast] @ git+...@v0.2.0"
# PyPI 릴리스 전까지는 git 태그로 고정한다.
```

Python ≥ 3.11. 의존성은 `kiwoom-client`(호가단위 표의 정본), `numpy`, `pandas` 뿐이다.
선택 extra `fast` 는 `numba` 를 더한다 — 없으면 같은 커널을 파이썬으로 돌려 같은 숫자를 낸다.

## 모듈

```
krx_quant_core/
├── market/     종목코드·Market, 호가단위, 상/하한가, KST 세션, 거래일 달력
├── costs/      일자별 거래세 스케줄, KoreanCostModel(Decimal), round_trip_cost(float)
├── execution/  키움 REST 주문 스펙, OrderIntent/OrderResult, OrderGuard(순수 가드)
├── risk/       DART 중대공시 분류·RiskGate, DartDisclosureDB, KillSwitch
├── backtest/   호가 스윕 VWAP·왕복비용, 지정가 체결 규칙, 트레이드 원장 지표, 횡단면 시뮬
│   └── lob/    틱·호가 → 초 격자 특징·경로, 배치=실시간 공용 커널, 에피소드 시뮬, 무작위 대조군
├── stats/      Deflated/Probabilistic Sharpe, t-haircut, purged walk-forward, 부트스트랩, 취약성
└── runtime/    호스트 가드(백테스트는 simnode 에서만)
```

```python
from datetime import date
from krx_quant_core.market import Market, limit_up_price, limit_down_price, shift_ticks
from krx_quant_core.costs import round_trip_cost, tax_rate

limit_up_price(24_250)          # 31500
limit_down_price(239_000)       # 167500.0
shift_ticks(49_950, 2)          # Decimal('50100') — 밴드 경계를 넘어도 유효 호가
tax_rate(date(2025, 6, 2), Market.KOSPI).total   # Decimal('0.0015')
round_trip_cost(date(2026, 9, 14), Market.KOSDAQ, slippage_one_way=0.0015)  # 0.0053
```

### 초 격자 호가 리플레이 (`backtest.lob`, v0.2)

scalp-it 80·81·82번에 흩어져 있던 틱·호가 리플레이를 옮겼다. 원본 `build_code`·
`_strength_exit_labels`·`random_control` 사본과 합성 틱·호가로 대조해 **비트 단위 동일**
(골든 테스트), numba 경로와 파이썬 폴백도 동일, 배치와 실시간 증분도 동일하다.

```python
from datetime import date
from krx_quant_core.backtest.lob import build_second_grid, simulate_exits, random_entry_control

feat, path = build_second_grid(ticks, quotes)     # 09:00~15:20 초 격자, 09:05~15:10 특징 행
r = simulate_exits(feat.sec, path["bid"], path["ask"], path["strength"],
                   strength_drop=3.0, max_hold=300,          # 82번 청산 규칙
                   trade_date=date(2026, 9, 8), market="KOSDAQ")  # 비용 = round_trip_cost
r.net, r.hold, r.reason                          # 순수익·보유초·청산사유(EXIT_*)
```

실시간은 `SecondFeatureStream().update(...)` 를 초마다 부른다 — 배치와 같은 `step` 커널이다.
체결 가정은 원본 그대로 낙관적이다(1호가 전량 체결, 잔량·대기열 무시).

## 설계 원칙

1. **이식은 수치 동일.** 소비 레포가 실매매일에 갈아탈 수 있어야 한다. 포트마다 원본
   코드와 무작위 입력으로 대조했다 — 주문 가드 16,000건 사유 문자열 동일, 킬스위치
   9,000 스텝 동일, 실제 `dart.db` 공시 26,551건 분류 동일, 통계·횡단면 시뮬 40 시드
   NaN 포함 동일.
2. **판정하지 않는다.** 통계 함수는 숫자를 리포트할 뿐 `PASS`/`FAIL` 을 돌려주지 않는다
   (테스트가 강제). 합격선은 사전등록 문서에 있어야 한다.
3. **모르는 건 모른다고 적는다.** 시장가 `trde_tp` 는 `"03"`/`"3"` 둘 중 무엇인지 실호출로
   확인된 적이 없어 `TRDE_TP_MARKET_VERIFIED = False` 로 둔다. 2025 이전 거래세 단계는
   2차 출처만 있다고 docstring 에 적었다.
4. **토큰은 공유 자원이다.** 두 실매매 프로세스가 같은 앱키를 쓴다. 연결 해제 시 토큰을
   폐기하면 다른 프로세스가 죽는다(`kiwoom_spec.TOKEN_SHARING_WARNING`).

## 가격제한폭 — 실측 대조

`daily_bars`(2023-02 이후)에서 고가·저가가 기준가 ±29~31% 인 행에 후보 규칙을 대조했다.

| 규칙 | 규칙끼리 갈리는 행에서 적중 |
|---|---|
| 상한가 = ⌊기준가×1.3 을 **그 가격대** 틱⌋ | **1,211** |
| 상한가 = 기준가 + ⌊기준가×0.3 을 기준가 틱⌋ | 165 |
| 하한가 = ⌈기준가×0.7 을 그 가격대 틱⌉ (대칭식) | 25 |
| 하한가 = 기준가 − ⌊기준가×0.3 을 **기준가** 틱⌋ | **105** |

즉 상한가와 하한가는 **대칭이 아니다**.

## 거래세 스케줄 (매도 시)

| 시행일 | KOSPI 거래세 + 농특세 | KOSDAQ |
|---|---|---|
| 2026-01-01 | 0.05% + 0.15% = 0.20% | 0.20% |
| 2025-01-01 | 0.00% + 0.15% = 0.15% | 0.15% |
| 2024-01-01 | 0.03% + 0.15% = 0.18% | 0.18% |
| 2023-01-01 | 0.05% + 0.15% = 0.20% | 0.20% |
| 2021-01-01 | 0.08% + 0.15% = 0.23% | 0.23% |
| 2019-06-03 | 0.10% + 0.15% = 0.25% | 0.25% |

2026 개정은 시행령 개정 보도로 확인했고, 그 이전 단계는 2차 출처다. 법령 원문으로 확인되면
이 표를 갱신한다.

## 개발

```bash
uv sync --extra dev
uv sync --extra dev --extra fast   # numba 경로까지
uv run pytest -q        # 454 tests
uv run ruff check src tests
```

태그 `v*` 를 푸시하면 `publish.yml` 이 PyPI 로 올린다(trusted publishing 설정 후).

## 로드맵

- **v0.3** 키움 주문 HTTP 클라이언트 통합 — 지금은 scalp-it(원시 httpx)과 daytrade-it
  (kiwoom-client 기반 `KiwoomBroker`)이 서로 다른 주문 스택을 쓴다. 가드·스펙은 공유했고
  전송 계층은 실주문 대조 뒤에 합친다.
- scalp-it `RiskGuard` 의 킬 판정을 `KillSwitch` 로 위임(대조 테스트는 이미 있음).
- 이벤트 기반 일중 백테스트 엔진 — v0.2 에 초 격자 리플레이(`backtest.lob`)가 들어갔다. 남은 것:
  지정가 대기열 위치·부분 체결 모델, ETF 호가단위(kiwoom-client 정본 추가 대기).
- PyPI 릴리스.

## 라이선스

Apache-2.0
