<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/brand/wordmark-dark.svg">
  <img alt="krx-quant-core" src="docs/images/brand/wordmark-light.svg" height="52">
</picture>

# krx-quant-core

**한국 주식 호가단위·상·하한가·거래세를 거래일 기준으로 정확히 계산합니다.**

KOSPI·KOSDAQ 퀀트 시스템이 시장 규칙을 한 군데서 맞추는 공통 코어입니다. 상·하한가는 대칭이 아니고, 거래세는 시행일마다 다르며, 표본 외 구간은 한 번만 엽니다. 전략 코드는 포함하지 않습니다.

[![CI](https://github.com/younghwan91/krx-quant-core/actions/workflows/ci.yml/badge.svg)](https://github.com/younghwan91/krx-quant-core/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/krx-quant-core.svg)](https://pypi.org/project/krx-quant-core/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-younghwan--chae-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/younghwan-chae/)

[Quick start](#quick-start) · [상세 가이드](docs/guide.md) · [English](README.en.md) · [Issues](https://github.com/younghwan91/krx-quant-core/issues)

</div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/hero-dark.png">
  <img alt="KRX 상한가·하한가는 대칭이 아니다 — 기준가 1천~100만원 구간에서 호가단위 절사가 ±30% 를 비트는 폭과, 시행일별 증권거래세 계단" src="docs/images/hero-light.png" width="100%">
</picture>

<p align="center"><sub>기준가 구간별 상·하한가 비대칭 폭과 시행일별 증권거래세 계단</sub></p>

- **상한가와 하한가는 대칭이 아닙니다.** 기준가 154,200원의 상한가는 200,000원(+29.70%), 하한가는 108,000원(−29.96%)입니다. 규칙이 갈리는 `daily_bars` 130행 중 대칭식은 25행, 비대칭식은 105행을 맞혔습니다([실측 대조](docs/guide.md#가격제한폭--실측-대조)).
- **거래세는 날짜마다 다릅니다.** 2019년 0.25%에서 2025년 0.15%, 2026년 0.20%까지 여섯 단계로 바뀝니다. 비용 함수는 거래일을 받아 그날 세율을 적용합니다.
- **표본 외 구간은 한 번만 엽니다.** 실행 원장이 OOS 구간을 git에 잠그며, 최종 평가는 label당 한 번만 시작할 수 있습니다. 재시도도 거부합니다.

## Quick start

```bash
pip install krx-quant-core==0.9.0            # extras: [fast] numba · [opt] optuna · [db] psycopg · [evo] cmaes
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

Python 3.11 이상이 필요합니다. 필수 의존성은 `kiwoom-client`(호가단위 표 정본), `numpy`, `pandas`뿐입니다. `[fast]`가 없어도 같은 커널을 파이썬으로 돌려 같은 숫자를 냅니다.

```bash
# 릴리스 전 기능이 필요하면 main의 커밋에 고정해 설치합니다:
pip install "krx-quant-core @ git+https://github.com/younghwan91/krx-quant-core@<커밋 sha>"
```

## 왜 필요한가

한국 주식 백테스트·주문 코드에서 조용히 숫자를 틀리게 만드는 함정들입니다. 이 패키지는 각각을 함수 하나로 막습니다.

| 함정 | 손으로 짜면 | 이 패키지 |
|---|---|---|
| 거래세를 상수 하나로 둔다 | 매도 세율이 2024년 0.18%, 2025년 0.15%, 2026년 0.20%로 바뀌었습니다. 2025년 세율로 2026년 거래를 돌리면 매도마다 0.05%p를 덜 뺍니다 | `tax_rate(날짜, 시장)`이 시행일 표에서 그날 세율을 고릅니다 |
| 하한가를 상한가와 대칭으로 계산한다 | 기준가 239,000원의 하한가를 대칭식(기준가×0.7을 그 가격대 틱으로 올림)으로 구하면 167,300원이 나오지만 실제는 167,500원입니다. 규칙이 갈리는 실측 130행 중 대칭식은 25행만 맞았습니다 | `limit_down_price`는 기준가의 호가단위로 폭을 절사합니다(105행 일치) |
| 틱을 "가격 + n × 지금 호가단위"로 옮긴다 | 49,950원에서 2틱 위를 50,050원으로 잡으면 50,000원 이상 구간(호가단위 100원)에서 유효하지 않은 호가라 주문이 거부됩니다 | `shift_ticks(49_950, 2)`는 밴드 경계를 넘으며 50,100원을 돌려줍니다 |
| 호가단위 표를 레포마다 복사한다 | 경계 하나가 어긋나면 주문 거부나 상한가 오판이 생깁니다 | 표의 정본은 실제 주문을 내는 [kiwoom-client](https://github.com/younghwan91/kiwoom-client) 하나에 두고 여기서는 파생 연산만 합니다 |
| 종목코드를 숫자로 다룬다 | 알파벳이 들어간 신규 코드 `0155E0`을 숫자만 남겨 정규화하면 전혀 다른 회사 코드가 됩니다 | `normalize_code("0155E0")`는 `0155E0`을 그대로 유지하고 `normalize_code("5930")`은 `005930`을 돌려줍니다 |

**적용 범위와 출처를 함께 밝힙니다.**

- 호가단위 표는 2023-01-25 호가가격단위 개편(KOSPI·KOSDAQ·KONEX 통합) 이후 표입니다. 그 이전 KOSDAQ 저가 구간은 표가 달라 2023-01-25 이전 데이터에는 맞지 않습니다. 출처는 KRX 개편 안내와 증권사 공지입니다(`kiwoom_client.tick_size` docstring).
- 가격제한폭은 2015-06-15 이후 ±30% 규칙입니다. 권리락·신규상장처럼 기준가가 전일종가가 아닌 날은 호출부가 기준가를 넘겨야 합니다.
- 2026-01-01 세율 인상은 기획재정부 증권거래세법 시행령 개정 보도로 확인했습니다. 그 이전 단계는 2차 출처이고 법령 원문(law.go.kr)과는 아직 대조하지 않았습니다. KONEX 세율은 확인하지 못해 KOSDAQ과 같게 두었습니다([거래세 표](docs/guide.md#거래세-스케줄-매도-시)).

## 무엇이 들어 있나

| 모듈 | 하는 일 |
|---|---|
| `market/` | 종목코드 정규화·`Market`, 호가단위(스칼라·벡터·ETF), 상·하한가, KST 세션, 거래일 달력 |
| `costs/` | 시행일별 거래세 스케줄, `KoreanCostModel`(Decimal), `round_trip_cost`(float), `impact`(제곱근 충격 비용) |
| `execution/` | `OrderManager`·`InstanceLock`·`PositionBook`·`Broker`(`PaperBroker`·`KiwoomBroker`)·`EngineCore`·`OrderGuard` |
| `risk/` | DART 중대공시 분류·`RiskGate`, `DartDisclosureDB`, `KillSwitch`, `factor_model`(`RiskModel`), `attribution` |
| `backtest/` | 호가 스윕 VWAP·왕복비용, 지정가 체결 규칙, 트레이드 원장 지표, 횡단면 시뮬, `replay`, `drift`, `ticktime` |
| `backtest/lob/` | 틱·호가 → 초 격자 특징·경로(배치=실시간 공용 커널), 에피소드 시뮬, 무작위 대조군, 지정가 대기열 모델, `ceiling` |
| `data/` | 공용 DB `COPY` → 열 단위 `.npy` + `mmap`, 일봉·분봉·틱·호가 캐시, 정규장 기준가 복원, `risk_inputs.load_risk_inputs` |
| `stats/` | Deflated/Probabilistic Sharpe, t-haircut, purged walk-forward, 부트스트랩, 취약성, 선택편향, 층 매칭 대조군, `ic` |
| `portfolio/` | `optimize` — 팩터 위험모델 위 제약 평균-분산 최적화 (`[portfolio]` extra 필요) |
| `research/` | `run_sweep`(격자·병렬·캐시), `optuna_search`, `evolve`(BIPOP CMAwM·세대 프로토콜·짝 레이싱·봉인) |
| `runtime/` | 호스트 가드, 실행 기록 `start_run`, OOS 하드 잠금, `kqc` CLI(원격 실행·`nightly`·`pins`) |

각 전략 레포는 `krx-quant-core==X.Y.Z`로 버전을 고정해 사용합니다.

| 소비자 | 매매 스타일 | 실행 | 이 패키지에서 쓰는 기능 |
|---|---|---|---|
| [scalp-it](https://github.com/younghwan91/scalp-it) | 스캘핑 | 자동 실주문 | 호가·상한가·주문 가드·호가 스윕·체결 규칙·호스트 가드 |
| [daytrade-it](https://github.com/younghwan91/daytrade-it) | 데이트레이딩 | 자동 실주문 | 비용모델·공시 리스크 게이트·DART DB·코드 정규화·호스트 가드 |
| [swing-it](https://github.com/younghwan91/swing-it) | 스윙 | 리서치 + 수동 체결 | DSR·purged CV·부트스트랩·취약성 진단·횡단면 시뮬·진화 엔진·최적화 |

## 왜 따로 뺐나

세 레포가 같은 규칙을 각자 들고 있었고, 실제로 어긋나 있었습니다.

- **종목코드 정규화 불일치**: `normalize_code`가 두 레포에 복붙되어 있었고, 한쪽 옛 구현은 `0155E0`을 `001550`(전혀 다른 회사)으로 바꿨습니다.
- **흩어진 거래비용**: 거래비용이 레포마다 `0.0023`, `0.0034`, `0.0064`, `50bp`로 달랐고, 2023~2026년 세율 변경을 일자별로 적용하는 코드가 없었습니다.
- **반대 체결 판정**: 같은 레포 안에서도 알 수 없는 값을 받으면 한 곳은 `through`, 다른 곳은 `touch`로 반대로 읽었습니다.
- **누락된 하한가**: 하한가 계산이 없었고, 대칭식으로 짜면 실제 가격과 어긋납니다.

핵심 원칙은 네 가지입니다. 포트는 원본과 **수치 동일성**을 대조했고, 통계 함수는 **수치만 반환**(판정 금지)하며, 미검증 브로커 필드는 **`False` 플래그로 명시**하고, 공유 앱키의 **토큰 폐기를 금지**합니다.

## 같이 쓰면 좋은 것

| 레포 | 하는 일 |
|---|---|
| [kiwoom-client](https://github.com/younghwan91/kiwoom-client) | 키움 REST API 파이썬 클라이언트입니다. 엔드포인트 182개·실시간 WebSocket·MCP 서버를 지원하고, 이 패키지가 쓰는 호가단위 표의 정본이 있습니다 |
| [krx-news-client](https://github.com/younghwan91/krx-news-client) | DART 공시와 토스증권 뉴스를 한 스키마로 받는 한국 주식 뉴스·공시 클라이언트입니다 |
| [kiwoom-order-guard](https://github.com/younghwan91/kiwoom-order-guard) | AI 에이전트가 키움 MCP·플러그인으로 내는 주문을 전송 전에 검사하는 Claude Code 훅입니다 |
| [fin-checkup](https://github.com/younghwan91/fin-checkup) | 상장사 재무제표 지표 17개를 계산식·경계값과 함께 신호등으로 보여 줍니다. 측정값만 다루며 투자권유가 아닙니다 |

## 문서

| 문서 | 내용 |
|---|---|
| [상세 가이드](docs/guide.md) | 가격제한폭 실측, 거래세 표, 실행 원장·OOS 잠금, 실매매 안전 규칙, LOB 체결 가정, 우위 판정, 데이터 층, 스윕·진화, 개발·로드맵 |
| [설계 문서](docs/superpowers/specs/) | 모듈별 설계 배경과 의사결정 기록 |
| [구현 계획](docs/superpowers/plans/) | 버전별 구현 계획 기록 |
| [릴리스 프로세스 다이어그램](docs/diagrams/release-process.html) | 배포·핀 동기화 흐름 그래프 |

## 기여 및 지원

- 버그 제보, 질문, 기능 제안은 [GitHub Issues](https://github.com/younghwan91/krx-quant-core/issues)에 남겨 주세요.
- 개발 환경·테스트·커밋 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md), 처음이라면 [good first issue](https://github.com/younghwan91/krx-quant-core/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)부터 보세요. 명령 상세는 [가이드의 개발 절](docs/guide.md#개발)에 있습니다.
- **만든 사람**: **채영환 (Younghwan Chae)** · [GitHub @younghwan91](https://github.com/younghwan91) · [LinkedIn](https://www.linkedin.com/in/younghwan-chae/)

## 라이선스

[Apache-2.0](LICENSE)
