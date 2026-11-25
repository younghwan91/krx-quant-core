# 기여 가이드

krx-quant-core 에 관심 가져 주셔서 감사합니다. 이 패키지는 여러 실매매·연구 레포가 `krx-quant-core==X.Y.Z` 로 정확히 핀해 쓰는
공통 코어라, 숫자가 바뀌는 변경에 특히 엄격합니다. 버그 제보, 문서 수정, 테스트 추가는 환영합니다.
처음이라면 [good first issue](https://github.com/younghwan91/krx-quant-core/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) 목록에서 시작하세요.

> English: PRs and issues in English are welcome. Short version: `uv run pytest -q` and `uv run ruff check src tests` must pass
> (also with `KRX_QUANT_CORE_DISABLE_NUMBA=1`); existing outputs must not change (add options, don't alter defaults); use English Conventional Commits.

## 개발 환경

Python 3.11 이상과 [uv](https://docs.astral.sh/uv/) 를 권장합니다.

```bash
git clone https://github.com/younghwan91/krx-quant-core.git
cd krx-quant-core
uv sync --extra dev          # 필요하면 --extra fast(numba) --extra opt(optuna) --extra db(psycopg) --extra evo(cmaes)
```

## 테스트와 린트

```bash
uv run pytest -q                         # 1분 안팎
uv run ruff check src tests              # CI 와 같은 ruff==0.16.2
KRX_QUANT_CORE_DISABLE_NUMBA=1 uv run --extra dev --extra fast pytest -q   # numba 가 있어도 파이썬 폴백 경로로
```

CI 는 3.11~3.13 에서 `[dev,fast]` 와 numba 없는 `[dev]` 를 따로 돌립니다. 두 경로가 같은 골든 값을 통과해야 합니다.

## 바꾸기 전에 지킬 것

- **수치 동일성이 계약입니다.** 기존 함수의 출력이 바뀌는 변경은 새 옵션·새 함수로 내고 기본값은 그대로 둡니다.
- 통계 함수는 숫자만 돌려줍니다. `pass`/`fail`/`verdict` 같은 판정 키는 테스트가 막습니다.
- 시장 규칙(호가단위·상하한가·세율)을 고칠 때는 근거(규정·실측 날짜)를 PR 에 적어 주세요.
- 전략 코드는 이 레포에 두지 않습니다.

## 커밋과 PR

1. `main` 에서 브랜치를 만듭니다(`feat/…`, `fix/…`, `docs/…`).
2. 커밋 메시지는 **영어 [Conventional Commits](https://www.conventionalcommits.org/)** 입니다(`fix(costs): use 2026 sell tax`). 본문에는 왜 바꿨는지 적어 주세요.
3. 새 동작에는 테스트를 붙이고 위 명령이 통과하는지 확인한 뒤 PR 을 엽니다. 작고 한 가지 일만 하는 PR 이 빨리 머지됩니다.
4. CI 가 통과해야 머지됩니다. 버전 올림·태그·PyPI 배포는 메인테이너가 합니다(PR 에서 `pyproject.toml` 버전을 바꾸지 마세요).

## 이슈

- 버그·기능 제안: [Issues](https://github.com/younghwan91/krx-quant-core/issues)
- good first issue: [목록 보기](https://github.com/younghwan91/krx-quant-core/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
