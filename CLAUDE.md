# krx-quant-core

KRX 퀀트 공용 코어(시장 규칙·비용·주문 가드·검증 통계·백테스트 엔진·데이터 층·진화 엔진). **전략은 여기 두지 않는다.**
소비 레포(scalp-it·daytrade-it·swing-it, Airflow 이미지)가 `krx-quant-core==X.Y.Z` 로 **정확히 핀**해 PyPI 에서 받는다 —
여기 main 에 머지해도 소비 레포는 릴리스·핀 올림 전까지 안 바뀐다. 모듈 설명·안전 규칙·로드맵은 `README.md`.

## 명령 (uv)

```bash
uv sync --extra dev                      # + --extra fast(numba) --extra opt(optuna) --extra db(psycopg) --extra evo(cmaes)
uv run pytest -q                         # ~950 tests, 1분 안팎
uv run ruff check src tests              # CI 와 같은 ruff==0.16.2
KRX_QUANT_CORE_DISABLE_NUMBA=1 uv run --extra dev --extra fast pytest -q   # numba 있어도 파이썬 폴백 경로로
uv run kqc pins                          # ~/git/<소비레포> 의 핀·lock 표. 뒤처진 레포가 있으면 exit 1
```

CI(`.github/workflows/ci.yml`)는 3.11~3.13 × `[dev,fast]` + numba 없는 `[dev]` 잡을 따로 돈다 — numba 경로와 폴백이 **같은 골든**을
통과해야 한다. `git push` 때 전역 pre-push 훅이 ci.yml 을 로컬에서 다시 돌려 몇 분 걸린다(멈춘 게 아니다).

## 릴리스 (= 소비 레포에 영향이 가는 유일한 길)

1. 기능 PR 머지 뒤 버전 올림 커밋: `pyproject.toml` `version`, `src/krx_quant_core/__init__.py` `__version__`,
   `README.md` 설치 줄(`==X.Y.Z`·`@vX.Y.Z`), `uv lock` — 네 곳을 같이 바꾼다(예: 830f1c2, fe3860f).
2. main 에서 `git tag -a vX.Y.Z -m "..."` → `git push origin vX.Y.Z` → `publish.yml` 이 PyPI 에 올린다. **되돌릴 수 없다** —
   사용자가 배포를 지시했을 때만.
3. 소비 레포마다 `krx-quant-core==X.Y.Z` + `uv lock` PR, 그 레포 테스트 통과 뒤 머지, 끝에 `uv run kqc pins` 가 green.
   방금 올린 직후엔 PyPI simple index 지연으로 소비 레포 CI 가 실패할 수 있다 — 재실행하면 된다(v0.7.0 때 swing-it).
- `kqc pins` 는 `~/git/<repo>` **기본 체크아웃만** 본다. 워크트리(`swing-it-*` 등)와 `quant-airflow/docker/requirements.txt`
  (Airflow 이미지, 재빌드는 사용자 승인 필요)의 핀은 따로 확인한다.

## 바꾸기 전에 지킬 것

- **수치 동일성이 계약이다.** 소비 레포는 실매매·연구 숫자를 코어로 갈아타며 바이트 동일을 증명했다. 기존 함수의 출력이
  바뀌는 변경은 새 옵션·새 함수로 내고 기본값은 그대로 둔다. 이식(port)은 원본과 무작위·실데이터 대조 테스트를 같이 넣는다.
- 통계 함수는 숫자만 낸다 — `pass`/`fail`/`verdict` 같은 판정 키를 돌려주면 `tests/stats/test_stats_sharpe.py` 가 막는다.
  합격선은 소비 레포의 사전등록 문서 몫이다.
- 호가단위 표의 정본은 `kiwoom-client` 다 — 여기서 다시 정의하지 않는다.
- `execution`(OMS·KiwoomBroker)은 실주문 경로다. 주문 제출 재시도 금지·청산은 킬이 막지 않음·`fills_verified=False` 기본 등
  원칙은 README "안전 규칙"을 먼저 읽고 바꾼다.
- numba 커널은 `_jit.njit`(=`cache=True`)이다. numba 디스크 캐시는 **다른 파일의 피호출 커널이 바뀌어도 무효화되지 않는다** —
  커널을 고친 뒤 결과가 이상하면 `__pycache__` 의 `*.nbi`/`*.nbc` 를 지우고 다시 돌린다.

## 호스트

이 PC(`simnode`)가 백테스트 전용 호스트이자 지금은 라이브·페이퍼 데몬과 크론도 돈다(trader 는 2026-10 초 처분).
장중(09:00~15:30 KST)엔 무거운 벤치·스윕·풀 병렬을 피한다. 소비 레포 크론은 `~/git/<repo>` 작업 트리에서 돌므로
그 체크아웃을 다른 브랜치에 두고 떠나지 않는다 — 작업은 워크트리에서.

## 문서

- `docs/superpowers/specs/` — 버전별 설계(왜 이렇게 만들었나). 해당 모듈을 크게 바꿀 때 먼저 읽는다.
- `docs/superpowers/plans/` — 구현 계획 기록.
