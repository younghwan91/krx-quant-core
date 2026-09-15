# simnode 백테스트 실행 기반 — 설계 (krx-quant-core v0.4.0)

2026-09-15. 사용자 결정: 기록은 레포 JSONL(정본) + simnode Postgres(색인), OOS 는 하드 잠금,
실행은 trader 에서 `kqc run` 으로 원격 발사.

## 목표

세 매매 레포(scalp-it·daytrade-it·swing-it)의 백테스트가 **어디서·어떤 코드로·어떤 데이터로·
몇 번 시도해서** 나온 숫자인지 기계가 보증한다. 사람이 적는 규율을 코드 게이트로 바꾼다.

비목표: 작업 큐·동시 실행 제한·대시보드(YAGNI). 체결·검증 엔진(다른 하위 프로젝트).

## 1. `krx_quant_core.runtime.runs` — 실행 기록

```python
with start_run("scalp84-flow", config, repo_root=ROOT, data=DataSpec("2026-08-24", "2026-09-07"),
               seed=84, prereg=None) as run:
    ...
    run.log_result({"mean_bp": -3.1, "n": 812})
```

- 진입 시 순서대로 검사하고 하나라도 실패하면 `RunRefused`(RuntimeError):
  1. `require_backtest_host()` — simnode 아님.
  2. `repo_root` git 작업트리가 깨끗해야 한다(`git status --porcelain` 비어 있음, 추적 안 되는
     파일은 무시). HEAD sha 기록. `allow_dirty=True` 는 탐색용 — 기록에 `dirty: true` 가 남는다.
  3. `data` 가 그 label 의 OOS 잠금 구간과 겹치면 `oos.check` 규칙을 따른다(§2).
- 진입 시 `research/runs/<label>/RUNS.jsonl` 에 `start` 행 append:
  `run_id(uuid4 hex 12)`, `ts`(KST ISO), `host`, `repo`, `git_sha`, `dirty`, `config`, `fingerprint`
  (`stats.trials.config_fingerprint`), `data`(start·end·split), `seed`, `versions`(python·
  krx-quant-core·numpy·pandas), `argv`.
- 같은 진입에서 `stats.trials.record_trial(label, config, logs_dir=research/runs)` — 원장 파일은
  `research/runs/<label>/TRIALS.jsonl`. DSR 의 N 은 `run.n_trials` 로 읽는다.
- 종료 시 `end` 행 append: `run_id`, `status`(`ok`/`failed`), `duration_s`, `result`(log_result 로
  넘긴 dict, JSON 직렬화 안 되는 값은 `str`), 예외면 `error`(타입·메시지 한 줄). 예외는 다시 던진다.
- 색인: 행을 쓸 때마다 `KR_QUANT_DB` 가 있으면 `kqc_runs` 테이블에 upsert(run_id PK, label, repo,
  git_sha, status, started_at, ended_at, fingerprint, result jsonb, row jsonb). DB 실패는 경고 로그만
  — JSONL 쓰기가 성공하면 실행은 계속된다. `kqc runs sync <repo_root>` 가 JSONL 에서 테이블을
  다시 채운다. psycopg 는 선택 의존(`[db]` extra); 없으면 색인을 건너뛴다.

## 2. `krx_quant_core.runtime.oos` — OOS 하드 잠금

파일: `research/runs/<label>/OOS.json`(구간 정의), `PREREG.lock`, `FINAL.lock`. 모두 git 추적.

- `kqc oos define <label> --start D --end D` → `OOS.json`. 이미 있으면 거부(구간을 옮기는 건
  새 label).
- `kqc prereg lock <label> <doc.md>` → `PREREG.lock` = `{doc, sha256, locked_at, git_sha}`. 이미
  있으면 거부.
- `oos.check(label, start, end, *, repo_root, final=False)`:
  - `OOS.json` 없음 → 통과(잠금 안 한 label).
  - 구간이 OOS 와 안 겹침 → 통과.
  - 겹침 + `final=False` → `OOSLocked`.
  - 겹침 + `final=True`: `PREREG.lock` 없음 → 거부; 문서 현재 sha256 ≠ 잠금 값 → 거부(사전등록
    사후 수정); `FINAL.lock` 이미 있음 → 거부(한 번만). 통과하면 `FINAL.lock` 을 **즉시** 만든다
    (`run_id`, `ts`, `git_sha`) — 평가가 중간에 죽어도 재시도 불가. 되돌리려면 사람이 git 에서 파일을
    지우고 그 커밋이 기록으로 남는다.
- `start_run(..., final=True)` 가 `data` 로 `oos.check` 를 부른다. 데이터 로더도 직접 부를 수 있다.

## 3. `kqc` CLI (콘솔 스크립트)

- `kqc run <repo> [--ref REF] -- <cmd...>` (trader 에서):
  1. 로컬 `~/git/<repo>` 가 깨끗하고 HEAD(또는 REF) 가 origin 에 푸시돼 있어야 한다.
  2. `ssh simnode-local` 로: `~/git/<repo>` 에서 `git fetch origin`, `git worktree add --detach
     ~/.kqc/wt/<repo>-<sha12> <sha>`(이미 있으면 재사용), 그 안에서 `uv sync --frozen --all-extras`
     실패 시 `uv sync --frozen`, `../quant-airflow/.env` 에서 `KR_QUANT_DB`·`DART_API_KEY` 만 export,
     `<cmd>` 실행. worktree 의 `research/runs` 기록은 실행 뒤 `~/git/<repo>` 로 옮겨지지 않는다 —
     대신 `KQC_RUNS_ROOT=~/git/<repo>` 를 넘겨 `start_run` 이 원 체크아웃의 `research/runs` 에 쓴다.
  3. 표준출력·에러를 그대로 스트리밍, 원격 종료코드로 끝난다. `~/.kqc/logs/<repo>-<sha12>-<ts>.log`
     에도 남긴다(simnode).
- simnode 에서 직접 부르면(호스트가 simnode) ssh 없이 같은 절차.
- `kqc runs ls <label> [--repo-root .]`, `kqc runs sync <repo_root>`, `kqc oos define`,
  `kqc prereg lock`.

git 깨끗함 검사는 `KQC_RUNS_ROOT` 가 있으면 코드는 worktree(`cwd`), 기록은 원 체크아웃에 둔다.
기록 파일 커밋은 호출자(연구 세션)의 몫이다 — 원 체크아웃의 `research/runs` 변경은 worktree 깨끗함
검사에 영향을 주지 않는다.

## 4. 매매 레포 적용 (오늘 밤 범위)

규칙: 실주문·수집·크론 경로는 건드리지 않는다(07:50 `pull-all --ci-gate` 로 아침에 반영된다).

- 세 레포 `krx-quant-core>=0.4.0`.
- daytrade-it: `gptq backtest` CLI 에서 `start_run` 사용(label = 전략명).
- swing-it: 백테스트 진입 스크립트가 `record_trial` 을 직접 부르는 곳을 `start_run` 으로.
- scalp-it: 84번 연구 진입 스크립트(현재 진행 중)를 `start_run` 으로, 82번 `FINAL_DONE` 은 그대로 둔다
  (이미 끝난 평가 — 소급 변환은 기록을 새로 만드는 일이라 하지 않는다).

## 5. 테스트

- 단위: 가짜 git 레포(tmp, `git init`)·호스트명 monkeypatch 로 거부 사유 전부, JSONL 행 모양,
  예외 시 `failed` 행, DB 없음·DB 실패 시 계속, OOS 규칙 표 전체, FINAL 한 번, 사전등록 해시 변경.
- CLI: `kqc run` 은 ssh 를 인자로 받는 러너 함수로 분리해 명령 문자열을 검증.
- 통합(수동, simnode): `kqc run krx-quant-core -- uv run python -c "..."` 한 번.

## 6. 내일 준비 점검 (마지막)

두 레포 `preopen_check.sh` 읽기 전용 실행(dart.db 는 08:40 갱신이라 밤에는 FAIL 이 정상), 세 레포 CI
녹색, `pull-all --check`, 2026-09-16 거래일 확인.
