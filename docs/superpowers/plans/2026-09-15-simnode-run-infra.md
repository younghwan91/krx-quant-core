# simnode 실행 기반 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 백테스트마다 호스트·git sha·데이터 구간·시도 수·결과를 기계가 기록하고, OOS 를 하드 잠금하며, trader 에서 simnode 로 원격 실행한다.

**Architecture:** `krx_quant_core.runtime` 에 순수 파이썬 모듈 3개(`gitstate`·`oos`·`runs`) + 선택 DB 색인(`runindex`) + `kqc` CLI(`cli`). 파일 기록(JSONL)이 정본, Postgres 는 실패해도 되는 색인. CLI 원격 실행은 명령 문자열을 만드는 순수 함수와 subprocess 러너로 나눈다.

**Tech Stack:** Python ≥3.11 표준 라이브러리(argparse·subprocess·json·hashlib·uuid), 선택 `psycopg[binary]>=3.1`(`[db]` extra), pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-simnode-run-infra-design.md`

## Global Constraints

- 버전 `0.4.0`. 새 필수 의존성 없음 — DB 는 `[db]` extra, 없으면 색인 건너뜀.
- 기록 위치 `research/runs/<label>/{RUNS.jsonl,TRIALS.jsonl,OOS.json,PREREG.lock,FINAL.lock}`; 루트는 `KQC_RUNS_ROOT` 환경변수가 있으면 그것, 없으면 `repo_root`.
- 시각은 KST ISO(`market.session.now_kst`).
- 예외: `RunRefused(RuntimeError)`, `OOSLocked(RunRefused)`.
- 실주문·수집·크론 경로는 어떤 레포에서도 수정 금지.
- 백테스트·통합 확인은 simnode 에서만. 단위 테스트는 호스트명 monkeypatch.

---

### Task 1: gitstate + oos

**Files:**
- Create: `src/krx_quant_core/runtime/gitstate.py`, `src/krx_quant_core/runtime/oos.py`
- Test: `tests/runtime/test_oos.py`

**Interfaces:**
- Produces: `git_head(repo: Path) -> str`, `git_dirty(repo: Path) -> bool` (추적 파일 변경만), `runs_root(repo_root: Path) -> Path` (`KQC_RUNS_ROOT` 우선, `/research/runs` 붙임), `label_dir(repo_root, label) -> Path`;
  `class RunRefused(RuntimeError)`, `class OOSLocked(RunRefused)`;
  `define_oos(label, start: str, end: str, *, repo_root) -> Path`, `lock_prereg(label, doc: Path, *, repo_root) -> Path`, `check(label, start: str, end: str, *, repo_root, final=False, run_id: str | None = None) -> None`.

- [ ] Step 1: 테스트 — 규칙 표: OOS.json 없음 통과 / 안 겹침 통과 / 겹침 비final → OOSLocked / final+PREREG 없음 거부 / final+문서 해시 변경 거부 / final 통과 시 FINAL.lock 생성 / 두 번째 final 거부 / define·lock 재실행 거부 / 경계일(start==oos.end) 겹침으로 판정 / KQC_RUNS_ROOT 반영.
- [ ] Step 2: 실패 확인 `uv run pytest tests/runtime/test_oos.py -q`
- [ ] Step 3: 구현(날짜는 `date.fromisoformat`, 겹침 = `start <= oos_end and end >= oos_start`; 해시 sha256 of bytes; JSON `sort_keys, ensure_ascii=False, indent=2`).
- [ ] Step 4: 통과 확인, ruff
- [ ] Step 5: 커밋 `feat(runtime.oos): OOS 하드 잠금`

### Task 2: runs (start_run) + runindex

**Files:**
- Create: `src/krx_quant_core/runtime/runs.py`, `src/krx_quant_core/runtime/runindex.py`
- Modify: `src/krx_quant_core/runtime/__init__.py` (export), `pyproject.toml` (`db = ["psycopg[binary]>=3.1"]`, version 0.4.0)
- Test: `tests/runtime/test_runs.py`

**Interfaces:**
- Consumes: Task 1 전부, `runtime.host.require_backtest_host`, `stats.trials.record_trial/count_trials/config_fingerprint`.
- Produces: `@dataclass(frozen=True) DataSpec(start: str, end: str, split: str | None = None)`;
  `start_run(label, config: dict, *, repo_root, data: DataSpec | None = None, seed: int | None = None, final=False, allow_dirty=False, index=True) -> ContextManager[Run]`;
  `Run.run_id: str`, `Run.n_trials: int`, `Run.log_result(result: dict) -> None`, `Run.dir: Path`;
  `read_runs(label, *, repo_root) -> list[dict]` (start·end 행을 run_id 로 합친 dict 목록);
  `runindex.upsert(row: dict) -> bool` (DSN 없거나 psycopg 없거나 실패면 False, 경고 로그), `runindex.sync(repo_root) -> int`.

- [ ] Step 1: 테스트 — simnode 아니면 RunRefused(파일 안 씀) / dirty 거부·allow_dirty 기록 / start 행 필드 / TRIALS 원장 증가·같은 config 는 안 늘어남 / 예외 시 failed 행 + 예외 재발생 / log_result 비직렬화 값 str / final=True 가 oos.check 경유(FINAL.lock) / index 가 False 반환해도 실행 계속 / read_runs 병합.
- [ ] Step 2: 실패 확인
- [ ] Step 3: 구현(행 쓰기 = open "a" + fsync 불필요; DB upsert 는 `KR_QUANT_DB` 있을 때만, `CREATE TABLE IF NOT EXISTS kqc_runs(...)` 후 `INSERT ... ON CONFLICT (run_id) DO UPDATE`).
- [ ] Step 4: 통과, ruff
- [ ] Step 5: 커밋 `feat(runtime.runs): 실행 기록 start_run + DB 색인`

### Task 3: kqc CLI

**Files:**
- Create: `src/krx_quant_core/runtime/cli.py`
- Modify: `pyproject.toml` (`[project.scripts] kqc = "krx_quant_core.runtime.cli:main"`), README
- Test: `tests/runtime/test_cli.py`

**Interfaces:**
- Consumes: Task 1·2.
- Produces: `main(argv: list[str] | None = None) -> int`;
  `remote_script(repo: str, sha: str, cmd: list[str]) -> str` (simnode 에서 돌 bash 스크립트 문자열);
  `local_ref_check(repo_dir: Path, ref: str, runner) -> str` (sha 반환, 미푸시·dirty 면 RunRefused).
  서브커맨드: `run <repo> [--ref] -- cmd`, `runs ls <label> [--repo-root]`, `runs sync <repo_root>`, `oos define <label> --start --end [--repo-root]`, `prereg lock <label> <doc> [--repo-root]`.

- [ ] Step 1: 테스트 — remote_script 가 worktree 경로·`git worktree add --detach`·`uv sync --frozen`·env export(KR_QUANT_DB, DART_API_KEY 만)·`KQC_RUNS_ROOT`·`shlex.quote` 된 cmd 포함 / 미푸시 sha 거부(`git branch -r --contains` 비면) / dirty 거부 / oos·prereg·runs ls 서브커맨드가 파일을 만들고 출력.
- [ ] Step 2: 실패 확인
- [ ] Step 3: 구현(호스트가 simnode 면 `bash -lc script`, 아니면 `ssh simnode-local bash -s` 에 stdin 으로 스크립트).
- [ ] Step 4: 통과, ruff, 전체 스위트
- [ ] Step 5: 커밋, PR, CI, 머지, 태그 v0.4.0, simnode 통합 확인 `kqc run krx-quant-core -- uv run python -c "import krx_quant_core; print(krx_quant_core.__name__)"`

### Task 4: 매매 레포 적용

각 레포 브랜치·PR·CI 녹색 확인 후 머지. 실주문·수집·크론 파일 diff 0 을 PR 에서 확인.

- [ ] daytrade-it: `pyproject` core `>=0.4.0`, `gptq backtest` 에서 `start_run(label=strategy, config=CLI 인자 dict, repo_root=레포, data=DataSpec(start,end))` 감싸고 결과 metrics 를 `log_result`. 테스트: 호스트 monkeypatch 로 기존 CLI 테스트 통과.
- [ ] swing-it(simnode 레포): core `>=0.4.0`, `record_trial` 직접 호출 지점을 `start_run` 으로.
- [ ] scalp-it: core `>=0.4.0`, 84번 진입 스크립트 `start_run`.

### Task 5: 내일 준비 점검

- [ ] 두 `preopen_check.sh` 실행, 세 레포 CI 녹색, `pull-all --check`, 결과 보고.
