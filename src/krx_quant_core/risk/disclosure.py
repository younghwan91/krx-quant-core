"""공시·감성 리스크 게이트 — 진입 차단기이지 신호 생성기가 아니다.

daytrade-it ``domain/services/risk_gate.py`` 를 **글자 그대로** 옮겼다(키워드 표·
매매거래정지 사유 판정·결정 순서·사유 문자열 전부). 바뀐 것은 하나 — 원본이
daytrade-it ``Sentiment`` 엔티티를 import 하던 자리를 :class:`SentimentLike` 구조적
타입으로 받는다. 아래 원문 docstring 은 근거 기록이라 영어 그대로 둔다.

----

Risk-gate domain service: sentiment/disclosure as a RISK GATE, not a signal generator.

Grounding (folded into the plan from deep research, see
``.superpowers/sdd/PLAN/task-11-brief.md``):

    Finding #1 -- news sentiment alone does not reliably beat a benchmark as
    a standalone trading signal; it only helps *combined* with
    technical/ML signals (this is why ``signal_generator.py`` and
    ``trading_strategy.py`` no longer produce a standalone BUY/SELL signal
    from sentiment alone -- see those modules' docstrings). The strongest,
    most defensible *new* use of sentiment/disclosures is as a RISK GATE
    that can block or halt trading independent of whatever the
    ML/technical/composite ``SignalGenerator`` path produced -- never as a
    source of BUY/SELL direction itself.

Design decision -- entry-blocker, not liquidator (brief step 5):
    A sibling project on this machine (scalp-it, an intraday KRX
    pairs-trading system on surging small-caps) independently
    pre-registered and offline-backtested a "halt/exit on bad DART
    disclosure" rule for its own strategy. It was REJECTED by real
    backtesting: the triggering sample was too rare (2.3% of trades) to
    move net returns or win rate. That result is specific to scalp-it's
    strategy shape and does not mechanically transfer to gpt-quant-v2's
    different strategy shape -- so this feature is *not* skipped because of
    it -- but it is a real, hard-won caution against assuming a generic
    "bad sentiment -> force-exit-everything" rule pays off without
    evidence. Accordingly this gate is scoped as:

    - Generic bearish/bad sentiment -> **entry-blocker only**
      (``REJECT``/``HALT_ENTRY``). Existing positions are left alone.
    - Hard-severity DART disclosures (delisting, trading halt,
      embezzlement-class events, going-concern filings) -> the one
      exception that *also* forces an exit of any existing position,
      because these categories are unambiguous and severe enough that
      riding them out is not a defensible default -- unlike a generic
      negative sentiment score, which is noisy and, per scalp-it's result,
      too weak/rare a trigger on its own to justify liquidating a working
      position.

Disclosure-severity taxonomy:
    ``disclosure_type`` (see ``infrastructure/external/dart_client.py``) is
    populated directly from OpenDART's raw ``report_nm`` filing-title field
    (confirmed by reading ``krx_news_client``'s ``DartScraper``, which sets
    ``disclosure_type=item.get("report_nm", "")``) -- free text, not a coded
    enum. Classification here is therefore keyword/substring matching
    against that text, the same way a human analyst reads a DART headline.

    Sources (public Korean regulatory/market-convention information,
    researched the same way as Task 9's tick-size research):

    - 찾기쉬운 생활법령정보 (법제처), "관리종목 지정 및 상장폐지":
      https://www.easylaw.go.kr/CSP/CnpClsMain.laf?csmSeq=1701&ccfNo=1&cciNo=2&cnpClsNo=2
      -- confirms 횡령·배임, 감사의견 부적정/의견거절(중대한 회계기준 위반
      포함) 등이 관리종목지정/상장폐지실질심사/상장폐지 사유에 해당함을
      명시.
    - KRX KIND, 유가증권시장 공시·상장 업무해설서 (공식 상장규정 해설):
      https://kind.krx.co.kr/external/dst/reference/11635/ -- 관리종목지정
      -> 상장폐지실질심사 -> 상장폐지, 그리고 매매거래정지를 KRX의 단계적
      투자자 보호 조치로 설명.
    - DART 기업공시 길라잡이 (금융감독원), "주요사항 보고서":
      https://dart.fss.or.kr/info/main.do?menu=220 -- 회생절차개시신청,
      부도발생, 은행거래정지 등이 주요사항보고서의 최우선순위 즉시공시
      항목임을 확인.
    - General, well-established domain knowledge of DART/KRX standard
      수시공시/주요사항보고서 report-title conventions (e.g.
      "횡령ㆍ배임혐의발생", "감사보고서제출" with 의견거절/부적정,
      "상장폐지결정", "관리종목지정", "매매거래정지", "회생절차개시신청",
      "파산신청", "부도발생", "은행거래정지") -- these specific report
      titles are standard OpenDART report-name conventions, classified here
      with reasonable confidence per the task's allowance to cite own
      knowledge when web research alone doesn't nail the exact string.

    Left **UNCLASSIFIED** (default: no gate, ``PASS``) because severity is
    genuinely ambiguous without more context (deal size, company health)
    and guessing would be irresponsible: 최대주주변경, 유상증자/무상증자,
    전환사채발행, 자기주식취득/처분, 단일판매·공급계약체결,
    실적공시/잠정실적, 정기보고서(분기/반기/사업보고서), and anything else
    not matched below. Routine filings must not gate trading -- that is the
    whole point of having a taxonomy instead of gating on "any disclosure".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SentimentLike(Protocol):
    """RiskGate 가 감성 객체에서 읽는 것 전부.

    daytrade-it ``domain.value_objects.Sentiment``(pydantic)가 그대로 만족한다 —
    ``score``·``confidence`` 필드와 ``is_bearish`` 프로퍼티. 코어가 daytrade-it
    엔티티를 import 하지 않으려고 구조적 타입으로 받는다. ``is_bearish`` 를 label
    비교로 다시 짜지 않는 이유: 원본 판정이 ``sentiment.is_bearish`` 를 부르므로
    그 정의를 따라가는 게 동작 보존이다.
    """

    @property
    def score(self) -> float: ...

    @property
    def confidence(self) -> float: ...

    @property
    def is_bearish(self) -> bool: ...


class DisclosureSeverity(StrEnum):
    """Severity tier for a DART disclosure's ``disclosure_type`` (report_nm) text."""

    HARD = "HARD"
    """Delisting / trading-halt / embezzlement-breach-of-trust / going-concern
    class events. See module docstring for the researched taxonomy and
    sources. Forces FORCE_EXIT (if a position is open) or
    REJECT/HALT_ENTRY (otherwise)."""

    UNCLASSIFIED = "UNCLASSIFIED"
    """No confident severity classification -- default: no gate (PASS).
    Includes both genuinely routine filings and anything not researched
    with enough confidence to classify (see module docstring)."""


class GateAction(StrEnum):
    """Action a :class:`RiskGate` evaluation can force, independent of the
    ``TradingSignal`` the ML/technical/composite path produced."""

    PASS = "PASS"
    """No gate condition met -- defer entirely to the normal signal/risk path."""

    REJECT = "REJECT"
    """Reject this specific new-entry attempt (a signal/order being
    evaluated right now)."""

    HALT_ENTRY = "HALT_ENTRY"
    """Block new entries into this ticker more broadly (e.g. checked before
    signal generation even runs, not tied to one specific signal)."""

    FORCE_EXIT = "FORCE_EXIT"
    """Force-close any existing position in this ticker. Reserved for
    hard-severity disclosures only -- see design decision in module
    docstring."""


# Substrings matched against `disclosure_type` (DART's report_nm free text).
# Matching is intentionally conservative/exact-phrase: prefer a false
# negative (a routine filing slips through unclassified) over a false
# positive (a routine filing wrongly halts trading) -- per the brief's
# "leave unclassified rather than guess" instruction.
_HARD_SEVERITY_KEYWORDS: tuple[str, ...] = (
    "상장폐지",  # delisting (decision / cause occurred)
    "관리종목지정",  # designated as administrative issue (관리종목)
    "매매거래정지",  # trading halt
    "감사의견거절",  # auditor: disclaimer of opinion
    "의견거절",  # (shorter form as it appears in some report_nm strings)
    "감사범위제한",  # audit scope limitation (grounds for qualified/disclaimer opinion)
    "부적정",  # adverse audit opinion
    "횡령",  # embezzlement
    "배임",  # breach of trust
    "회생절차개시신청",  # application for court receivership/rehabilitation
    "파산신청",  # bankruptcy filing
    "부도발생",  # default occurred
    "은행거래정지",  # bank transaction suspension
)


#: Halt-family titles ("주권매매거래정지 (사유)", "...해제 (...)", "...기간변경 (...)").
_HALT_KEYWORD = "매매거래정지"

#: A trading-halt title is HARD only when its stated reason contains one of
#: these. Checked against scalp-it's data/dart.db (26,551 OpenDART titles,
#: 2026-07..09, 2026-09-13): of the 121+87+20 halt/lift/extension notices,
#: most are mechanical -- stock split/merge re-listings (81 + 55), bonus
#: issues, capital reductions, and the 30-minute pause after a material
#: disclosure such as "(단일판매공급계약)", which is the very news a
#: news-driven BUY acts on. Gating on the bare word would block those.
#: Rumor/press-inquiry halts ("풍문") stay unclassified: they resolve either
#: way, and the embezzlement-rumor case is still caught by "횡령"/"배임".
_HARD_HALT_REASON_KEYWORDS: tuple[str, ...] = (
    "상장폐지",
    "상장적격성",  # listing-eligibility review
    "개선기간",  # improvement period granted in a delisting review
    "관리종목",
    "투자자 보호",
    "투자자보호",
    "회생",
    "파산",
    "부도",
    "횡령",
    "배임",
    "의견거절",
    "부적정",
    "감사범위제한",
)


def _is_hard_halt(disclosure_type: str) -> bool:
    """Judge a halt-family title by its reason; no stated reason stays HARD."""
    reason = disclosure_type.split(_HALT_KEYWORD, 1)[1]
    # Drop the title's own suffix words, keep only the parenthesized reason.
    reason = reason.split("(", 1)[1] if "(" in reason else ""
    if not reason.strip(" )"):
        return True
    return any(keyword in reason for keyword in _HARD_HALT_REASON_KEYWORDS)


def classify_disclosure_severity(disclosure_type: str | None) -> DisclosureSeverity:
    """Classify a DART ``disclosure_type`` (report_nm) string by severity.

    Args:
        disclosure_type: Raw DART report title text, e.g. from
            ``NewsArticle.disclosure_type`` / ``dart_client.py``. ``None``
            or unmatched text classifies as UNCLASSIFIED (no gate).

    Returns:
        ``DisclosureSeverity.HARD`` if the text matches a researched
        hard-severity category, else ``DisclosureSeverity.UNCLASSIFIED``.
    """
    if not disclosure_type:
        return DisclosureSeverity.UNCLASSIFIED
    if _HALT_KEYWORD in disclosure_type:
        return (
            DisclosureSeverity.HARD
            if _is_hard_halt(disclosure_type)
            else DisclosureSeverity.UNCLASSIFIED
        )
    for keyword in _HARD_SEVERITY_KEYWORDS:
        if keyword in disclosure_type:
            return DisclosureSeverity.HARD
    return DisclosureSeverity.UNCLASSIFIED


@dataclass(frozen=True)
class RiskGateConfig:
    """Configuration for sentiment-driven entry-blocking.

    Disclosure severity is not tunable (it's a researched taxonomy, not a
    dial); only the generic-bad-sentiment entry-blocker threshold is.

    Attributes:
        bearish_score_threshold: Sentiment score at/below this value is
            eligible to gate (score is in [-1.0, 1.0]).
        bearish_confidence_threshold: Sentiment confidence at/above this
            value is eligible to gate. Deliberately stricter than
            ``SignalGenerator.min_confidence``'s 0.5 default, since this
            gate blocks trading outright rather than just weighting a
            signal -- see the scalp-it caution in the module docstring.
    """

    bearish_score_threshold: float = -0.5
    bearish_confidence_threshold: float = 0.6


@dataclass(frozen=True)
class RiskGateDecision:
    """Result of a :class:`RiskGate` evaluation.

    Attributes:
        action: The forced action (or PASS if no gate condition met).
        reason: Human-readable reason for the decision.
        severity: The disclosure severity classification used, if any.
        details: Additional details about the decision (e.g. the matched
            disclosure text or sentiment values).
    """

    action: GateAction
    reason: str
    severity: DisclosureSeverity = DisclosureSeverity.UNCLASSIFIED
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks_entry(self) -> bool:
        """True if this decision blocks a new entry (any action but PASS)."""
        return self.action != GateAction.PASS

    @property
    def forces_exit(self) -> bool:
        """True if this decision forces closing an existing position."""
        return self.action == GateAction.FORCE_EXIT


class RiskGate:
    """Domain service: sentiment/disclosure as a risk gate, independent of ``TradingSignal``.

    See the module docstring for the full design rationale, taxonomy, and
    sources. In short: hard-severity disclosures force an exit (and block
    entries); everything else -- including generic bad sentiment -- only
    ever blocks *new* entries, never forces an exit of an existing
    position.
    """

    def __init__(self, config: RiskGateConfig | None = None) -> None:
        """Initialize the risk gate with configuration.

        Args:
            config: Entry-blocker sentiment thresholds, uses defaults if
                not provided.
        """
        self.config = config or RiskGateConfig()

    def evaluate(
        self,
        *,
        sentiment: SentimentLike | None = None,
        disclosure_type: str | None = None,
        has_open_position: bool = False,
        is_new_entry_attempt: bool = True,
    ) -> RiskGateDecision:
        """Evaluate the risk gate for a ticker.

        Args:
            sentiment: Latest sentiment for the ticker (Task 7's
                toss_news_client/dart_client output), if available.
            disclosure_type: Latest DART ``disclosure_type`` (report_nm)
                for the ticker, if available.
            has_open_position: Whether the account currently holds a
                position in this ticker.
            is_new_entry_attempt: ``True`` when evaluating one specific
                new-entry signal/order (returns ``REJECT`` on block);
                ``False`` when checking whether entries should be halted
                more broadly, independent of any one signal (returns
                ``HALT_ENTRY`` on block).

        Returns:
            ``RiskGateDecision`` with the forced action, or
            ``GateAction.PASS`` if neither sentiment nor disclosure trips
            the gate.
        """
        severity = classify_disclosure_severity(disclosure_type)

        if severity is DisclosureSeverity.HARD:
            if has_open_position:
                return RiskGateDecision(
                    action=GateAction.FORCE_EXIT,
                    reason=f"Hard-severity DART disclosure forces exit: {disclosure_type!r}",
                    severity=severity,
                    details={"disclosure_type": disclosure_type},
                )
            action = GateAction.REJECT if is_new_entry_attempt else GateAction.HALT_ENTRY
            return RiskGateDecision(
                action=action,
                reason=f"Hard-severity DART disclosure blocks new entries: {disclosure_type!r}",
                severity=severity,
                details={"disclosure_type": disclosure_type},
            )

        if sentiment is not None and self._is_gating_bearish(sentiment):
            action = GateAction.REJECT if is_new_entry_attempt else GateAction.HALT_ENTRY
            return RiskGateDecision(
                action=action,
                reason=(
                    "Bearish sentiment halts new entries only (entry-blocker "
                    "design -- does not force-exit existing positions; see "
                    "module docstring for the scalp-it-informed rationale)"
                ),
                severity=severity,
                details={
                    "sentiment_score": sentiment.score,
                    "sentiment_confidence": sentiment.confidence,
                },
            )

        return RiskGateDecision(
            action=GateAction.PASS,
            reason="No gate condition met",
            severity=severity,
        )

    def _is_gating_bearish(self, sentiment: SentimentLike) -> bool:
        """Whether sentiment is bearish enough to trigger the entry-blocker."""
        return (
            sentiment.is_bearish
            and sentiment.score <= self.config.bearish_score_threshold
            and sentiment.confidence >= self.config.bearish_confidence_threshold
        )


__all__ = [
    "DisclosureSeverity",
    "GateAction",
    "RiskGate",
    "RiskGateConfig",
    "RiskGateDecision",
    "SentimentLike",
    "classify_disclosure_severity",
]
