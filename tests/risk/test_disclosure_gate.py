"""daytrade-it tests/unit/domain/test_risk_gate.py 이식(원문 유지).

Unit tests for the risk-gate domain service (Task 11 / Phase 4a).

Covers each disclosure-severity category's REJECT/HALT_ENTRY/FORCE_EXIT
decision, the entry-blocker-only scoping of generic bad sentiment (never
forces an exit), and the "leave unclassified rather than guess" default.
"""

from __future__ import annotations

from dataclasses import dataclass

from krx_quant_core.risk import (
    DisclosureSeverity,
    GateAction,
    RiskGate,
    RiskGateConfig,
    SentimentLike,
    classify_disclosure_severity,
)


@dataclass(frozen=True)
class Sentiment:
    """daytrade-it ``Sentiment`` 의 대역 — RiskGate 가 읽는 모양만 같다."""

    score: float
    label: str
    confidence: float

    @property
    def is_bearish(self) -> bool:
        return self.label == "BEARISH"

    @classmethod
    def bearish(cls, score: float = -0.8, confidence: float = 0.9) -> Sentiment:
        return cls(score, "BEARISH", confidence)

    @classmethod
    def bullish(cls, score: float = 0.8, confidence: float = 0.9) -> Sentiment:
        return cls(score, "BULLISH", confidence)

    @classmethod
    def neutral(cls, score: float = 0.0, confidence: float = 0.5) -> Sentiment:
        return cls(score, "NEUTRAL", confidence)


def test_stand_in_satisfies_protocol() -> None:
    assert isinstance(Sentiment.bearish(), SentimentLike)


def test_bearish_label_is_what_gates_not_score_alone() -> None:
    # 원본은 sentiment.is_bearish 를 먼저 본다 — 점수만 낮고 label 이 NEUTRAL 이면 통과.
    gate = RiskGate()
    decision = gate.evaluate(sentiment=Sentiment(-0.9, "NEUTRAL", 0.99))
    assert decision.action == GateAction.PASS


class TestClassifyDisclosureSeverity:
    """The researched taxonomy: hard-severity categories vs. unclassified/routine."""

    def test_none_is_unclassified(self) -> None:
        assert classify_disclosure_severity(None) is DisclosureSeverity.UNCLASSIFIED

    def test_empty_string_is_unclassified(self) -> None:
        assert classify_disclosure_severity("") is DisclosureSeverity.UNCLASSIFIED

    def test_routine_filing_is_unclassified(self) -> None:
        # Routine filings must default to no-gate -- this is the whole
        # point of a taxonomy instead of gating on "any disclosure".
        routine_titles = [
            "분기보고서 (2026.06)",
            "주요사항보고서(유상증자결정)",
            "주요사항보고서(전환사채권발행결정)",
            "자기주식취득결정",
            "최대주주변경",
            "단일판매ㆍ공급계약체결",
            "잠정실적(공정공시)",
        ]
        for title in routine_titles:
            assert classify_disclosure_severity(title) is DisclosureSeverity.UNCLASSIFIED, title

    def test_delisting_is_hard(self) -> None:
        assert classify_disclosure_severity("상장폐지결정") is DisclosureSeverity.HARD

    def test_administrative_designation_is_hard(self) -> None:
        assert classify_disclosure_severity("관리종목지정") is DisclosureSeverity.HARD

    def test_trading_halt_is_hard(self) -> None:
        assert classify_disclosure_severity("매매거래정지") is DisclosureSeverity.HARD

    def test_audit_opinion_disclaimer_is_hard(self) -> None:
        assert (
            classify_disclosure_severity("감사보고서제출(감사의견거절)") is DisclosureSeverity.HARD
        )

    def test_audit_opinion_adverse_is_hard(self) -> None:
        assert classify_disclosure_severity("감사보고서제출(부적정)") is DisclosureSeverity.HARD

    def test_embezzlement_is_hard(self) -> None:
        assert classify_disclosure_severity("횡령ㆍ배임혐의발생") is DisclosureSeverity.HARD

    def test_breach_of_trust_alone_is_hard(self) -> None:
        assert classify_disclosure_severity("배임 혐의 확인") is DisclosureSeverity.HARD

    def test_rehabilitation_filing_is_hard(self) -> None:
        assert classify_disclosure_severity("회생절차개시신청") is DisclosureSeverity.HARD

    def test_bankruptcy_filing_is_hard(self) -> None:
        assert classify_disclosure_severity("파산신청") is DisclosureSeverity.HARD

    def test_default_occurred_is_hard(self) -> None:
        assert classify_disclosure_severity("부도발생") is DisclosureSeverity.HARD

    def test_bank_transaction_suspension_is_hard(self) -> None:
        assert classify_disclosure_severity("은행거래정지") is DisclosureSeverity.HARD


class TestTradingHaltReasons:
    """Trading-halt titles are judged by their stated reason.

    Every title below is copied verbatim from scalp-it's data/dart.db
    (OpenDART report_nm, 2026-07..09). Most halts on KRX are mechanical --
    a stock split's re-listing, a bonus issue, or the 30-minute pause that
    follows a material contract -- and gating on the bare word would block
    exactly the news this system buys.
    """

    def test_bare_halt_without_reason_stays_hard(self) -> None:
        # No reason given -> can't tell it's benign -> conservative.
        assert classify_disclosure_severity("[기재정정]주권매매거래정지") is DisclosureSeverity.HARD

    def test_mechanical_halts_are_unclassified(self) -> None:
        mechanical = [
            "주권매매거래정지 (주식의 병합, 분할 등 전자등록 변경, 말소)",
            "주권매매거래정지해제 (액면병합 주권 변경상장)",
            "주권매매거래정지해제 (감자 주권 변경상장)",
            "주권매매거래정지 (단일판매공급계약)",
            "주권매매거래정지 (무상증자)",
            "매매거래정지및정지해제(중요내용공시)",
            "주권매매거래정지기간변경 (주식의 병합, 분할 등 전자등록 변경, 말소)",
            "주권매매거래정지해제 (상장유지 결정)",
            "주권매매거래정지 (풍문 또는 보도 관련)",
        ]
        for title in mechanical:
            assert classify_disclosure_severity(title) is DisclosureSeverity.UNCLASSIFIED, title

    def test_halts_with_a_hard_reason_stay_hard(self) -> None:
        hard = [
            "주권매매거래정지 (상장폐지 사유발생)",
            "주권매매거래정지해제 (상장폐지에 따른 정리매매 개시)",
            "주권매매거래정지기간변경 (상장적격성 실질심사 대상(사유발생))",
            "주권매매거래정지기간변경 (개선기간 부여)",
            "주권매매거래정지 (투자자 보호)",
            "주권매매거래정지기간변경 (회생절차 개시결정)",
            "주권매매거래정지기간변경 (풍문사유(현직 임원의 횡령·배임혐의설) 미해소)",
        ]
        for title in hard:
            assert classify_disclosure_severity(title) is DisclosureSeverity.HARD, title

    def test_non_halt_hard_titles_are_unchanged(self) -> None:
        # The reason rule applies to halt titles only.
        assert classify_disclosure_severity("상장폐지결정") is DisclosureSeverity.HARD
        assert classify_disclosure_severity("횡령ㆍ배임혐의발생") is DisclosureSeverity.HARD
        assert (
            classify_disclosure_severity("[기재정정]주요사항보고서(회생절차개시신청)")
            is DisclosureSeverity.HARD
        )


class TestRiskGateDisclosureDriven:
    """Hard-severity disclosures force REJECT/HALT_ENTRY/FORCE_EXIT."""

    def test_hard_disclosure_no_position_new_entry_attempt_rejects(self) -> None:
        gate = RiskGate()
        decision = gate.evaluate(
            disclosure_type="상장폐지결정",
            has_open_position=False,
            is_new_entry_attempt=True,
        )
        assert decision.action == GateAction.REJECT
        assert decision.severity == DisclosureSeverity.HARD
        assert decision.blocks_entry
        assert not decision.forces_exit

    def test_hard_disclosure_no_position_broad_check_halts_entry(self) -> None:
        gate = RiskGate()
        decision = gate.evaluate(
            disclosure_type="관리종목지정",
            has_open_position=False,
            is_new_entry_attempt=False,
        )
        assert decision.action == GateAction.HALT_ENTRY
        assert decision.severity == DisclosureSeverity.HARD
        assert decision.blocks_entry
        assert not decision.forces_exit

    def test_hard_disclosure_with_open_position_forces_exit(self) -> None:
        gate = RiskGate()
        decision = gate.evaluate(
            disclosure_type="횡령ㆍ배임혐의발생",
            has_open_position=True,
            is_new_entry_attempt=True,
        )
        assert decision.action == GateAction.FORCE_EXIT
        assert decision.severity == DisclosureSeverity.HARD
        assert decision.blocks_entry
        assert decision.forces_exit

    def test_hard_disclosure_with_open_position_forces_exit_regardless_of_entry_attempt_flag(
        self,
    ) -> None:
        gate = RiskGate()
        decision = gate.evaluate(
            disclosure_type="매매거래정지",
            has_open_position=True,
            is_new_entry_attempt=False,
        )
        assert decision.action == GateAction.FORCE_EXIT

    def test_unclassified_disclosure_passes(self) -> None:
        gate = RiskGate()
        decision = gate.evaluate(
            disclosure_type="주요사항보고서(유상증자결정)",
            has_open_position=True,
        )
        assert decision.action == GateAction.PASS
        assert not decision.blocks_entry
        assert not decision.forces_exit

    def test_no_disclosure_no_sentiment_passes(self) -> None:
        gate = RiskGate()
        decision = gate.evaluate()
        assert decision.action == GateAction.PASS


class TestRiskGateSentimentDriven:
    """Generic bad sentiment is an entry-blocker only -- never forces an exit.

    This is the design decision from the brief (step 5), informed by the
    scalp-it sibling project's rejected "halt/exit on bad sentiment"
    backtest: reserve forced-exit for hard-severity disclosures, not
    generic negative sentiment scores.
    """

    def test_strongly_bearish_high_confidence_sentiment_rejects_new_entry(self) -> None:
        gate = RiskGate()
        sentiment = Sentiment.bearish(score=-0.8, confidence=0.9)
        decision = gate.evaluate(
            sentiment=sentiment, has_open_position=False, is_new_entry_attempt=True
        )
        assert decision.action == GateAction.REJECT
        assert decision.blocks_entry
        assert not decision.forces_exit

    def test_strongly_bearish_sentiment_never_forces_exit_even_with_open_position(self) -> None:
        gate = RiskGate()
        sentiment = Sentiment.bearish(score=-0.95, confidence=0.99)
        decision = gate.evaluate(
            sentiment=sentiment, has_open_position=True, is_new_entry_attempt=True
        )
        # Entry-blocker only: a bad sentiment score, however extreme, must
        # never force-exit an existing position -- only hard-severity
        # disclosures do that.
        assert decision.action != GateAction.FORCE_EXIT
        assert decision.action == GateAction.REJECT

    def test_mildly_bearish_sentiment_below_threshold_passes(self) -> None:
        gate = RiskGate()
        sentiment = Sentiment.bearish(score=-0.2, confidence=0.9)
        decision = gate.evaluate(sentiment=sentiment)
        assert decision.action == GateAction.PASS

    def test_low_confidence_bearish_sentiment_passes(self) -> None:
        gate = RiskGate()
        sentiment = Sentiment.bearish(score=-0.9, confidence=0.3)
        decision = gate.evaluate(sentiment=sentiment)
        assert decision.action == GateAction.PASS

    def test_bullish_sentiment_never_gates(self) -> None:
        gate = RiskGate()
        sentiment = Sentiment.bullish(score=0.9, confidence=0.95)
        decision = gate.evaluate(sentiment=sentiment)
        assert decision.action == GateAction.PASS

    def test_neutral_sentiment_never_gates(self) -> None:
        gate = RiskGate()
        sentiment = Sentiment.neutral()
        decision = gate.evaluate(sentiment=sentiment)
        assert decision.action == GateAction.PASS

    def test_custom_config_thresholds(self) -> None:
        gate = RiskGate(
            RiskGateConfig(bearish_score_threshold=-0.3, bearish_confidence_threshold=0.5)
        )
        sentiment = Sentiment.bearish(score=-0.4, confidence=0.6)
        decision = gate.evaluate(sentiment=sentiment, is_new_entry_attempt=False)
        assert decision.action == GateAction.HALT_ENTRY


class TestRiskGateDisclosurePrecedence:
    """Hard disclosure severity takes precedence over sentiment in the same evaluation."""

    def test_hard_disclosure_wins_over_bullish_sentiment(self) -> None:
        gate = RiskGate()
        sentiment = Sentiment.bullish(score=0.9, confidence=0.95)
        decision = gate.evaluate(
            sentiment=sentiment,
            disclosure_type="상장폐지결정",
            has_open_position=True,
        )
        assert decision.action == GateAction.FORCE_EXIT
        assert decision.severity == DisclosureSeverity.HARD
