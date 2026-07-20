"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Unified Analyst schema
# ---------------------------------------------------------------------------


class SignalType(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"


class AnalystReport(BaseModel):
    """统一分析师报告 schema"""
    signal: SignalType = Field(description="信号方向")
    confidence: float = Field(ge=0, le=1, description="置信度 0-1")
    evidence: list[str] = Field(default_factory=list, description="支撑证据")
    time_horizon: str = Field(default="short", description="时间周期 short/medium/long")
    risk_flags: list[str] = Field(default_factory=list, description="风险标记")
    data_freshness: str = Field(default="realtime", description="数据时效性")
    invalidating_conditions: list[str] = Field(default_factory=list, description="失效条件")


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


# ---------------------------------------------------------------------------
# Quant Factor Analyst
# ---------------------------------------------------------------------------


class SignalDirection(str, Enum):
    """Directional signal produced by analyst agents."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class QuantFactorReport(BaseModel):
    """Structured report produced by the Quant Factor Analyst.

    Evaluates 30 gtja191 factor signals from MarketMind and produces a
    consolidated factor-level view: are the factors aligning bullishly,
    bearishly, or mixed for the instrument?
    """

    signal: SignalDirection = Field(
        description=(
            "Overall factor signal direction. Exactly one of bullish / bearish / neutral. "
            "Use bullish when a majority of factors suggest upward price pressure, "
            "bearish for downward pressure, and neutral when factors are inconclusive or mixed."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Confidence in the factor signal, from 0.0 (no confidence) to 1.0 (very confident). "
            "Based on the proportion of factors aligned with the signal direction and their "
            "historical reliability as documented in the factor metadata."
        ),
    )
    factor_summary: str = Field(
        description=(
            "Summary of the factor analysis, including: (1) which factor categories "
            "(momentum, value, quality, volatility, etc.) are driving the signal; "
            "(2) notable factor divergences; (3) any extreme factor readings (>2 std dev)."
        ),
    )
    evidence: list[str] = Field(
        description=(
            "Supporting evidence for the signal. Each entry should cite specific factors "
            "and their values, e.g. 'Factor RSI_14D: 28.3 (oversold, bullish signal)'."
        ),
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Risk flags raised by the factor analysis. Examples: 'Factor crowding — 70%+ "
            "of factors in same direction', 'Extreme volatility factor reading', "
            "'Factor momentum reversal pattern detected'."
        ),
    )


def render_quant_factor_report(report: QuantFactorReport) -> str:
    """Render a QuantFactorReport to markdown."""
    return "\n".join([
        f"**Quant Factor Signal**: {report.signal.value.upper()}",
        f"**Confidence**: {report.confidence:.2f}",
        "",
        f"**Factor Summary**: {report.factor_summary}",
        "",
        "**Evidence**:",
        *[f"- {e}" for e in report.evidence],
        "",
        "**Risk Flags**:",
        *[f"- {r}" for r in (report.risk_flags or ["None"])],
    ])


# ---------------------------------------------------------------------------
# Capital Flow Analyst
# ---------------------------------------------------------------------------


class CapitalFlowReport(BaseModel):
    """Structured report produced by the Capital Flow Analyst.

    Analyses main capital (主力资金) and northbound (北向资金) flow data
    to assess whether institutional money is flowing into or out of the
    instrument.
    """

    signal: SignalDirection = Field(
        description=(
            "Overall capital flow signal. Exactly one of bullish / bearish / neutral. "
            "Use bullish when net inflows are sustained and accelerating, bearish for "
            "sustained net outflows, and neutral when flows are mixed or flat."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Confidence in the capital flow signal, from 0.0 to 1.0. Based on the "
            "magnitude and consistency of flows, data recency, and alignment between "
            "main capital and northbound flows."
        ),
    )
    capital_flow_summary: str = Field(
        description=(
            "Summary of capital flow analysis, covering: (1) main capital net flow direction "
            "and magnitude; (2) northbound capital flow direction and magnitude; "
            "(3) flow trend (accelerating, decelerating, reversing); "
            "(4) comparison to sector peers if available."
        ),
    )
    evidence: list[str] = Field(
        description=(
            "Supporting evidence. Should include specific numbers: net inflow/outflow amounts, "
            "percentage changes, and comparison periods."
        ),
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Risk flags. Examples: 'Northbound flow reversal — past 3 days of inflows "
            "reversed by 1 day of heavy outflow', 'Main capital outflow diverges from price uptrend'."
        ),
    )
    main_capital_net: float | None = Field(
        default=None,
        description=(
            "Net main capital (主力资金) flow for the current period, in RMB. "
            "Positive = net inflow, negative = net outflow."
        ),
    )
    northbound_net: float | None = Field(
        default=None,
        description=(
            "Net northbound (北向资金) flow for the current period, in RMB. "
            "Positive = net inflow, negative = net outflow."
        ),
    )


def render_capital_flow_report(report: CapitalFlowReport) -> str:
    """Render a CapitalFlowReport to markdown."""
    parts = [
        f"**Capital Flow Signal**: {report.signal.value.upper()}",
        f"**Confidence**: {report.confidence:.2f}",
        "",
        f"**Capital Flow Summary**: {report.capital_flow_summary}",
        "",
        "**Evidence**:",
        *[f"- {e}" for e in report.evidence],
        "",
        "**Risk Flags**:",
        *[f"- {r}" for r in (report.risk_flags or ["None"])],
    ]
    if report.main_capital_net is not None:
        parts.insert(3, f"**Main Capital Net Flow**: ¥{report.main_capital_net:,.0f}")
    if report.northbound_net is not None:
        parts.insert(4, f"**Northbound Net Flow**: ¥{report.northbound_net:,.0f}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sector Rotation Analyst
# ---------------------------------------------------------------------------


class SectorRotationReport(BaseModel):
    """Structured report produced by the Sector Rotation Analyst.

    Analyses sector rotation data to determine whether the instrument's
    industry is experiencing capital inflow/outflow relative to other
    sectors, and identifies rotation trends.
    """

    signal: SignalDirection = Field(
        description=(
            "Sector rotation signal for the instrument's industry. Exactly one of "
            "bullish / bearish / neutral. Use bullish when the sector is top-ranked "
            "in capital inflow and momentum, bearish when bottom-ranked or experiencing "
            "outflows, neutral when middling."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Confidence in the sector rotation signal, from 0.0 to 1.0. Based on "
            "the consistency and magnitude of the sector's ranking and flow data."
        ),
    )
    sector_rotation_summary: str = Field(
        description=(
            "Summary of sector rotation analysis, covering: (1) the instrument's industry "
            "ranking among all sectors; (2) top and bottom 3 sectors by capital flow; "
            "(3) rotation direction (e.g. defensive -> cyclical, growth -> value); "
            "(4) how the instrument's sector fits into the overall rotation picture."
        ),
    )
    evidence: list[str] = Field(
        description=(
            "Supporting evidence. Should cite specific sector rankings, flow amounts, "
            "and comparison timeframes."
        ),
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Risk flags. Examples: 'Sector ranking dropped 10+ spots in 5 days', "
            "'Sector rotation speed accelerating — rapid capital rotation suggests uncertainty'."
        ),
    )
    sector_rank: int | None = Field(
        default=None,
        ge=1,
        description="The instrument's industry rank among all sectors (1 = best).",
    )
    sector_count: int | None = Field(
        default=None,
        ge=1,
        description="Total number of sectors in the ranking.",
    )


def render_sector_rotation_report(report: SectorRotationReport) -> str:
    """Render a SectorRotationReport to markdown."""
    parts = [
        f"**Sector Rotation Signal**: {report.signal.value.upper()}",
        f"**Confidence**: {report.confidence:.2f}",
        "",
        f"**Sector Rotation Summary**: {report.sector_rotation_summary}",
        "",
        "**Evidence**:",
        *[f"- {e}" for e in report.evidence],
        "",
        "**Risk Flags**:",
        *[f"- {r}" for r in (report.risk_flags or ["None"])],
    ]
    if report.sector_rank is not None and report.sector_count is not None:
        parts.insert(3, f"**Sector Rank**: {report.sector_rank}/{report.sector_count}")
    return "\n".join(parts)


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
