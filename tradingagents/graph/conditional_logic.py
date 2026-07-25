# TradingAgents/graph/conditional_logic.py

import time

from tradingagents.agents.utils.agent_states import AgentState


def _is_conflict(bull_judgment: str, bear_judgment: str) -> bool:
    """Check if bull and bear signals conflict (one bullish, one bearish).

    Returns True when one side is clearly bullish and the other clearly bearish.
    """
    bull_bullish = any(
        w in bull_judgment.lower() for w in ["buy", "bullish", "overweight", "long"]
    )
    bear_bearish = any(
        w in bear_judgment.lower() for w in ["sell", "bearish", "underweight", "short"]
    )
    return bull_bullish and bear_bearish


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    MAX_RISK_DEBATE_TIME = 300  # 5 分钟超时

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if sentiment-analyst tool round should continue.

        Method name keeps the legacy ``social`` suffix to match the
        ``AnalystType.SOCIAL = "social"`` wire value (saved-config
        back-compat); the returned ``clear_node`` label uses the v0.2.5
        rename so it matches the node registered by the execution plan.
        """
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Sentiment"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_quant_factor(self, state: AgentState):
        """Determine if quant factor analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_quant_factor"
        return "Msg Clear Quant"

    def should_continue_capital_flow(self, state: AgentState):
        """Determine if capital flow analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_capital_flow"
        return "Msg Clear Capital"

    def should_continue_sector_rotation(self, state: AgentState):
        """Determine if sector rotation analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_sector_rotation"
        return "Msg Clear Sector"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue, with dynamic round count."""

        debate_state = state.get("investment_debate_state", {})
        count = debate_state.get("count", 0)
        max_rounds = self.max_debate_rounds

        # Dynamic escalation: if bull and bear signals conflict, auto-increase to 2 rounds
        bull_judgment = debate_state.get("bull_judgment", "")
        bear_judgment = debate_state.get("bear_judgment", "")
        if _is_conflict(bull_judgment, bear_judgment) and max_rounds < 2:
            max_rounds = 2

        if count >= 2 * max_rounds:  # 2 agents × max_rounds back-and-forth
            return "Research Manager"
        if debate_state.get("current_response", "").startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_enter_risk_debate(self, state: AgentState) -> str:
        """Check if Trader wants to trade — skip risk debate if HOLD."""

        trader_plan = state.get("trader_investment_plan", "")

        # early-exit: if Trader outputs HOLD (and no buy/sell), skip risk debate
        lower = trader_plan.lower()
        if "hold" in lower and "buy" not in lower and "sell" not in lower:
            return "Portfolio Manager"

        return "Aggressive Analyst"

    def should_route_from_gate(self, state: AgentState) -> str:
        """Route from A Share Gate: go to risk debate if gate passed, or END if rejected.

        The gate node sets ``gate_passed`` on the state. If it's explicitly False
        (rule violation), the trade is REJECTED and the graph terminates — the
        Portfolio Manager must NOT be allowed to override the gate decision (B1).
        Otherwise proceed to Aggressive Analyst for normal risk debate.
        """
        gate_passed = state.get("gate_passed")
        if gate_passed is not True:
            return "END"
        return "Aggressive Analyst"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        # 超时兜底
        start_time = state.get("risk_debate_start_time", 0)
        if start_time and time.time() - start_time > self.MAX_RISK_DEBATE_TIME:
            return "Portfolio Manager"  # 超时直接进 PM

        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
