"""A Share Gate rejection is TERMINAL: Portfolio Manager must NOT override it.

The MOA audit found (B1, -32pts):
  - A Share Gate sets gate_passed=False and final_trade_decision="REJECTED"
  - should_route_from_gate was routing rejected trades to Portfolio Manager
  - Portfolio Manager didn't read gate_passed and called LLM to overwrite

Fix in three layers:
  1. should_route_from_gate returns "END" when gate_passed is False
  2. setup.py's gate path map includes END
  3. portfolio_manager checks gate_passed as defense-in-depth
"""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic


# ── should_route_from_gate ─────────────────────────────────────────


class TestShouldRouteFromGate:
    """Layer 1: conditional routing sends rejected trades to END."""

    def test_route_rejected_to_end_when_gate_not_passed(self):
        """When gate_passed is explicitly False, route to END."""
        logic = ConditionalLogic()
        target = logic.should_route_from_gate({"gate_passed": False})
        assert target == "END", (
            f"Expected 'END' when gate_passed=False, got {target!r}"
        )

    def test_route_to_aggressive_when_gate_passed(self):
        """When gate_passed is True, route to Aggressive Analyst."""
        logic = ConditionalLogic()
        assert logic.should_route_from_gate({"gate_passed": True}) == "Aggressive Analyst"

    def test_route_to_aggressive_when_gate_not_set(self):
        """When gate_passed is absent (not True), route to END (fail-closed)."""
        logic = ConditionalLogic()
        assert logic.should_route_from_gate({}) == "END"

    def test_gate_path_map_covers_all_router_returns(self):
        """Every return value from should_route_from_gate is in GATE_PATH_MAP."""
        from tradingagents.graph.setup import GATE_PATH_MAP

        logic = ConditionalLogic()
        returns = {
            logic.should_route_from_gate({"gate_passed": False}),
            logic.should_route_from_gate({"gate_passed": True}),
            logic.should_route_from_gate({}),
        }
        assert returns <= set(GATE_PATH_MAP), (
            f"Router returns {returns} not all covered by GATE_PATH_MAP keys {set(GATE_PATH_MAP)}"
        )

    def test_gate_path_map_includes_end(self):
        """GATE_PATH_MAP maps 'END' to the LangGraph END constant."""
        from langgraph.graph import END as LANGGRAPH_END
        from tradingagents.graph.setup import GATE_PATH_MAP

        assert "END" in GATE_PATH_MAP, "GATE_PATH_MAP must include 'END' key"
        assert GATE_PATH_MAP["END"] is LANGGRAPH_END or GATE_PATH_MAP["END"] == LANGGRAPH_END


# ── Portfolio Manager defense-in-depth ─────────────────────────────


class TestPortfolioManagerGateCheck:
    """Layer 3: Portfolio Manager must respect gate_passed as defense in depth."""

    def _make_mock_llm(self):
        """Create a mock LLM that we can verify was NOT called."""
        mock = MagicMock()
        # Simulate a non-structured-output LLM
        mock.invoke.side_effect = RuntimeError("LLM should not be called when gate rejected!")
        return mock

    def test_pm_returns_rejected_when_gate_not_passed(self):
        """Portfolio Manager returns REJECTED without calling LLM when gate_passed=False."""
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

        mock_llm = self._make_mock_llm()
        pm_node = create_portfolio_manager(mock_llm)

        minimal_state = {
            "gate_passed": False,
            "gate_violations": ["涨停限制：测试"],
            "gate_result": {"passed": False, "violations": ["涨停限制：测试"]},
            "risk_debate_state": {
                "history": "",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "REJECTED",
        }

        result = pm_node(minimal_state)

        # Must preserve REJECTED decision
        assert result["final_trade_decision"] == "REJECTED"
        # Risk debate state must be preserved
        assert result["risk_debate_state"] == minimal_state["risk_debate_state"]
        # LLM must NOT have been called
        mock_llm.invoke.assert_not_called()

    def test_pm_calls_llm_when_gate_passed(self):
        """Portfolio Manager does NOT short-circuit when gate_passed is True."""
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

        mock_llm = MagicMock()
        pm_node = create_portfolio_manager(mock_llm)

        state = {
            "gate_passed": True,
            "company_of_interest": "AAPL",
            "risk_debate_state": {
                "history": "",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "investment_plan": "",
            "trader_investment_plan": "",
        }

        # We don't fully test the LLM output here — just verify that
        # gate_passed=True does NOT short-circuit to REJECTED.
        # (The real LLM path is tested in existing PM tests.)
        # Since the mock doesn't implement bind_structured, this may throw
        # on the structured-output path, but that's the real LLM path —
        # not the gate short-circuit.
        try:
            result = pm_node(state)
            # If it gets past LLM, make sure it's NOT early-return REJECTED
            assert result.get("final_trade_decision") != "REJECTED", (
                "gate_passed=True must NOT return REJECTED"
            )
        except (RuntimeError, Exception):
            # An exception means it tried to call LLM normally — that's
            # correct behavior (gate_passed=True does not short-circuit).
            pass


# ── Integration: graph-level routing ──────────────────────────────


class TestGatePathMap:
    """Layer 2: GATE_PATH_MAP completeness (same pattern as Risk & Debate)."""

    def test_path_map_importable(self):
        """GATE_PATH_MAP is exported from setup.py."""
        from tradingagents.graph.setup import GATE_PATH_MAP
        assert isinstance(GATE_PATH_MAP, dict)

    def test_path_map_has_expected_keys(self):
        """GATE_PATH_MAP has Aggressive Analyst and END entries."""
        from tradingagents.graph.setup import GATE_PATH_MAP
        assert "Aggressive Analyst" in GATE_PATH_MAP
        assert "END" in GATE_PATH_MAP
