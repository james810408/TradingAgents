"""B3 Phase 2-4: TraderProposal → OrderContract 适配器测试。

覆盖：
1. adapter core: BUY/HOLD/SELL action ↔ OrderContract
2. quantity 推算（按可用现金，持仓）
3. 从 state 提取 ticker
4. metadata 传递（reasoning/stop_loss/position_sizing）
5. gate_node adapter 路径（TraderProposal → gate）
6. 显式 order_quantity 覆盖
7. HOLD 订单
"""
from __future__ import annotations

import warnings

import pytest

from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.contracts.adapter import trader_proposal_to_order_contract
from tradingagents.contracts.order_contract import OrderContract

# Import gate_node directly without triggering graph/__init__.py's full chain
from tradingagents.graph.a_share_gate import create_a_share_gate_node


# ─── 基础适配器功能 ─────────────────────────────────────────────


class TestTraderProposalToOrderContract:
    """trader_proposal_to_order_contract 核心功能测试。"""

    def test_buy_action_maps_to_buy(self):
        """BUY 映射到 OrderContract action=BUY。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="技术指标和金叉信号",
            entry_price=150.0,
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519", {"available_cash": 100000.0},
        )
        assert order.action == "BUY"
        assert order.ticker == "600519"
        assert order.a_share is True
        assert order.agent_id == "trader"

    def test_hold_action_maps_to_hold(self):
        """HOLD 映射到 OrderContract action=HOLD。"""
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="当前无明确信号",
        )
        order = trader_proposal_to_order_contract(proposal, "000858")
        assert order.action == "HOLD"
        assert order.quantity == 0  # HOLD 时 quantity 自动为 0

    def test_sell_action_maps_to_sell(self):
        """SELL 映射到 OrderContract action=SELL。"""
        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="技术面转空",
            entry_price=200.0,
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519",
            {"position": {"shares": 300}},
        )
        assert order.action == "SELL"
        assert order.ticker == "600519"

    def test_entry_price_becomes_order_price(self):
        """entry_price 映射到 OrderContract.price。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="买入",
            entry_price=155.50,
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519", {"available_cash": 100000.0, "order_quantity": 100},
        )
        assert order.price == 155.50

    def test_hold_always_zero_quantity(self):
        """HOLD 订单 quantity 始终为 0。"""
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="观察中",
        )
        # 即使 state 中有显式 quantity，HOLD 也应返回 0
        order = trader_proposal_to_order_contract(
            proposal, "600519", {"order_quantity": 100},
        )
        assert order.quantity == 0

    def test_metadata_passes_reasoning_and_stop_loss(self):
        """reasoning 和 stop_loss 进入 metadata。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="三线金叉确认多头",
            stop_loss=145.0,
            position_sizing="10% of portfolio",
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519", {"available_cash": 100000.0, "order_quantity": 100},
        )
        assert order.metadata.get("reasoning") == "三线金叉确认多头"
        assert order.metadata.get("stop_loss") == 145.0
        assert order.metadata.get("position_sizing") == "10% of portfolio"

    def test_signal_id_from_state(self):
        """signal_id 从 state 传递。"""
        proposal = TraderProposal(action=TraderAction.BUY, reasoning="买入")
        order = trader_proposal_to_order_contract(
            proposal, "600519",
            {"signal_id": "sig-001", "available_cash": 100000.0, "order_quantity": 100},
        )
        assert order.signal_id == "sig-001"


# ─── 数量推算 ──────────────────────────────────────────────────


class TestQuantityComputation:
    """quantity 推算逻辑。"""

    def test_buy_quantity_from_available_cash(self):
        """BUY: 按可用现金 / entry_price 推算整数手。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="买入",
            entry_price=150.0,
        )
        # 可用现金 100,000 → 100000 / 150 = 666.67 股 = 6 手 = 600 股
        order = trader_proposal_to_order_contract(
            proposal, "600519", {"available_cash": 100000.0},
        )
        assert order.quantity == 600

    def test_buy_without_price_falls_back_to_default_quantity(self):
        """BUY: 无 entry_price 和 state.price 时使用 default_quantity。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="买入",
            entry_price=None,
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519", {"default_quantity": 200},
        )
        assert order.quantity == 200

    def test_buy_with_large_cash(self):
        """BUY: 大额现金计算正确。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="大额买入",
            entry_price=100.0,
        )
        # 1,000,000 / 100 = 10,000 股 = 100 手
        order = trader_proposal_to_order_contract(
            proposal, "600519", {"available_cash": 1_000_000.0},
        )
        assert order.quantity == 10_000

    def test_sell_quantity_from_position(self):
        """SELL: quantity 从持仓提取。"""
        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="卖出",
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519",
            {"position": {"shares": 500}},
        )
        assert order.quantity == 500

    def test_sell_with_quantity_key_in_position(self):
        """SELL: 兼容 position 中 quantity 字段。"""
        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="卖出",
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519",
            {"position": {"quantity": 300}},
        )
        assert order.quantity == 300

    def test_explicit_order_quantity_override(self):
        """BUY: 显式 order_quantity 覆盖 cash 推算。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="买入",
            entry_price=150.0,
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519",
            {"available_cash": 1_000_000.0, "order_quantity": 500},
        )
        assert order.quantity == 500  # 使用显式值，不按 cash 推算

    def test_minimum_lot_for_small_cash(self):
        """BUY: 小额现金时至少 1 手（100 股）。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="小额买入",
            entry_price=50.0,
        )
        # 可用现金 3000 → 3000/50 = 60 股 → max(0, 100) = 100
        order = trader_proposal_to_order_contract(
            proposal, "600519", {"available_cash": 3000.0},
        )
        assert order.quantity == 100

    def test_zero_cash_uses_default_quantity(self):
        """BUY: 无可用现金时使用 default_quantity。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="买入",
            entry_price=100.0,
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519",
            {"available_cash": 0.0, "default_quantity": 100},
        )
        assert order.quantity == 100

    def test_hold_zero_quantity_ignores_all(self):
        """HOLD: 无视所有 quantity 来源。"""
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="观望",
        )
        order = trader_proposal_to_order_contract(
            proposal, "600519",
            {"available_cash": 1_000_000.0, "position": {"shares": 500}},
        )
        assert order.quantity == 0


# ─── Gate Node 集成 ────────────────────────────────────────────


class TestGateNodeWithAdapter:
    """gate_node 通过 adapter 使用 TraderProposal。"""

    def test_trader_proposal_through_gate_node(self):
        """TraderProposal 通过 gate_node adapter 路径通过检查。"""
        gate = create_a_share_gate_node()
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="买入信号确认",
            entry_price=155.0,
        )
        state = {
            "trader_proposal": proposal,
            "company_of_interest": "600519",
            "price": 155.0,
            "limit_up": 170.0,
            "limit_down": 140.0,
            "available_cash": 100000.0,
        }
        result = gate(state)
        assert result["gate_passed"] is True

    def test_trader_proposal_blocked_at_limit_up(self):
        """TraderProposal 通过 adapter 路径触发涨停拒绝。"""
        gate = create_a_share_gate_node()
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="追涨停",
            entry_price=110.0,
        )
        state = {
            "trader_proposal": proposal,
            "company_of_interest": "600519",
            "price": 110.0,
            "limit_up": 110.0,
            "limit_down": 90.0,
            "available_cash": 100000.0,
            "order_quantity": 100,
        }
        result = gate(state)
        assert result["gate_passed"] is False
        assert result["final_trade_decision"] == "REJECTED"
        assert any("涨停" in v for v in result["gate_violations"])

    def test_trader_proposal_hold_passes_gate(self):
        """HOLD TraderProposal 通过 gate。"""
        gate = create_a_share_gate_node()
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="暂不操作",
        )
        state = {
            "trader_proposal": proposal,
            "company_of_interest": "000858",
        }
        result = gate(state)
        assert result["gate_passed"] is True

    def test_trader_proposal_t1_rejection(self):
        """TraderProposal T+1 拒绝通过 adapter 路径。"""
        gate = create_a_share_gate_node()
        proposal = TraderProposal(
            action=TraderAction.SELL,
            reasoning="卖出",
        )
        state = {
            "trader_proposal": proposal,
            "company_of_interest": "600519",
            "price": 100.0,
            "limit_up": 110.0,
            "limit_down": 90.0,
            "position": {"today_bought_shares": 200, "shares": 500},
        }
        result = gate(state)
        assert result["gate_passed"] is False
        assert any("T+1" in v for v in result["gate_violations"])

    def test_explicit_order_contract_still_takes_priority(self):
        """显式 OrderContract 优先级高于 TraderProposal。"""
        gate = create_a_share_gate_node()
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="买入信号",
        )
        # 显式 OrderContract 应该被优先使用
        explicit_order = OrderContract(
            action="SELL", ticker="600519", quantity=200,
        )
        state = {
            "order_contract": explicit_order,
            "trader_proposal": proposal,
            "company_of_interest": "600519",
            "price": 100.0,
            "limit_up": 110.0,
            "limit_down": 90.0,
            "position": {"today_bought_shares": 0},
        }
        result = gate(state)
        # 显式 OrderContract action=SELL 应被使用（不是 proposal 的 BUY）
        assert result["gate_passed"] is True

    def test_adapter_fallback_to_text_parsing_on_failure(self):
        """adapter 失败时优雅回退到文本解析。"""
        gate = create_a_share_gate_node()
        # 用非 TraderProposal 对象模拟 adapter 失败
        state = {
            "trader_proposal": "not_a_trader_proposal",
            "company_of_interest": "600519",
            "trader_investment_plan": "just thinking",
        }
        result = gate(state)
        assert "gate_passed" in result  # 不会崩溃


# ─── 边缘情况 ──────────────────────────────────────────────────


class TestEdgeCases:
    """适配器边缘情况。"""

    def test_empty_state(self):
        """空 state 默认 HOLD。"""
        gate = create_a_share_gate_node()
        result = gate({})
        assert result["gate_passed"] is True

    def test_trader_proposal_none_state(self):
        """state 中 trader_proposal 为 None 时走文本解析。"""
        gate = create_a_share_gate_node()
        state = {
            "trader_proposal": None,
            "trader_investment_plan": "I think we should buy",
        }
        result = gate(state)
        assert "gate_passed" in result

    def test_adapter_buy_with_price_from_state(self):
        """BUY: 使用 state.price 当 entry_price 为 None 时推算 quantity。"""
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="价格合适买入",
            entry_price=None,
        )
        # entry_price 为 None，但有 state.price=100
        order = trader_proposal_to_order_contract(
            proposal, "600519",
            {"price": 100.0, "available_cash": 50000.0},
        )
        # 50000 / 100 = 500 股 = 5 手
        assert order.quantity == 500

    def test_invalid_ticker_still_passes(self):
        """ticker 格式不影响适配器核心逻辑。"""
        proposal = TraderProposal(
            action=TraderAction.HOLD,
            reasoning="无操作",
        )
        order = trader_proposal_to_order_contract(proposal, "000001")
        assert order.action == "HOLD"
        assert order.ticker == "000001"
