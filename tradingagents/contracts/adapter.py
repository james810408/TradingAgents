"""TraderProposal → OrderContract 适配器。

将 Trader 的结构化交易提案 (TraderProposal) 转换为 A 股门控
所需的强类型订单契约 (OrderContract)，并从 state 中提取 ticker、
可用现金等信息计算 quantity。

用法:
    from tradingagents.contracts.adapter import trader_proposal_to_order_contract

    order = trader_proposal_to_order_contract(proposal, ticker="600519", state={
        "available_cash": 100000.0,
    })
"""
from __future__ import annotations

from typing import Any

from tradingagents.agents.schemas import TraderAction, TraderProposal
from tradingagents.contracts.order_contract import OrderContract


def _map_action(action: TraderAction | str) -> str:
    """Map TraderAction enum (or str) to OrderContract action literal.

    TraderAction is "Buy" / "Hold" / "Sell" (title-case enum values).
    OrderContract expects "BUY" / "HOLD" / "SELL" (upper-case literals).
    Also accepts raw strings for flexibility.
    """
    if isinstance(action, TraderAction):
        action_str = action.value
    else:
        action_str = str(action)
    return action_str.upper()


def _compute_quantity(
    proposal: TraderProposal,
    state: dict[str, Any],
) -> int:
    """从 state 和 proposal 推算买入/卖出数量（整数股）。

    优先级:
    1. state["order_quantity"] — 显式指定（用于测试/覆盖）
    2. 按可用现金 / entry_price 推算（仅 BUY，非 HOLD）
    3. 从 position 提取持有股数（仅 SELL）
    4. 默认 0（HOLD 或无法推算）
    """
    # 1. HOLD: quantity 必须为 0
    action_upper = _map_action(proposal.action)
    if action_upper == "HOLD":
        return 0

    # 2. 显式指定
    explicit = state.get("order_quantity")
    if explicit is not None:
        return int(explicit)

    if action_upper == "BUY":
        available_cash = state.get("available_cash", 0.0) or 0.0
        price = proposal.entry_price or state.get("price")
        if price and price > 0 and available_cash > 0:
            # 手数取整（A 股 100 股 = 1 手）
            qty = int(available_cash / price / 100) * 100
            return max(qty, 100)  # 至少 1 手
        # 没有价格信息时使用 state 中的 default_quantity
        default_qty = state.get("default_quantity", 0)
        return int(default_qty)

    # 3. SELL: 从 position 提取持有股数
    if action_upper == "SELL":
        position = state.get("position", {}) or {}
        held = position.get("shares", 0) or position.get("quantity", 0)
        return int(held)

    return 0


def trader_proposal_to_order_contract(
    proposal: TraderProposal,
    ticker: str,
    state: dict[str, Any] | None = None,
) -> OrderContract:
    """将 TraderProposal 转换为 OrderContract。

    Args:
        proposal: Trader 输出的结构化交易提案。
        ticker: 股票代码（6 位数字字符串，如 "600519"）。
        state: 当前 state 字典，用于提取：
            - available_cash: 可用现金（用于 BUY 数量推算）
            - price: 当前价格（用于 BUY 数量推算）
            - position: 持仓信息（用于 SELL 数量推算）
            - order_quantity: 显式指定数量（覆盖推算）
            - default_quantity: 无价格信息时的 BUY 默认数量

    Returns:
        可用于 A 股门控的 OrderContract 实例。

    Raises:
        ValueError: 如果 action 映射失败或 ticker 无效。
    """
    if state is None:
        state = {}

    # 映射 action
    action_upper = _map_action(proposal.action)
    if action_upper not in ("BUY", "SELL", "HOLD"):
        raise ValueError(
            f"无法映射 TraderAction {proposal.action!r} 到 OrderContract action"
        )

    # 推算数量
    quantity = _compute_quantity(proposal, state)

    # 从 state 提取 signal_id（如果有）
    signal_id = state.get("signal_id")

    # 构建 metadata — 保留推理和风控信息
    metadata: dict[str, Any] = {}
    if proposal.reasoning:
        metadata["reasoning"] = proposal.reasoning
    if proposal.stop_loss is not None:
        metadata["stop_loss"] = proposal.stop_loss
    if proposal.position_sizing:
        metadata["position_sizing"] = proposal.position_sizing

    return OrderContract(
        action=action_upper,
        ticker=ticker,
        quantity=quantity,
        price=proposal.entry_price,
        agent_id="trader",
        a_share=True,
        signal_id=signal_id,
        metadata=metadata,
    )
