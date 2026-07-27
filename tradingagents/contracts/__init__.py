"""TradingAgents 强类型契约。

包含 OrderContract 等用于交易流中强类型数据交换的 Pydantic 模型，
以及 TraderProposal → OrderContract 适配器。
"""
from __future__ import annotations

from tradingagents.contracts.adapter import trader_proposal_to_order_contract
from tradingagents.contracts.order_contract import OrderContract

__all__ = [
    "OrderContract",
    "trader_proposal_to_order_contract",
]
