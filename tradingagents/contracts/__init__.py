"""TradingAgents 强类型契约。

包含 OrderContract 等用于交易流中强类型数据交换的 Pydantic 模型。
"""
from __future__ import annotations

from tradingagents.contracts.order_contract import OrderContract

__all__ = ["OrderContract"]
