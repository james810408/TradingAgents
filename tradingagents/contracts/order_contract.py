"""B3 强类型订单契约 (OrderContract) — TradingAgents 镜像。

替代散落的 dict / tuple 形式订单字段，统一股票交易订单的强类型表示。
覆盖字段：
- action: BUY / SELL / HOLD
- ticker: 股票代码（如 600519）
- quantity: 数量（整数股）
- price: 价格（None 时市价单）
- limit_state: PENDING / FILLED / CANCELLED / PARTIAL
- agent_id: 决策 Agent（如 market_analyst / risk_manager）
- timestamp: ISO8601 时间戳
- signal_id: 关联信号 ID（可选）
- a_share: A 股规则校验（涨跌停 / T+1）
- metadata: 额外字段（dict）

Pydantic 2.13 BaseModel，subprocess JSON 双向序列化。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OrderContract(BaseModel):
    """B3 强类型订单契约。

    用法：
        from tradingagents.contracts import OrderContract
        order = OrderContract(action="BUY", ticker="600519", quantity=100, price=None)
        order_dict = order.to_subprocess_dict()
        roundtrip = OrderContract.from_subprocess_dict(order_dict)
    """

    # 必填字段
    action: Literal["BUY", "SELL", "HOLD"]
    ticker: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")
    quantity: int = Field(ge=0)
    price: Optional[float] = Field(default=None, ge=0.0)

    # 状态字段
    limit_state: Literal["PENDING", "FILLED", "CANCELLED", "PARTIAL"] = "PENDING"
    agent_id: str = "system"

    # 时间戳（自动 ISO8601 UTC）
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # 可选字段
    signal_id: Optional[str] = None
    a_share: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _ticker_must_be_six_digits(cls, v: str) -> str:
        if not re.match(r"^\d{6}$", v):
            raise ValueError(f"ticker 必须为 6 位数字字符串, 实际: {v!r}")
        return v

    @model_validator(mode="after")
    def _check_quantity_action_consistency(self):
        # HOLD 时 quantity 必须为 0
        if self.action == "HOLD" and self.quantity > 0:
            raise ValueError(f"HOLD 订单 quantity 必须为 0, 实际: {self.quantity}")
        # BUY/SELL 时 quantity 必须 > 0
        if self.action in ("BUY", "SELL") and self.quantity <= 0:
            raise ValueError(f"{self.action} 订单 quantity 必须 > 0, 实际: {self.quantity}")
        return self

    def to_subprocess_dict(self) -> dict[str, Any]:
        """序列化为 subprocess JSON 友好 dict（datetime → ISO8601 字符串）。

        Returns:
            dict: JSON 可序列化的字典
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_subprocess_dict(cls, data: dict[str, Any]) -> "OrderContract":
        """从 subprocess JSON dict 反序列化。

        Args:
            data: dict (可来自 json.loads / json.load)

        Returns:
            OrderContract 实例
        """
        # 兼容缺失字段（forward-compat）：缺字段时用默认值
        return cls.model_validate(data)

    def to_legacy_dict(self) -> dict[str, Any]:
        """兼容旧版 dict 风格（state.db 的 order_dict 字段）。

        Returns:
            dict: 旧版 dict
        """
        return {
            "action": self.action,
            "ticker": self.ticker,
            "quantity": self.quantity,
            "price": self.price,
            "limit_state": self.limit_state,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "signal_id": self.signal_id,
            "a_share": self.a_share,
            "metadata": self.metadata,
        }

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any]) -> "OrderContract":
        """从旧版 dict 构造（用于回填老数据）。

        Args:
            data: 旧版 dict

        Returns:
            OrderContract 实例
        """
        return cls.model_validate(data)
