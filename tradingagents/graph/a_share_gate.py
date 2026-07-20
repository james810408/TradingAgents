"""A 股交易规则门控

在 Trader 输出后、风险辩论前，检查交易提案是否违反 A 股规则。
如果违反，直接拒绝并返回原因。
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def check_a_share_rules(trader_proposal: dict, stock_data: dict) -> dict:
    """检查交易提案是否符合 A 股规则

    Args:
        trader_proposal: Parsed trader proposal with keys like
            action (str), ticker (str), side (str), quantity (int).
        stock_data: Current stock data with keys like
            price (float), limit_up (float), limit_down (float),
            suspended (bool), is_st (bool), delisting_period (bool).

    Returns:
        {"passed": bool, "violations": list[str], "adjusted_proposal": dict}
    """
    violations = []
    adjusted = dict(trader_proposal)

    action = trader_proposal.get("action", "").upper()
    side = trader_proposal.get("side", action)  # Buy/Sell
    ticker = trader_proposal.get("ticker", "")

    # 1. 涨跌停检查
    limit_up = stock_data.get("limit_up")
    limit_down = stock_data.get("limit_down")
    price = stock_data.get("price")

    if side in ("BUY", "Buy") and limit_up is not None and price is not None:
        if abs(price - limit_up) / (limit_up or 1) < 0.001:
            violations.append(
                f"涨停限制：{ticker} 当前价格 {price} 已涨停 (涨停价 {limit_up})，无法买入。"
            )

    if side in ("SELL", "Sell") and limit_down is not None and price is not None:
        if abs(price - limit_down) / (limit_down or 1) < 0.001:
            violations.append(
                f"跌停限制：{ticker} 当前价格 {price} 已跌停 (跌停价 {limit_down})，无法卖出。"
            )

    # 2. T+1 检查
    position = stock_data.get("position", {})
    today_bought_shares = position.get("today_bought_shares", 0)
    if side in ("SELL", "Sell") and today_bought_shares > 0:
        violations.append(
            f"T+1 限制：当日已买入 {today_bought_shares} 股，A 股不允许当日卖出。"
        )

    # 3. 停牌检查
    if stock_data.get("suspended", False):
        violations.append(
            f"停牌限制：{ticker} 当前停牌，无法交易。"
        )

    # 4. ST 股限制
    quantity = trader_proposal.get("quantity", 0) or trader_proposal.get("shares", 0)
    if stock_data.get("is_st", False) and quantity > 500000:
        violations.append(
            f"ST 股限制：ST 股票单日买入不得超过 50 万股，当前请求 {quantity} 股。"
        )
        adjusted["quantity"] = 500000
        adjusted["shares"] = 500000

    # 5. 退市整理期
    if stock_data.get("delisting_period", False):
        adjusted["manual_confirm_required"] = True
        violations.append(
            f"退市整理期：{ticker} 处于退市整理期，交易需要手动确认。标记为 NEED_MANUAL_CONFIRM。"
        )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "adjusted_proposal": adjusted,
    }


def _parse_trader_proposal(trader_plan: str) -> dict:
    """Parse a trader's investment plan text to extract action, ticker, quantity.

    Returns a dict with whatever can be parsed: action, ticker, side, quantity, shares.
    """
    proposal = {}
    lower = trader_plan.lower()

    # Try to find action
    if "buy" in lower or "long" in lower:
        proposal["action"] = "Buy"
        proposal["side"] = "Buy"
    elif "sell" in lower or "short" in lower:
        proposal["action"] = "Sell"
        proposal["side"] = "Sell"
    else:
        proposal["action"] = "Hold"
        proposal["side"] = "Hold"

    # Try to extract ticker (first uppercase ticker-like word)
    ticker_match = re.search(r'\b([A-Z]{2,6}(?:\.[A-Z]{2})?)\b', trader_plan)
    if ticker_match:
        proposal["ticker"] = ticker_match.group(1)

    # Try to extract quantity/shares
    qty_match = re.search(r'(\d[\d,]*)\s*(?:shares?|股|份)', lower)
    if qty_match:
        proposal["quantity"] = int(qty_match.group(1).replace(",", ""))
        proposal["shares"] = proposal["quantity"]

    return proposal


def _extract_stock_data_from_state(state: dict) -> dict:
    """Extract or infer stock data from the state for gate checking."""
    stock_data = {}

    # Try to get from market tools / state
    for key in ("market_report", "fundamentals_report"):
        report = state.get(key, "")
        if not report:
            continue
        # Parse price info from reports
        price_match = re.search(
            r'(?:price|收盘[价]?|current price)[:\s]*¥?(\d+\.?\d*)',
            report, re.IGNORECASE
        )
        if price_match:
            stock_data["price"] = float(price_match.group(1))

    return stock_data


def create_a_share_gate_node():
    """创建 A 股门控节点

    Placed between Trader and risk debate. Checks the trader's proposal
    against A-share trading rules and blocks violations.
    """
    def gate_node(state) -> dict:
        trader_plan = state.get("trader_investment_plan", "")
        if not trader_plan:
            return {"gate_result": {"passed": True, "violations": []}, "gate_passed": True}

        # Parse the trader proposal from text
        proposal = _parse_trader_proposal(trader_plan)

        # Extract stock data from state
        stock_data = _extract_stock_data_from_state(state)

        # Run rule checks
        result = check_a_share_rules(proposal, stock_data)

        if not result["passed"]:
            violations_text = "; ".join(result["violations"])
            logger.warning(
                "A Share Gate BLOCKED proposal for %s: %s",
                proposal.get("ticker", "?"), violations_text,
            )
            # Update state to reflect rejection
            return {
                "gate_result": result,
                "gate_passed": False,
                "final_trade_decision": "REJECTED",
                "gate_violations": result["violations"],
            }

        return {
            "gate_result": result,
            "gate_passed": True,
        }

    return gate_node
