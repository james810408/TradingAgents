"""Quant Risk: 为风险辩论员提供定量风险指标。

计算 VaR、CVaR、最大回撤、波动率、Sharpe ratio 等量化指标，
注入到 3 个风险辩论员的 prompt 中作为硬约束。
"""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """计算 VaR（Value at Risk）"""
    if len(returns) < 30:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """计算 CVaR（Conditional VaR）"""
    var = calculate_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def calculate_max_drawdown(prices: pd.Series) -> float:
    """计算最大回撤"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return float(drawdown.min())


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """计算波动率"""
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(252)
    return float(vol)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """计算 Sharpe ratio"""
    excess = returns - risk_free_rate / 252
    vol = calculate_volatility(returns, annualize=True)
    if vol == 0:
        return 0.0
    return float(excess.mean() * 252 / vol)


def calculate_position_limit(var: float, max_loss_pct: float = 0.02, portfolio_value: float = 100000) -> float:
    """基于 VaR 计算仓位上限"""
    if var >= 0:
        return 0.0  # 无风险，不限
    max_loss = portfolio_value * max_loss_pct
    position = max_loss / abs(var)
    return min(position, portfolio_value)


def generate_risk_report(returns: pd.Series, prices: pd.Series) -> dict:
    """生成完整的风险报告"""
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)
    max_dd = calculate_max_drawdown(prices)
    vol = calculate_volatility(returns)
    sharpe = calculate_sharpe_ratio(returns)
    pos_limit = calculate_position_limit(var_95)

    return {
        "var_95": round(var_95, 4),
        "cvar_95": round(cvar_95, 4),
        "max_drawdown": round(max_dd, 4),
        "annualized_volatility": round(vol, 4),
        "sharpe_ratio": round(sharpe, 2),
        "position_limit_pct": round(pos_limit / 100000 * 100, 2),
        "risk_level": _classify_risk(var_95, max_dd, vol),
    }


def _classify_risk(var, max_dd, vol) -> str:
    """风险分级"""
    if vol > 0.4 or max_dd < -0.3:
        return "HIGH"
    elif vol > 0.25 or max_dd < -0.15:
        return "MEDIUM"
    else:
        return "LOW"


def format_risk_for_prompt(risk_report: dict) -> str:
    """格式化风险报告供 LLM prompt 使用"""
    return f"""=== Quantitative Risk Metrics ===
VaR (95%): {risk_report['var_95']}
CVaR (95%): {risk_report['cvar_95']}
Max Drawdown: {risk_report['max_drawdown']}
Annualized Volatility: {risk_report['annualized_volatility']}
Sharpe Ratio: {risk_report['sharpe_ratio']}
Position Limit: {risk_report['position_limit_pct']}% of portfolio
Risk Level: {risk_report['risk_level']}
=== End Risk Metrics ==="""
