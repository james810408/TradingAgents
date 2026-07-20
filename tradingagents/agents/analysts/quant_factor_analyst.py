"""Quant Factor Analyst: 读取 MarketMind 的因子信号，输出结构化因子报告。

与 MarketMind 桥接：通过 subprocess 调用 MarketMind 的 stockctl.py factor compute
获取 30 个 gtja191 因子的最新值。
"""

import json
import logging
import subprocess

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import QuantFactorReport, render_quant_factor_report
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)

logger = logging.getLogger(__name__)

MARKETMIND_PYTHON = "/root/TradingAgents/venv/bin/python"
MARKETMIND_STOCKCTL = "/root/market-mind/stockctl.py"
MARKETMIND_PYTHONPATH = "/root/market-mind:/root/TradingAgents"


def _fetch_factor_data(symbol: str) -> dict:
    """Call MarketMind stockctl.py factor compute to get factor signals."""
    if not symbol:
        return {}
    try:
        result = subprocess.run(
            [
                MARKETMIND_PYTHON, MARKETMIND_STOCKCTL,
                "factor", "compute",
                "--symbol", symbol,
                "--days", "60",
                "--json",
            ],
            capture_output=True, text=True, timeout=30,
            env={"PYTHONPATH": MARKETMIND_PYTHONPATH},
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        if result.stderr:
            logger.warning("MarketMind factor stderr: %s", result.stderr[:500])
    except subprocess.TimeoutExpired:
        logger.warning("MarketMind factor compute timed out for %s", symbol)
    except json.JSONDecodeError as exc:
        logger.warning("MarketMind factor output not valid JSON: %s", exc)
    except Exception as exc:
        logger.warning("MarketMind factor compute failed for %s: %s", symbol, exc)
    return {}


def create_quant_factor_analyst(llm):
    """创建量化因子分析师 Agent"""

    structured_llm = bind_structured(llm, QuantFactorReport, "Quant Factor Analyst")

    def quant_factor_analyst_node(state) -> dict:
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        symbol = state.get("company_of_interest", "")

        # 1. Fetch factor data from MarketMind
        factor_data = _fetch_factor_data(symbol)
        factor_json = json.dumps(factor_data, ensure_ascii=False, indent=2)[:4000]

        language_instruction = get_language_instruction(state)

        system_message = (
            "You are a Quant Factor Analyst specializing in A-share quantitative factor analysis. "
            "Your role is to analyse the 30 gtja191 factor signals from MarketMind and produce "
            "a structured factor report.\n\n"
            "The factor data contains values for 30+ technical/momentum/value/quality/volatility "
            "factors. Each factor has a direction (positive/negative/neutral) and z-score.\n\n"
            "Analyse:\n"
            "1. Which factor categories are aligned vs. conflicting\n"
            "2. Any extreme factor readings (|z-score| > 2.0)\n"
            "3. Overall factor crowding (what % of factors point in the same direction)\n"
            "4. Factor momentum — are the signals strengthening or weakening\n\n"
            "Output your analysis as a structured QuantFactorReport with signal, confidence, "
            "factor_summary, evidence, and risk_flags."
            + language_instruction
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants. "
                    "Do not call external tools. "
                    f"{NO_EXTERNAL_TOOLS} "
                    f"Today's date is {current_date}; treat it as 'now' for all analysis. "
                    f"{instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "human",
                    "Here is the factor data from MarketMind:\n\n"
                    "```json\n{factor_json}\n```\n\n"
                    "Analyse the factor signals and produce your QuantFactorReport.",
                ),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(factor_json=factor_json)

        report_text = invoke_structured_or_freetext(
            structured_llm, llm, prompt,
            render_quant_factor_report, "Quant Factor Analyst",
        )

        return {
            "quant_factor_report": report_text,
            "messages": [AIMessage(content=report_text)],
        }

    return quant_factor_analyst_node
