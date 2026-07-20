"""Capital Flow Analyst: 通过 MarketMind 获取主力资金和北向资金流向数据。

桥接方式：
- 主力资金: stockctl.py fund --symbol <symbol> [--days N]
- 北向资金: stockctl.py northbound [--days N]
"""

import json
import logging
import subprocess

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import CapitalFlowReport, render_capital_flow_report
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


def _fetch_main_capital(symbol: str, days: int = 10) -> dict:
    """Call MarketMind stockctl.py fund to get main capital (主力资金) flow data."""
    if not symbol:
        return {}
    try:
        result = subprocess.run(
            [
                MARKETMIND_PYTHON, MARKETMIND_STOCKCTL,
                "fund",
                "--symbol", symbol,
                "--days", str(days),
            ],
            capture_output=True, text=True, timeout=30,
            env={"PYTHONPATH": MARKETMIND_PYTHONPATH},
        )
        if result.returncode == 0 and result.stdout.strip():
            # Try JSON first, fall back to text
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw_text": result.stdout[:3000]}
        if result.stderr:
            logger.warning("MarketMind fund stderr: %s", result.stderr[:300])
    except Exception as exc:
        logger.warning("MarketMind fund failed for %s: %s", symbol, exc)
    return {}


def _fetch_northbound(days: int = 10) -> dict:
    """Call MarketMind stockctl.py northbound to get northbound (北向资金) flow data."""
    try:
        result = subprocess.run(
            [
                MARKETMIND_PYTHON, MARKETMIND_STOCKCTL,
                "northbound",
                "--days", str(days),
            ],
            capture_output=True, text=True, timeout=30,
            env={"PYTHONPATH": MARKETMIND_PYTHONPATH},
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw_text": result.stdout[:3000]}
        if result.stderr:
            logger.warning("MarketMind northbound stderr: %s", result.stderr[:300])
    except Exception as exc:
        logger.warning("MarketMind northbound failed: %s", exc)
    return {}


def create_capital_flow_analyst(llm):
    """创建资金面分析师 Agent"""

    structured_llm = bind_structured(llm, CapitalFlowReport, "Capital Flow Analyst")

    def capital_flow_analyst_node(state) -> dict:
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        symbol = state.get("company_of_interest", "")

        # 1. Fetch capital flow data from MarketMind
        main_capital = _fetch_main_capital(symbol, days=10)
        northbound = _fetch_northbound(days=10)

        main_capital_json = json.dumps(main_capital, ensure_ascii=False, indent=2)[:3000]
        northbound_json = json.dumps(northbound, ensure_ascii=False, indent=2)[:3000]

        language_instruction = get_language_instruction(state)

        system_message = (
            "You are a Capital Flow Analyst specializing in A-share capital flow analysis. "
            "Your role is to analyse main capital (主力资金) and northbound (北向资金) flows "
            "and produce a structured capital flow report.\n\n"
            "Analyse:\n"
            "1. Main capital net flow direction and magnitude (net inflow/outflow)\n"
            "2. Northbound capital net flow direction and magnitude\n"
            "3. Flow trends — is the flow accelerating, decelerating, or reversing?\n"
            "4. Alignment between main capital and northbound flows\n"
            "5. Any abnormal flow patterns (e.g. sudden reversal, extreme volume)\n\n"
            "Output your analysis as a structured CapitalFlowReport with signal, confidence, "
            "capital_flow_summary, evidence, and risk_flags."
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
                    "Here is the capital flow data from MarketMind:\n\n"
                    "**Main Capital (主力资金) Flow:**\n"
                    "```json\n{main_capital_json}\n```\n\n"
                    "**Northbound (北向资金) Flow:**\n"
                    "```json\n{northbound_json}\n```\n\n"
                    "Analyse the capital flow data and produce your CapitalFlowReport.",
                ),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(main_capital_json=main_capital_json)
        prompt = prompt.partial(northbound_json=northbound_json)

        report_text = invoke_structured_or_freetext(
            structured_llm, llm, prompt,
            render_capital_flow_report, "Capital Flow Analyst",
        )

        return {
            "capital_flow_report": report_text,
            "messages": [AIMessage(content=report_text)],
        }

    return capital_flow_analyst_node
