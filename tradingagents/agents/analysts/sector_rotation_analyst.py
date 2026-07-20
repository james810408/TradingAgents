"""Sector Rotation Analyst: 通过 MarketMind 获取行业板块轮动数据。

桥接方式：
- 行业板块数据: stockctl.py sector [--days N] [--symbol <symbol>]
"""

import json
import logging
import subprocess

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SectorRotationReport, render_sector_rotation_report
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


def _fetch_sector_data(days: int = 5) -> dict:
    """Call MarketMind stockctl.py sector to get sector rotation data."""
    try:
        result = subprocess.run(
            [
                MARKETMIND_PYTHON, MARKETMIND_STOCKCTL,
                "sector",
                "--days", str(days),
            ],
            capture_output=True, text=True, timeout=30,
            env={"PYTHONPATH": MARKETMIND_PYTHONPATH},
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw_text": result.stdout[:4000]}
        if result.stderr:
            logger.warning("MarketMind sector stderr: %s", result.stderr[:300])
    except Exception as exc:
        logger.warning("MarketMind sector failed: %s", exc)
    return {}


def create_sector_rotation_analyst(llm):
    """创建行业轮动分析师 Agent"""

    structured_llm = bind_structured(llm, SectorRotationReport, "Sector Rotation Analyst")

    def sector_rotation_analyst_node(state) -> dict:
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        symbol = state.get("company_of_interest", "")

        # 1. Fetch sector rotation data from MarketMind
        sector_data = _fetch_sector_data(days=5)
        sector_json = json.dumps(sector_data, ensure_ascii=False, indent=2)[:4000]

        language_instruction = get_language_instruction(state)

        system_message = (
            "You are a Sector Rotation Analyst specializing in A-share industry rotation analysis. "
            "Your role is to analyse sector/industry rotation data and determine how the "
            "instrument's industry is positioned relative to other sectors.\n\n"
            "Analyse:\n"
            "1. The instrument's industry ranking among all sectors by capital flow\n"
            "2. Top 3 and bottom 3 sectors by net capital flow\n"
            "3. Rotation direction — is capital rotating from defensive to cyclical,\n"
            "   growth to value, or vice versa?\n"
            "4. Whether the instrument's industry is gaining or losing relative position\n"
            "5. Any notable sector rotation themes (e.g. 'AI infrastructure', 'consumption recovery')\n\n"
            "Output your analysis as a structured SectorRotationReport with signal, confidence, "
            "sector_rotation_summary, evidence, and risk_flags."
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
                    f"The instrument is {symbol}. "
                    f"{instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "human",
                    "Here is the sector rotation data from MarketMind:\n\n"
                    "```json\n{sector_json}\n```\n\n"
                    "Analyse the sector rotation data and produce your SectorRotationReport.",
                ),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(sector_json=sector_json)

        report_text = invoke_structured_or_freetext(
            structured_llm, llm, prompt,
            render_sector_rotation_report, "Sector Rotation Analyst",
        )

        return {
            "sector_rotation_report": report_text,
            "messages": [AIMessage(content=report_text)],
        }

    return sector_rotation_analyst_node
