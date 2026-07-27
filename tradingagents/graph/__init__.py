# TradingAgents/graph/__init__.py

from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor

# Lazy import to avoid triggering chain of langgraph.checkpoint.sqlite
# when only a_share_gate or other submodules are needed.
try:
    from .trading_graph import TradingAgentsGraph
except ImportError:
    import warnings
    warnings.warn(
        "TradingAgentsGraph unavailable (missing dependency). "
        "Submodules like a_share_gate can still be imported directly.",
        ImportWarning,
        stacklevel=2,
    )
    TradingAgentsGraph = None  # type: ignore[assignment]

__all__ = [
    "TradingAgentsGraph",
    "ConditionalLogic",
    "GraphSetup",
    "Propagator",
    "Reflector",
    "SignalProcessor",
]
