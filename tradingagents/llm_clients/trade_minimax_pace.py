"""Trade sidecar: the debate's MiniMax requests take a slot on Trade's one account pace.

Inside the Trade monorepo the MiniMax account has one cross-process pace (Trade DECISIONS D271)
that every client spending it carries as an httpx request hook. The debate's chat models send their
own HTTP, so without this a debate burst ran unpaced against the same account the adapter queue
paces (and drew the 429s the pace exists to prevent).

Standalone TradingAgents (``trade_integrations`` not importable) and a client pointed at any other
host get ``None`` and are left exactly as they were. Kept in its own module per Trade's
docs/FORK_CONVENTIONS.md sidecar pattern.
"""

from __future__ import annotations

from typing import Any


def paced_clients(base_url: str | None) -> tuple[Any, Any] | None:
    """``(sync, async)`` httpx clients carrying Trade's pace hooks for a MiniMax ``base_url``,
    else ``None``."""
    try:
        from trade_integrations.dataflows.model_adapters.minimax_queue import paced_httpx_clients
    except ImportError:
        return None
    clients = paced_httpx_clients(None, base_url)
    return clients if clients else None
