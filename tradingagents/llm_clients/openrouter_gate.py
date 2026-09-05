"""Fork-only guard: Trade never uses OpenRouter as an LLM provider.

Sidecar module per docs/FORK_CONVENTIONS.md — kept separate from
``llm_clients/openai_client.py`` (an upstream-shared, general-purpose
provider-dispatch file) rather than spliced inline, so upstream syncs never
conflict with this Trade-specific policy.

Policy (2026-09-05, explicit user directive): Trade never uses OpenRouter as
an LLM provider, anywhere, in any project or submodule — all LLM calls should
route through Trade's own LLM adapter
(``integrations/trade_integrations/dataflows/model_adapters``) instead. See
.claude/backlog/items/2026-09-05-openrouter-ban-llm-adapter-audit.md.

Deliberately does not touch ``OPENAI_COMPATIBLE_PROVIDERS["openrouter"]`` in
``openai_client.py`` or the provider list in ``cli/utils.py``'s interactive
picker — neither of those constructs a live client by itself. This gate is
called at the one real chokepoint that does (``OpenAIClient.get_llm``), so
blocking there is sufficient without churning upstream-shaped tables/lists.
"""

from __future__ import annotations

_BANNED_PROVIDERS = frozenset({"openrouter"})


def assert_not_openrouter(provider: str | None) -> None:
    """Raise ``ValueError`` if ``provider`` resolves to OpenRouter.

    Call this at any point that is about to construct a live LLM client or
    otherwise treat ``provider`` as usable, before that construction happens.
    """
    normalized = (provider or "").strip().lower()
    if normalized in _BANNED_PROVIDERS:
        raise ValueError(
            "OpenRouter is disabled as an LLM provider in this repo (Trade "
            "policy, 2026-09-05) — configure a supported provider (e.g. "
            "minimax, nvidia, openai) instead of 'openrouter'. See "
            ".claude/backlog/items/2026-09-05-openrouter-ban-llm-adapter-audit.md."
        )
