
import logging

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


def create_llm_client(
    provider: str,
    model: str,
    base_url: str | None = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Provider modules are imported lazily so that simply importing this
    factory (e.g. during test collection) does not pull in heavy LLM SDKs
    or fail when their API keys are absent.

    Args:
        provider: LLM provider name
        model: Model name/identifier
        base_url: Optional base URL for API endpoint
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured BaseLLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    # Native (non-OpenAI) APIs are matched first so their string check doesn't
    # import the OpenAI client. Everything else is OpenAI-compatible and routes
    # through the provider registry (single source of truth).
    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return _instrument_llm_client(AnthropicClient(model, base_url, **kwargs))

    if provider_lower == "google":
        from .google_client import GoogleClient
        return _instrument_llm_client(GoogleClient(model, base_url, **kwargs))

    if provider_lower == "azure":
        from .azure_client import AzureOpenAIClient
        return _instrument_llm_client(AzureOpenAIClient(model, base_url, **kwargs))

    if provider_lower == "bedrock":
        from .bedrock_client import BedrockClient
        return _instrument_llm_client(BedrockClient(model, base_url, **kwargs))

    from .openai_client import OpenAIClient, is_openai_compatible
    if is_openai_compatible(provider_lower):
        return _instrument_llm_client(OpenAIClient(model, base_url, provider=provider_lower, **kwargs))

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _instrument_llm_client(client: BaseLLMClient) -> BaseLLMClient:
    """Wrap ``get_llm()`` so debate-graph invocations emit Tier 0 LLM events."""
    if getattr(client, "_trade_obs_instrumented", False):
        return client
    original_get_llm = client.get_llm

    def get_llm():
        llm = original_get_llm()
        if getattr(llm, "_trade_obs_wrapped", False):
            return llm
        provider = client.get_provider_name()
        model = getattr(client, "model", "unknown")
        original_invoke = llm.invoke

        def invoke(input, config=None, **kwargs):
            try:
                from trade_integrations.observability.hooks import llm_call_span

                with llm_call_span(provider=provider, model=model, tier="debate") as meta:
                    result = original_invoke(input, config=config, **kwargs)
                    tool_calls = getattr(result, "tool_calls", None) or []
                    meta["tool_calls"] = len(tool_calls)
                    return result
            except ImportError:
                return original_invoke(input, config=config, **kwargs)

        # LangChain chat models are pydantic models, and pydantic v2 rejects
        # assignment to an undeclared field -- `llm.invoke = invoke` raises
        # ValueError: "MinimaxChatOpenAI" object has no field "invoke".  The
        # caller in research_prefetch.py wraps the debate in a broad
        # `except Exception` and logs at WARNING, so this did not surface as a
        # crash: it silently skipped the agent's entire bootstrap debate, for
        # every provider, not just MiniMax.  `object.__setattr__` bypasses
        # pydantic's validating `__setattr__` and shadows the bound method via
        # the instance dict, which is what the original assignment intended.
        # See .claude/backlog/items/2026-09-07-minimax-adapter-missing-invoke-kills-debate-prefetch.md
        try:
            object.__setattr__(llm, "invoke", invoke)
            object.__setattr__(llm, "_trade_obs_wrapped", True)
        except Exception:  # noqa: BLE001
            # Telemetry must never cost us the debate itself.  Returning the
            # un-instrumented client loses Tier 0 LLM events for this call and
            # keeps the reasoning working, which is the right trade.
            logger.warning(
                "LLM observability instrumentation could not be attached to %s; "
                "continuing without Tier 0 LLM events for this client",
                type(llm).__name__,
                exc_info=True,
            )
            return llm
        return llm

    client.get_llm = get_llm  # type: ignore[method-assign]
    client._trade_obs_instrumented = True  # type: ignore[attr-defined]
    return client
