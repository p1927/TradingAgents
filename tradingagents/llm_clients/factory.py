
from .base_client import BaseLLMClient


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

        llm.invoke = invoke  # type: ignore[method-assign]
        llm._trade_obs_wrapped = True  # type: ignore[attr-defined]
        return llm

    client.get_llm = get_llm  # type: ignore[method-assign]
    client._trade_obs_instrumented = True  # type: ignore[attr-defined]
    return client
