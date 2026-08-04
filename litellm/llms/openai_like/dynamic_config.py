"""
Dynamic configuration class generator for JSON-based providers.
"""

from collections.abc import Coroutine
from typing import Any, Final, Literal, overload

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from .json_loader import SimpleProviderConfig


# Endpoint routing for providers that support multiple endpoints.
# Each provider maps models to their correct endpoint path.
# Models not listed use /chat/completions by default.
# Source: https://opencode.ai/docs/go/#endpoints and
#         https://opencode.ai/docs/zen/#endpoints
_OPENCODE_ENDPOINTS: tuple[tuple[str, tuple[tuple[str, frozenset[str]], ...]], ...] = (
    (
        "opencode-go",
        (
            (
                "messages",
                frozenset(
                    (
                        "minimax-m3",
                        "minimax-m2.7",
                        "minimax-m2.5",
                        "qwen3.8-max",
                        "qwen3.7-max",
                        "qwen3.7-plus",
                        "qwen3.6-plus",
                    )
                ),
            ),
            ("responses", frozenset(("gpt-5.6-luna",))),
        ),
    ),
    (
        "opencode-zen",
        (
            (
                "messages",
                frozenset(
                    (
                        "claude-fable-5",
                        "claude-opus-5",
                        "claude-opus-4-8",
                        "claude-opus-4-7",
                        "claude-opus-4-6",
                        "claude-opus-4-5",
                        "claude-opus-4-1",
                        "claude-sonnet-5",
                        "claude-sonnet-4-6",
                        "claude-sonnet-4-5",
                        "claude-sonnet-4",
                        "claude-haiku-4-5",
                        "qwen3.7-max",
                        "qwen3.7-plus",
                        "qwen3.6-plus",
                        "qwen3.5-plus",
                    )
                ),
            ),
            (
                "responses",
                frozenset(
                    (
                        "gemini-3.6-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3.5-flash",
                        "gemini-3.1-pro",
                        "gemini-3-flash",
                        "gpt-5.6-sol",
                        "gpt-5.6-terra",
                        "gpt-5.6-luna",
                        "gpt-5.5",
                        "gpt-5.5-pro",
                        "gpt-5.4",
                        "gpt-5.4-pro",
                        "gpt-5.4-mini",
                        "gpt-5.4-nano",
                        "gpt-5.3-codex-spark",
                        "gpt-5.3-codex",
                        "gpt-5.2",
                        "gpt-5.2-codex",
                        "gpt-5.1",
                        "gpt-5.1-codex-max",
                        "gpt-5.1-codex",
                        "gpt-5.1-codex-mini",
                        "gpt-5",
                        "gpt-5-codex",
                        "gpt-5-nano",
                        "grok-build-0.1",
                        "grok-4.5",
                    )
                ),
            ),
        ),
    ),
)


def _resolve_endpoint(api_base: str | None, model: str, provider_slug: str) -> str:
    """Select the correct endpoint suffix based on model name.

    Some providers serve different model families on different endpoint
    paths — chat completions for OpenAI-style models, Anthropic /messages
    for others, and OpenAI /responses for yet others.
    """
    if api_base is None:
        return ""

    clean = api_base.rstrip("/")

    # If caller already specified an endpoint, honour it
    if any(clean.endswith(suffix) for suffix in ("/chat/completions", "/messages", "/responses")):
        return clean

    # Look up provider-specific routing
    for slug, endpoints in _OPENCODE_ENDPOINTS:
        if slug != provider_slug:
            continue
        for endpoint, models in endpoints:
            if model in models:
                return f"{clean}/{endpoint}"

    return f"{clean}/chat/completions"


def create_config_class(provider: SimpleProviderConfig):
    """Generate config class dynamically from JSON configuration"""

    # Choose base class
    base_class: Final[type] = OpenAIGPTConfig if provider.base_class == "openai_gpt" else OpenAILikeChatConfig

    class JSONProviderConfig(base_class):  # type: ignore[valid-type,misc]
        @overload
        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
        ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...

        @overload
        def _transform_messages(
            self,
            messages: list[AllMessageValues],
            model: str,
            is_async: Literal[False] = False,
        ) -> list[AllMessageValues]: ...

        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: bool = False
        ) -> list[AllMessageValues] | Coroutine[Any, Any, list[AllMessageValues]]:
            """Transform messages based on special_handling config"""

            # Handle content list to string conversion if configured
            if provider.special_handling.get("convert_content_list_to_string"):
                messages = handle_messages_with_content_list_to_str_conversion(messages)

            if is_async:
                return super()._transform_messages(messages=messages, model=model, is_async=True)
            else:
                return super()._transform_messages(messages=messages, model=model, is_async=False)

        def _get_openai_compatible_provider_info(
            self, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            """Get API base and key from JSON config"""

            # Resolve base URL
            resolved_base = api_base
            if not resolved_base and provider.api_base_env:
                resolved_base = get_secret_str(provider.api_base_env)
            if not resolved_base:
                resolved_base = provider.base_url

            # Resolve API key
            resolved_key: Final = api_key or get_secret_str(provider.api_key_env)

            return resolved_base, resolved_key

        def get_complete_url(
            self,
            api_base: str | None,
            api_key: str | None,
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: bool | None = None,
        ) -> str:
            """Build complete URL for the API endpoint"""
            if not api_base:
                api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            return _resolve_endpoint(api_base, model, provider.slug)

        def get_supported_openai_params(self, model: str) -> list:
            """Get supported OpenAI params, excluding tool-related params for models
            that don't support function calling."""
            from litellm.utils import supports_function_calling, supports_reasoning

            supported_params: Final = super().get_supported_openai_params(model=model)

            _supports_fc: Final = supports_function_calling(model=model, custom_llm_provider=provider.slug)

            if not _supports_fc:
                tool_params: Final = [
                    "tools",
                    "tool_choice",
                    "function_call",
                    "functions",
                    "parallel_tool_calls",
                ]
                for param in tool_params:
                    if param in supported_params:
                        supported_params.remove(param)
                verbose_logger.debug(
                    "Model %s on provider %s does not support function calling — removed tool-related params from supported params.",
                    model,
                    provider.slug,
                )

            _supports_reasoning: Final = supports_reasoning(model=model, custom_llm_provider=provider.slug)
            if _supports_reasoning and "reasoning_effort" not in supported_params:
                supported_params.append("reasoning_effort")

            return supported_params

        def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            """Apply parameter mappings and constraints"""

            supported_params: Final = self.get_supported_openai_params(model)

            # Apply supported params
            for param, value in non_default_params.items():
                # Check parameter mappings first
                if param in provider.param_mappings:
                    optional_params[provider.param_mappings[param]] = value
                elif param in supported_params:
                    optional_params[param] = value

            # Apply temperature constraints if present
            if "temperature" in optional_params:
                temp = optional_params["temperature"]
                constraints: Final = provider.constraints

                # Clamp to max
                if "temperature_max" in constraints:
                    temp = min(temp, constraints["temperature_max"])

                # Clamp to min
                if "temperature_min" in constraints:
                    temp = max(temp, constraints["temperature_min"])

                # Special case: temperature_min_with_n_gt_1
                if "temperature_min_with_n_gt_1" in constraints:
                    n: Final = optional_params.get("n", 1)
                    if n > 1 and temp < constraints["temperature_min_with_n_gt_1"]:
                        temp = constraints["temperature_min_with_n_gt_1"]

                optional_params["temperature"] = temp

            return optional_params

        @property
        def custom_llm_provider(self) -> str | None:
            return provider.slug

    return JSONProviderConfig


_responses_config_cache: Final[dict] = {}


def create_responses_config_class(provider: SimpleProviderConfig):
    """Generate a Responses API config class dynamically from JSON configuration.

    Parallel to create_config_class() but for /v1/responses endpoints.
    Classes are cached per provider slug to avoid regeneration on every request.
    """
    if provider.slug in _responses_config_cache:
        return _responses_config_cache[provider.slug]

    from litellm.llms.openai_like.responses.transformation import (
        OpenAILikeResponsesConfig,
    )
    from litellm.types.llms.openai import ResponseInputParam
    from litellm.types.router import GenericLiteLLMParams

    class JSONProviderResponsesConfig(OpenAILikeResponsesConfig):
        @property
        def custom_llm_provider(self):  # type: ignore[override]
            return provider.slug

        def validate_environment(
            self,
            headers: dict,
            model: str,
            litellm_params: GenericLiteLLMParams | None,
        ) -> dict:
            litellm_params = litellm_params or GenericLiteLLMParams()
            api_key: Final = litellm_params.api_key or get_secret_str(provider.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            return headers

        def get_complete_url(
            self,
            api_base: str | None,
            litellm_params: dict,
        ) -> str:
            if not api_base:
                if provider.api_base_env:
                    api_base = get_secret_str(provider.api_base_env)
                if not api_base:
                    api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            api_base = api_base.rstrip("/")
            return f"{api_base}/responses"

        def transform_responses_api_request(
            self,
            model: str,
            input: str | ResponseInputParam,
            response_api_optional_request_params: dict,
            litellm_params: GenericLiteLLMParams,
            headers: dict,
        ) -> dict:
            if provider.special_handling.get("force_store_false"):
                response_api_optional_request_params["store"] = False
            return super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

    _responses_config_cache[provider.slug] = JSONProviderResponsesConfig
    return JSONProviderResponsesConfig
