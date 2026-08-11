from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseAnthropicMessagesConfig(ABC):
    @abstractmethod
    def validate_anthropic_messages_environment(  # use different name because return type is different from base config's validate_environment
        self,
        headers: dict,
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:
        """
        OPTIONAL

        Validate the environment for the request

        Returns:
        - headers: dict
        - api_base: Optional[str] - If the provider needs to update the api_base, return it here. Otherwise, return None.
        """
        return headers, api_base

    @abstractmethod
    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    @abstractmethod
    def get_supported_anthropic_messages_params(self, model: str) -> list:
        pass

    @abstractmethod
    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        pass

    @abstractmethod
    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        pass

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:
        """
        OPTIONAL

        Sign the request, providers like Bedrock need to sign the request before sending it to the API

        For all other providers, this is a no-op and we just return the headers
        """
        return headers, None

    def should_filter_anthropic_beta_headers(self) -> bool:
        """
        Whether ``anthropic-beta`` header values should be filtered down to the
        ones the routed provider supports before the upstream request.

        Cross-provider translation paths (bedrock, vertex_ai, ...) need this so
        unsupported betas are dropped. Configs that forward natively to an
        Anthropic-compatible endpoint return False to pass betas through verbatim.
        """
        return True

    def handles_web_search_natively(self) -> bool:
        """
        Whether the upstream this config routes to executes ``web_search`` tools
        itself as part of its Anthropic Messages agentic loop.

        The web-search interception handler short-circuits web-search-only
        requests (running the search itself and returning synthetic results) only
        for providers that do NOT. Providers whose agentic loop already performs
        the search plus a follow-up synthesis step (bedrock, vertex_ai, ...)
        return True so those requests flow through untouched.
        """
        return True

    def translate_developer_role_to_system_role(
        self,
        messages: list,
    ) -> list:
        """
        Translate ``developer`` role to ``system`` role for non-OpenAI providers.
        """
        from litellm.llms.base_llm.base_utils import (
            map_developer_role_to_system_role,
        )

        return map_developer_role_to_system_role(messages=messages)

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        raise NotImplementedError("Subclasses must implement this method")

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict | httpx.Headers
    ) -> "BaseLLMException":
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        return BaseLLMException(message=error_message, status_code=status_code, headers=headers)

    def get_supported_openai_params(self, model: str) -> list:
        """OpenAI params that the Anthropic Messages wire format accepts."""
        return [
            "max_tokens",
            "stream",
            "store",
            "metadata",
            "stop",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "logprobs",
            "top_logprobs",
            "prediction",
            "response_format",
            "service_tier",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """Copy supported OpenAI params from ``non_default_params`` into
        ``optional_params``.  The base class returns the identity because
        ``get_optional_params`` already initialised ``optional_params`` with
        a small hard-coded set (stream, stop, temperature, top_p); this
        method adds the remaining ones the messages wire format accepts.

        This intentionally does **not** handle params that require
        provider-specific translation (e.g. tool definitions) — those live
        on the chat arm or are handled separately.
        """
        supported = self.get_supported_openai_params(model=model)
        for key, value in non_default_params.items():
            if key in supported:
                optional_params[key] = value
        return optional_params

    def should_fake_stream(
        self,
        model: str | None,
        stream: bool | None,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """No-op: messages arm streaming is handled by the messages handler."""
        return False

    def validate_environment(
        self,
        api_key: str | None,
        headers: dict,
        model: str,
        messages: list[Any],
        optional_params: dict,
        api_base: str | None = None,
        litellm_params: dict | None = None,
        provider_config: "BaseAnthropicMessagesConfig | None" = None,
    ) -> tuple[dict, str | None]:
        """Adapter for sync completion path — delegates to
        ``validate_anthropic_messages_environment``."""
        return self.validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params or {},
            api_key=api_key,
            api_base=api_base,
        )

    def transform_request(
        self,
        model: str,
        messages: list[dict],
        optional_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """Adapter for sync completion path — delegates to
        ``transform_anthropic_messages_request``."""
        return self.transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,  # noqa: ANN401  # pass-through adapter to transform_anthropic_messages_response
        custom_llm_provider: str,
        response_obj: dict | None = None,
        conversation_id: str | None = None,
        client_response: Any = None,  # noqa: ANN401  # optional payload for debug logging
    ) -> Any:  # noqa: ANN401  # pass-through adapter — caller determines response type
        """Adapter for sync completion path — delegates to
        ``transform_anthropic_messages_response``."""
        return self.transform_anthropic_messages_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    @property
    def max_retry_on_anthropic_messages_http_error(self) -> int:
        """
        Max HTTP attempts for /v1/messages when the handler may mutate the body and
        retry (e.g. strip invalid encrypted thinking signatures after a deployment or
        credential change).
        """
        return 2

    def should_retry_anthropic_messages_on_http_error(self, e: httpx.HTTPStatusError, litellm_params: dict) -> bool:
        """
        When True, async_anthropic_messages_handler will transform the request body
        and issue one more attempt (bounded by max_retry_on_anthropic_messages_http_error).
        """
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_signature_error,
        )

        return e.response.status_code == 400 and is_anthropic_invalid_thinking_signature_error(e.response.text)

    def transform_anthropic_messages_request_on_http_error(self, e: httpx.HTTPStatusError, request_data: dict) -> dict:
        """
        Mutates request_data in place when retrying after a recoverable HTTP error.
        """
        from litellm.llms.anthropic.common_utils import (
            is_anthropic_invalid_thinking_signature_error,
            strip_thinking_blocks_from_anthropic_messages_request_dict,
        )

        if e.response.status_code == 400 and is_anthropic_invalid_thinking_signature_error(e.response.text):
            strip_thinking_blocks_from_anthropic_messages_request_dict(request_data)
        return request_data
