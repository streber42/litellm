"""
OpenCode Zen Responses API Configuration.

Routes models in the Zen responses-model set to ``{base}/v1/responses``
with OpenAI-compatible request/response shape.  Uses ``Bearer`` auth.
"""

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

ZEN_MESSAGES_BASE: Final = "https://opencode.ai/zen"

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj: Final = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj: Final = Any  # mutable-ok: fallback placeholder for runtime


class OpenCodeZenResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for OpenCode Zen's Responses API.

    Inherits from OpenAIResponsesAPIConfig since Zen's Responses API
    is compatible with OpenAI's Responses API specification.

    Key differences from direct OpenAI:
    - Uses ``{base}/v1/responses`` as the API base (Zen gateway)
    - Uses ``OPENCODE_ZEN_API_KEY`` for authentication
    - Returns ``Bearer`` auth header (not ``x-api-key``)
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENCODE_ZEN

    def validate_environment(
        self,
        headers: MutableMapping[str, Any],  # mutable-ok: caller expects auth headers injected
        model: str,
        litellm_params: Mapping[str, Any] | None,
    ) -> MutableMapping[str, Any]:  # mutable-ok: caller expects auth headers injected
        litellm_params = litellm_params or GenericLiteLLMParams()  # rebind-ok: default to empty params
        api_key: Final = (
            litellm_params.api_key
            or litellm.opencode_zen_api_key
            or litellm.api_key
            or get_secret_str("OPENCODE_ZEN_API_KEY")
            or get_secret_str("OPENCODE_API_KEY")
        )

        if not api_key:
            raise ValueError(
                "OpenCode Zen API key is required. Set OPENCODE_ZEN_API_KEY environment variable or pass api_key parameter."
            )

        headers["Content-Type"] = "application/json"  # rebind-ok: caller expects auth header injected
        headers["Authorization"] = f"Bearer {api_key}"  # rebind-ok: caller expects auth header injected
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: Mapping[str, Any],
    ) -> str:
        url: Final = (api_base or litellm.opencode_zen_api_base or litellm.api_base or get_secret_str("OPENCODE_ZEN_BASE_URL") or ZEN_MESSAGES_BASE).rstrip("/")

        # Append /v1/responses
        if not url.endswith("/v1/responses"):
            if url.endswith("/v1"):
                url = f"{url}/responses"
            elif not url.endswith("/responses"):
                url = f"{url}/v1/responses"

        return url

    def supports_native_websocket(self) -> bool:
        """OpenCode Zen does not support native WebSocket for Responses API."""
        return False
