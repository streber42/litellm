"""
OpenCode Anthropic Messages wire-format config.

Routes models in the surface's messages-model set to ``{base}/v1/messages``
with Anthropic Messages body shape.  Both Zen and Go authenticate
``/v1/messages`` with ``x-api-key`` (Anthropic default); verified live that
Bearer returns 401 "Missing API key" on both surfaces.
"""

from collections.abc import Mapping
from typing import Any, Final  # noqa: TID251  # Anthropic Messages wire format uses Any in param/return shapes

import httpx

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.opencode.common_utils import OpenCodeException
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

# ---------- surface base URL ( /v1/messages appended downstream ) ----------

ZEN_MESSAGES_BASE: Final = "https://opencode.ai/zen"

# ------------------------------------------------------------------- model set
# Models the gateway serves via Anthropic Messages wire format.
# Source: models.dev ``npm == @ai-sdk/anthropic`` classification.

OPENCODE_ZEN_MESSAGES_MODELS: Final = frozenset(
    {
        "claude-fable-5",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-4",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "qwen3.5-plus",
        "qwen3.6-plus",
    }
)

OPENCODE_GO_MESSAGES_MODELS: Final = frozenset(
    {
        "minimax-m2.5",
        "minimax-m2.7",
        "minimax-m3",
        "qwen3.5-plus",
        "qwen3.6-plus",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.8-max",
    }
)


def is_messages_model(surface: str, model: str) -> bool:
    """Return True when *model* belongs on the messages arm of *surface*.

    The per-model classification comes from the user's generator script:
    ``models.dev`` field ``npm == "@ai-sdk/anthropic"``.

    Strips the ``opencode_{surface}/`` prefix when the caller passes the
    fully-qualified model name (e.g. ``opencode_zen/claude-sonnet-4``).
    """
    # Strip opencode_{surface}/ prefix if present
    bare = model  # rebind-ok: strip prefix from qualified model name
    if "/" in bare:
        bare = bare.rsplit("/", 1)[-1]  # rebind-ok: strip prefix from qualified model name

    if surface == "zen":
        return bare in OPENCODE_ZEN_MESSAGES_MODELS

    if surface == "go":
        # minimax family is always messages on Go
        if bare in ("minimax-m2.5", "minimax-m2.7", "minimax-m3"):
            return True
        # qwen3.<num>.plus / qwen3.<num>-max  (5-8)
        if bare.startswith("qwen3."):
            suffix: Final = bare[len("qwen3.") :]  # rebind-ok: extracted substring for parsing
            parts: Final = suffix.rsplit("-", 1)  # rebind-ok: split result for parsing
            if len(parts) == 2:
                prefix: Final = parts[0]  # rebind-ok: extracted prefix label for parsing
                suffix_label: Final = parts[1]  # rebind-ok: extracted suffix label for parsing
                if prefix in ("plus",):
                    try:
                        num_plus: Final = int(suffix_label)  # rebind-ok: parsed integer for range check
                    except ValueError:
                        pass
                    else:
                        return 5 <= num_plus <= 8
                if suffix_label in ("plus", "max"):
                    try:
                        num_pfx: Final = int(prefix)  # rebind-ok: parsed integer for range check
                    except ValueError:
                        pass
                    else:
                        return 5 <= num_pfx <= 8
    return False


def _cost_map_max_output_tokens(surface: str, model: str) -> int | None:
    """Return the cost-map ``max_output_tokens`` for a messages model.

    ``model`` arrives bare (e.g. ``qwen3.7-plus``); qualify it with the
    surface prefix for the ``litellm.model_cost`` lookup.  Returns ``None``
    when the model has no cost-map entry.
    """
    import litellm

    qualified: Final = f"opencode_{surface}/{model}"
    entry: Final = litellm.model_cost.get(qualified)
    if entry is None:
        return None
    if "max_output_tokens" in entry:
        return entry["max_output_tokens"]
    if "max_tokens" in entry:
        return entry["max_tokens"]
    return None


# ------------------------------------------------------------------ config class


class OpenCodeMessagesConfig(AnthropicMessagesConfig):
    """Anthropic Messages config for the OpenCode gateway.

    Parameters
    ----------
    surface :
        ``"zen"`` (default) or ``"go"``.  Determines the base URL.
    """

    def __init__(self, surface: str = "zen") -> None:
        self.surface: Final = surface

    @property
    def custom_llm_provider(self) -> str:
        return f"opencode_{self.surface}"

    def _base_url(self) -> str:
        return ZEN_MESSAGES_BASE

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, Any],
        litellm_params: Mapping[str, Any],
        stream: bool | None = None,
    ) -> str:
        """Return ``{api_base}/v1/messages``."""
        import litellm

        base: Final = (api_base or litellm.api_base or self._base_url()).rstrip("/")
        surface_base: Final = f"{base}/go" if self.surface == "go" and not base.endswith("/go") else base
        if surface_base.endswith("/v1/messages"):
            return surface_base
        return f"{surface_base}/v1/messages"

    def validate_anthropic_messages_environment(
        self,
        headers: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
        model: str,
        messages: list[Any],  # mutable-ok: signature must match AnthropicMessagesConfig
        optional_params: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
        litellm_params: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:  # mutable-ok: signature must match AnthropicMessagesConfig
        """Resolve key / base URL and let the base class inject the auth header.

        Both surfaces authenticate ``/v1/messages`` with ``x-api-key``
        (Anthropic default); Bearer returns 401 "Missing API key" on both.
        """
        import litellm

        surface_upper: Final = self.surface.upper()

        # -- key resolution (same chain as chat arm) ---------------------------
        key: str | None = api_key  # rebind-ok: iterative fallback through key sources
        if key is None:  # rebind-ok: iterative fallback through key sources
            key = getattr(  # rebind-ok: iterative fallback through key sources
                litellm, f"opencode_{self.surface}_api_key", None
            )
        if key is None:  # rebind-ok: iterative fallback through key sources
            key = get_secret_str(  # rebind-ok: iterative fallback through key sources
                f"OPENCODE_{surface_upper}_API_KEY"
            )
        if key is None:  # rebind-ok: iterative fallback through key sources
            key = get_secret_str("OPENCODE_API_KEY")  # rebind-ok: iterative fallback through key sources

        base_url: Final = api_base or litellm.api_base or self._base_url()

        # -- auth header per surface -------------------------------------------
        # OpenCode does not support OAuth, so we can skip the OAuth check that
        # the base class performs.  Both surfaces authenticate /v1/messages with
        # x-api-key (verified live: Bearer returns 401 "Missing API key" on both
        # Zen and Go), so leave headers empty and let the base class inject
        # x-api-key.
        # NOTE: this intentionally diverges from the chat arm, which uses Bearer.

        # -- base class handles defaults, beta headers, content-type ----------
        resolved_headers, resolved_base_url = super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=key,
            api_base=base_url,
        )
        return resolved_headers, resolved_base_url

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: signature must match AnthropicMessagesConfig
    ) -> BaseLLMException:
        return OpenCodeException(message=error_message, status_code=status_code, headers=headers)

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],  # mutable-ok: signature must match AnthropicMessagesConfig
        anthropic_messages_optional_request_params: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: signature must match AnthropicMessagesConfig
    ) -> dict:  # mutable-ok: signature must match AnthropicMessagesConfig
        """Default ``max_tokens`` from the cost map before the base class runs.

        The Anthropic ``/v1/messages`` API requires ``max_tokens``, but the
        messages arm receives ``optional_params`` straight from the caller
        (e.g. a playground wildcard request with no explicit ``max_tokens``).
        The base class raises if it is absent, so default it from the model's
        cost-map ``max_output_tokens`` here.  ``model`` arrives bare (e.g.
        ``qwen3.7-plus``), so qualify it with the surface prefix for the
        ``litellm.model_cost`` lookup.
        """
        default_max_tokens: Final = (
            _cost_map_max_output_tokens(surface=self.surface, model=model)
            if anthropic_messages_optional_request_params.get("max_tokens") is None
            else None
        )
        params: Final = (
            {
                **anthropic_messages_optional_request_params,
                "max_tokens": default_max_tokens,
            }  # mutable-ok: base config requires a mutable dict
            if default_max_tokens is not None
            else anthropic_messages_optional_request_params
        )

        return super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=params,
            litellm_params=litellm_params,
            headers=headers,
        )
