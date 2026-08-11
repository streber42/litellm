"""
OpenCode Anthropic Messages wire-format config.

Routes models in the surface's messages-model set to ``{base}/v1/messages``
with Anthropic Messages body shape.  Both Zen and Go authenticate
``/v1/messages`` with ``x-api-key`` (Anthropic default); verified live that
Bearer returns 401 "Missing API key" on both surfaces.
"""

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Final  # noqa: TID251  # Anthropic Messages wire format uses Any in param/return shapes

import httpx

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
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
                        return 5 <= num_plus <= 8
                    except ValueError:
                        pass
                if suffix_label in ("plus", "max"):
                    try:
                        num_pfx: Final = int(prefix)  # rebind-ok: parsed integer for range check
                        return 5 <= num_pfx <= 8
                    except ValueError:
                        pass
    return False


def _cost_map_max_output_tokens(surface: str, model: str) -> int | None:
    """Return the cost-map ``max_output_tokens`` for a messages model.

    ``model`` arrives bare (e.g. ``qwen3.7-plus``); qualify it with the
    surface prefix for the ``litellm.model_cost`` lookup.  Returns ``None``
    when the model has no cost-map entry.
    """
    from litellm.utils import get_max_tokens

    qualified: Final = f"opencode_{surface}/{model}"
    try:
        return get_max_tokens(qualified)
    except Exception:
        return None


# ------------------------------------------------------------------ config class


class OpenCodeMessagesConfig(AnthropicMessagesConfig):
    """Anthropic Messages config for the OpenCode gateway.

    Parameters
    ----------
    surface :
        ``"zen"`` (default) or ``"go"``.  Determines the base URL.
    """

    def __init__(self, surface: str = "zen"):
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

        if self.surface == "go":
            url: Final = (api_base or litellm.api_base or self._base_url()).rstrip("/")
            if not url.endswith("/go"):
                url = f"{url}/go"
        else:
            url = (api_base or litellm.api_base or self._base_url()).rstrip("/")

        if not url.endswith("/v1/messages"):
            url = f"{url}/v1/messages"
        return url

    def validate_anthropic_messages_environment(
        self,
        headers: MutableMapping[str, Any],  # mutable-ok: caller expects auth headers injected
        model: str,
        messages: Sequence[Any],
        optional_params: Mapping[str, Any],
        litellm_params: Mapping[str, Any],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[MutableMapping[str, Any], str | None]:
        """Resolve key / base URL and let the base class inject the auth header.

        Both surfaces authenticate ``/v1/messages`` with ``x-api-key``
        (Anthropic default); Bearer returns 401 "Missing API key" on both.
        """
        import litellm

        surface_upper: Final = self.surface.upper()

        # -- key resolution (same chain as chat arm) ---------------------------
        key: str | None = api_key  # rebind-ok: iterative fallback through key sources
        if key is None:  # rebind-ok: iterative fallback through key sources
            key = getattr(
                litellm, f"opencode_{self.surface}_api_key", None
            )  # rebind-ok: iterative fallback through key sources
        if key is None:  # rebind-ok: iterative fallback through key sources
            key = get_secret_str(
                f"OPENCODE_{surface_upper}_API_KEY"
            )  # rebind-ok: iterative fallback through key sources
        if key is None:  # rebind-ok: iterative fallback through key sources
            key = get_secret_str("OPENCODE_API_KEY")  # rebind-ok: iterative fallback through key sources

        api_base_val: Final = api_base or litellm.api_base or self._base_url()

        # -- auth header per surface -------------------------------------------
        # OpenCode does not support OAuth, so we can skip the OAuth check that
        # the base class performs.  Both surfaces authenticate /v1/messages with
        # x-api-key (verified live: Bearer returns 401 "Missing API key" on both
        # Zen and Go), so leave headers empty and let the base class inject
        # x-api-key.
        # NOTE: this intentionally diverges from the chat arm, which uses Bearer.

        # -- base class handles defaults, beta headers, content-type ----------
        headers, api_base_val = (
            super().validate_anthropic_messages_environment(  # rebind-ok: base class may update headers and api_base
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=key,
                api_base=api_base_val,
            )
        )
        return headers, api_base_val

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, Any] | httpx.Headers | None = None
    ) -> Exception:
        return OpenCodeException(message=error_message, status_code=status_code, headers=headers)

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """Default ``max_tokens`` from the cost map before the base class runs.

        The Anthropic ``/v1/messages`` API requires ``max_tokens``, but the
        messages arm receives ``optional_params`` straight from the caller
        (e.g. a playground wildcard request with no explicit ``max_tokens``).
        The base class raises if it is absent, so default it from the model's
        cost-map ``max_output_tokens`` here.  ``model`` arrives bare (e.g.
        ``qwen3.7-plus``), so qualify it with the surface prefix for the
        ``litellm.model_cost`` lookup.
        """
        if anthropic_messages_optional_request_params.get("max_tokens") is None:
            default_max_tokens: Final = _cost_map_max_output_tokens(surface=self.surface, model=model)
            if default_max_tokens is not None:
                anthropic_messages_optional_request_params["max_tokens"] = default_max_tokens

        return super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
