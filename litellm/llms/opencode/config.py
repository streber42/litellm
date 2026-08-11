"""
Shared provider config for opencode surfaces.

Decides between chat-completions and Anthropic Messages arms per-model,
using the surface-specific messages-model sets.
"""

from typing import Final

from litellm.llms.opencode.chat.messages_transformation import (
    OpenCodeMessagesConfig,
    is_messages_model,
)

from .chat.transformation import OpenCodeConfig

# ----------------------------------------------------------------- go messages set
# Models the gateway serves via Anthropic Messages on the Go surface.
# Derived from models.dev ``npm == "@ai-sdk/anthropic"``.

OPENCODE_GO_MESSAGES_MODELS: Final = frozenset(
    {
        "minimax-m2.5",
        "minimax-m2.7",
        "minimax-m3",
        "qwen3.5-plus",
        "qwen3.6-plus",
        "qwen3.7-plus",
        "qwen3.8-plus",
        "qwen3.5-max",
        "qwen3.6-max",
        "qwen3.7-max",
        "qwen3.8-max",
    }
)


def get_opencode_config(surface: str, model: str):
    """Return the right config for *surface* / *model*.

    Models in the surface's messages-model set get an Anthropic Messages
    config; everything else falls through to chat-completions.
    """
    if is_messages_model(surface, model):
        return OpenCodeMessagesConfig(surface=surface)
    return OpenCodeConfig(surface=surface)
