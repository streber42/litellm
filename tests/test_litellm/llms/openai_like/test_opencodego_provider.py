"""
Unit tests for the OpenCode Go OpenAI-like provider.
"""

import os

import pytest

from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.openai_like.dynamic_config import create_config_class


OPENCODEGO_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENCODEGO_CHAT_URL = "https://opencode.ai/zen/go/v1/chat/completions"
OPENCODEGO_MESSAGES_URL = "https://opencode.ai/zen/go/v1/messages"


def _get_config():
    provider = JSONProviderRegistry.get("opencode-go")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_opencodego_provider_registered():
    """Test that opencode-go is in the JSON provider registry"""
    provider = JSONProviderRegistry.get("opencode-go")
    assert provider is not None
    assert provider.base_url == OPENCODEGO_BASE_URL
    assert provider.api_key_env == "OPENCODE_API_KEY"
    assert provider.api_base_env == "OPENCODE_BASE_URL"
    assert provider.param_mappings.get("max_completion_tokens") == "max_tokens"


def test_opencodego_in_provider_list():
    """Test that opencode-go is in the LlmProviders enum"""
    from litellm import LlmProviders

    assert hasattr(LlmProviders, "OPENCODE_GO")
    assert LlmProviders.OPENCODE_GO.value == "opencode-go"
    assert "opencode-go" in __import__("litellm", fromlist=["provider_list"]).provider_list


def test_opencodego_in_openai_compatible_providers():
    """Test that opencode-go is in the openai_compatible_providers list"""
    from litellm.constants import openai_compatible_providers

    assert "opencode-go" in openai_compatible_providers


def test_opencodego_resolves_env_api_key(monkeypatch):
    """Test that OPENCODE_API_KEY env var is used for auth"""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key-123")
    config = _get_config()
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == OPENCODEGO_BASE_URL
    assert api_key == "test-key-123"


def test_opencodego_api_base_override():
    """Test that an explicit api_base overrides the default"""
    config = _get_config()
    api_base, api_key = config._get_openai_compatible_provider_info(
        "https://custom.opencode.ai/v1", "sk-test"
    )
    assert api_base == "https://custom.opencode.ai/v1"
    assert api_key == "sk-test"


def test_opencodego_maps_max_completion_tokens():
    """Test that max_completion_tokens is mapped to max_tokens"""
    config = _get_config()
    params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 256},
        optional_params={},
        model="opencode-go/glm-5.2",
        drop_params=False,
    )
    assert params.get("max_tokens") == 256
    assert "max_completion_tokens" not in params


def test_opencodego_chat_url_resolution():
    """Test that opencode-go model prefix resolves to provider with base URL (endpoint appended at call time)"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="opencode-go/glm-5.2",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.2"
    assert provider == "opencode-go"
    # api_base is the base URL; the /chat/completions suffix is appended during the API call
    assert api_base == OPENCODEGO_BASE_URL


def test_opencodego_messages_url_resolution():
    """Test that opencode-go models requiring Anthropic messages endpoint resolve correctly"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="opencode-go/minimax-m3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "minimax-m3"
    assert provider == "opencode-go"
    assert api_base == OPENCODEGO_BASE_URL


def test_opencodego_custom_llm_provider_workaround():
    """Test that passing custom_llm_provider=opencode-go explicitly resolves the correct base URL"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="glm-5.2",
        custom_llm_provider="opencode-go",
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.2"
    assert provider == "opencode-go"
    assert api_base == OPENCODEGO_BASE_URL


def test_opencodego_router_config():
    """Test that opencode-go can be used in Router configuration"""
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "opencode-glm",
                "litellm_params": {
                    "model": "opencode-go/glm-5.2",
                    "api_key": "test-key",
                },
            }
        ]
    )

    assert len(router.model_list) == 1
    assert router.model_list[0]["model_name"] == "opencode-glm"


def test_opencodego_chat_endpoint_selection():
    """Test that OpenAI-style models resolve to /chat/completions"""
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.llms.openai_like.dynamic_config import create_config_class, _resolve_endpoint

    provider = JSONProviderRegistry.get("opencode-go")
    config_class = create_config_class(provider)
    cfg = config_class()

    url = cfg.get_complete_url(
        api_base=None, api_key=None, model="glm-5.2",
        optional_params={}, litellm_params={},
    )
    assert url == OPENCODEGO_CHAT_URL


def test_opencodego_messages_endpoint_selection():
    """Test that Anthropic-style models resolve to /messages"""
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.llms.openai_like.dynamic_config import create_config_class

    provider = JSONProviderRegistry.get("opencode-go")
    config_class = create_config_class(provider)
    cfg = config_class()

    for model in ("minimax-m3", "qwen3.7-max", "qwen3.6-plus"):
        url = cfg.get_complete_url(
            api_base=None, api_key=None, model=model,
            optional_params={}, litellm_params={},
        )
        assert url == OPENCODEGO_MESSAGES_URL, f"{model} should use /messages"


def test_opencodego_custom_api_base_with_endpoint_respected():
    """Test that a caller-supplied api_base already ending in an endpoint is not overwritten"""
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.llms.openai_like.dynamic_config import create_config_class

    provider = JSONProviderRegistry.get("opencode-go")
    config_class = create_config_class(provider)
    cfg = config_class()

    # Explicit /messages on custom base should pass through
    url = cfg.get_complete_url(
        api_base="https://custom.opencode.ai/v1/messages",
        api_key=None, model="minimax-m3",
        optional_params={}, litellm_params={},
    )
    assert url == "https://custom.opencode.ai/v1/messages"

    # Explicit /chat/completions on custom base should pass through
    url = cfg.get_complete_url(
        api_base="https://custom.opencode.ai/v1/chat/completions",
        api_key=None, model="glm-5.2",
        optional_params={}, litellm_params={},
    )
    assert url == "https://custom.opencode.ai/v1/chat/completions"


def test_opencodego_messages_models():
    """Test that _resolve_endpoint routes known messages models"""
    from litellm.llms.openai_like.dynamic_config import _resolve_endpoint, _MESSAGES_MODELS

    # Messages models should get /messages
    for model in _MESSAGES_MODELS:
        url = _resolve_endpoint(OPENCODEGO_BASE_URL, model, "opencode-go")
        assert url == OPENCODEGO_MESSAGES_URL, f"{model}"

    # Chat models should get /chat/completions
    for model in ("glm-5.2", "grok-4.5", "kimi-k3", "deepseek-v4-flash"):
        url = _resolve_endpoint(OPENCODEGO_BASE_URL, model, "opencode-go")
        assert url == OPENCODEGO_CHAT_URL, f"{model}"

    # Non-opencode providers always get /chat/completions
    url = _resolve_endpoint(OPENCODEGO_BASE_URL, "minimax-m3", "some-other-provider")
    assert url == OPENCODEGO_CHAT_URL