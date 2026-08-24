"""
Tests for the OpenCode Zen responses arm.

These tests fail before the feature exists and only pass when the cost-map
entries, responses config class, and resolver branch are all in place.

Acceptance criteria from Issue 03:
- opencode_zen/<model> is taken over by the responses bridge
- Takes over before any chat or messages dispatch
- responses-config resolver returns Zen Responses config
- All 26 models carry "mode": "responses"
- Streaming and acompletion work on the responses arm
"""

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import respx  # noqa: F401  # required for pytest-respx fixture
from httpx import Response

import litellm
import pytest

from litellm.llms.opencode.zen.responses.transformation import (
    OpenCodeZenResponsesAPIConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ZEN_RESPONSE_ENDPOINT = "https://opencode.ai/zen/v1/responses"


def _make_responses_response(model: str, content: str, **usage_kwargs) -> dict:
    """Build a standard Responses API response body."""
    prompt = usage_kwargs.get("prompt_tokens", 1)
    completion = usage_kwargs.get("completion_tokens", 1)
    return {
        "id": "resp-123",
        "object": "response",
        "created_at": 1700000000,
        "model": model,
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}],
        "usage": {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


# ---------------------------------------------------------------------------
# Responses config class
# ---------------------------------------------------------------------------


class TestOpenCodeZenResponsesAPIConfig:
    """Tests for the OpenCodeZenResponsesAPIConfig class itself."""

    def test_custom_llm_provider(self):
        """custom_llm_provider should return OPENCODE_ZEN."""
        cfg = OpenCodeZenResponsesAPIConfig()
        assert cfg.custom_llm_provider == LlmProviders.OPENCODE_ZEN

    def test_get_complete_url_default(self):
        """Default URL should point to Zen Responses API endpoint."""
        cfg = OpenCodeZenResponsesAPIConfig()
        url = cfg.get_complete_url(api_base=None, litellm_params={})
        assert url == ZEN_RESPONSE_ENDPOINT

    def test_get_complete_url_custom_base_with_v1(self):
        """Custom api_base ending in /v1 should append /responses."""
        cfg = OpenCodeZenResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="http://localhost:4000/v1",
            litellm_params={},
        )
        assert url == "http://localhost:4000/v1/responses"

    def test_get_complete_url_custom_base_with_v1_trailing_slash(self):
        """Trailing slash on /v1 should be stripped before appending /responses."""
        cfg = OpenCodeZenResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="http://localhost:4000/v1/",
            litellm_params={},
        )
        assert url == "http://localhost:4000/v1/responses"

    def test_get_complete_url_custom_base_with_responses(self):
        """api_base already ending in /responses should pass through."""
        cfg = OpenCodeZenResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="https://my-gateway.example.com/v1/responses",
            litellm_params={},
        )
        assert url == "https://my-gateway.example.com/v1/responses"

    def test_get_complete_url_custom_base_no_suffix(self):
        """Base without /v1 should get /v1/responses appended."""
        cfg = OpenCodeZenResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="http://localhost:4000",
            litellm_params={},
        )
        assert url == "http://localhost:4000/v1/responses"

    def test_get_complete_url_custom_base_trailing_slash(self):
        """Trailing slash on bare base should be stripped."""
        cfg = OpenCodeZenResponsesAPIConfig()
        url = cfg.get_complete_url(
            api_base="http://localhost:4000/",
            litellm_params={},
        )
        assert url == "http://localhost:4000/v1/responses"

    def test_get_complete_url_env_var_base(self, monkeypatch):
        """OPENCODE_ZEN_BASE_URL env var should be used as fallback base."""
        monkeypatch.setenv("OPENCODE_ZEN_BASE_URL", "http://env-gateway.example.com")
        try:
            cfg = OpenCodeZenResponsesAPIConfig()
            url = cfg.get_complete_url(api_base=None, litellm_params={})
            assert url == "http://env-gateway.example.com/v1/responses"
        finally:
            monkeypatch.delenv("OPENCODE_ZEN_BASE_URL")

    def test_get_complete_url_module_var_base(self, monkeypatch):
        """Module-level opencode_zen_base_url should override env var."""
        monkeypatch.setenv("OPENCODE_ZEN_BASE_URL", "http://env-gateway.example.com")
        litellm.opencode_zen_api_base = "http://module-gateway.example.com"
        try:
            cfg = OpenCodeZenResponsesAPIConfig()
            url = cfg.get_complete_url(api_base=None, litellm_params={})
            assert url == "http://module-gateway.example.com/v1/responses"
        finally:
            litellm.opencode_zen_api_base = None
            monkeypatch.delenv("OPENCODE_ZEN_BASE_URL")

    def test_no_native_websocket(self):
        """OpenCode Zen does not support native WebSocket for Responses API."""
        cfg = OpenCodeZenResponsesAPIConfig()
        assert cfg.supports_native_websocket() is False


# ---------------------------------------------------------------------------
# validate_environment — Bearer header injection
# ---------------------------------------------------------------------------


class TestValidateEnvironment:
    """Tests for header injection in validate_environment."""

    def setup_method(self):
        self.cfg = OpenCodeZenResponsesAPIConfig()

    def test_bearer_header_with_explicit_key(self):
        headers: dict = {}
        from litellm.types.router import GenericLiteLLMParams

        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.5",
            litellm_params=GenericLiteLLMParams(api_key="sk-test-123"),
        )
        assert result["Authorization"] == "Bearer sk-test-123"
        assert result["Content-Type"] == "application/json"

    def test_bearer_header_from_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "sk-env-123")
        headers: dict = {}
        from litellm.types.router import GenericLiteLLMParams

        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.5",
            litellm_params=GenericLiteLLMParams(),
        )
        assert result["Authorization"] == "Bearer sk-env-123"
        monkeypatch.delenv("OPENCODE_ZEN_API_KEY")

    def test_shared_fallback_key(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-shared-456")
        headers: dict = {}
        from litellm.types.router import GenericLiteLLMParams

        result = self.cfg.validate_environment(
            headers=headers,
            model="gpt-5.5",
            litellm_params=GenericLiteLLMParams(),
        )
        assert result["Authorization"] == "Bearer sk-shared-456"
        monkeypatch.delenv("OPENCODE_API_KEY")

    def test_raises_without_any_key(self, monkeypatch):
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

        from litellm.types.router import GenericLiteLLMParams

        with pytest.raises(ValueError, match="OpenCode Zen API key is required"):
            self.cfg.validate_environment(
                headers={},
                model="gpt-5.5",
                litellm_params=GenericLiteLLMParams(),
            )


# ---------------------------------------------------------------------------
# Responses-config resolver
# ---------------------------------------------------------------------------


class TestResponsesConfigResolver:
    """Test that the responses-config resolver returns the correct config."""

    def test_provider_config_manager_returns_zen_config(self):
        """
        ProviderConfigManager.get_provider_responses_api_config should return
        OpenCodeZenResponsesAPIConfig for OPENCODE_ZEN, not None.
        """
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_ZEN,
        )
        assert config is not None, "OpenCode Zen must be registered in the responses-config resolver"
        assert isinstance(config, OpenCodeZenResponsesAPIConfig)

    def test_resolver_returns_different_config_than_openrouter(self):
        """OpenCode Zen resolver should not return OpenRouter config."""
        from litellm.llms.openrouter.responses.transformation import (
            OpenRouterResponsesAPIConfig,
        )

        zen_config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_ZEN,
        )
        router_config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENROUTER,
        )

        assert zen_config is not None
        assert router_config is not None
        assert type(zen_config) is not type(router_config)
        assert isinstance(zen_config, OpenCodeZenResponsesAPIConfig)
        assert isinstance(router_config, OpenRouterResponsesAPIConfig)

    def test_zen_config_url_is_zens(self):
        """The resolver config URL should point to Zen, not OpenRouter."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_ZEN,
        )
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == ZEN_RESPONSE_ENDPOINT

    def test_non_responses_provider_does_not_raise(self):
        """Calling get_provider_responses_api_config for a non-responses provider must not raise."""
        from litellm.types.utils import LlmProviders as LP

        # Anthropic uses chat, not responses — should fall through gracefully
        result = ProviderConfigManager.get_provider_responses_api_config(
            provider=LP.ANTHROPIC,
        )
        # Accept None or any fallback handler — just no exceptions
        assert result is None or hasattr(result, "get_complete_url")

    def test_responses_config_not_chat_config(self):
        """Verify the responses config is distinct from the chat config."""
        from litellm.llms.opencode.chat.transformation import OpenCodeConfig

        responses_cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.OPENCODE_ZEN,
        )
        assert responses_cfg is not None
        assert type(responses_cfg) is not OpenCodeConfig
        # The URL should contain /responses/, not /chat/completions/
        url = responses_cfg.get_complete_url(api_base=None, litellm_params={})
        assert "/responses" in url
        assert "/chat/completions" not in url


# ---------------------------------------------------------------------------
# Integration — mocked completion call hits /v1/responses
# ---------------------------------------------------------------------------


class TestMockedCompletion:
    """Tests using mocked HTTP transport to verify the responses bridge."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        """Clean state for every test."""
        monkeypatch.setattr(litellm, "opencode_zen_api_key", None)
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)
        monkeypatch.setattr(litellm, "opencode_api_key", None)
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
        litellm.in_memory_llm_clients_cache.flush_cache()

    def test_responses_bridge_hits_responses_endpoint(self, respx_mock):
        """opencode_zen/gpt-5.5 is routed to /v1/responses, not /v1/chat/completions."""
        respx_mock.post(ZEN_RESPONSE_ENDPOINT).mock(
            return_value=Response(200, json=_make_responses_response("gpt-5.5", "responses work"))
        )

        litellm.api_key = "sk-fake"
        litellm.disable_aiohttp_transport = True
        result = litellm.completion(
            model="opencode_zen/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is not None
        assert result.model == "gpt-5.5"
        # Responses API returns nested content
        output = result.choices[0].message.content
        assert output is not None
        assert len(respx_mock.calls) > 0
        request = respx_mock.calls[0].request
        assert "/v1/responses" in str(request.url)
        assert request.headers["Authorization"] == "Bearer sk-fake"

    def test_bearer_auth_from_module_key(self, respx_mock):
        """Module-level api_key provides the Bearer token."""
        respx_mock.post(ZEN_RESPONSE_ENDPOINT).mock(
            return_value=Response(200, json=_make_responses_response("gpt-5.4", "auth ok"))
        )

        litellm.opencode_zen_api_key = "sk-module-key"
        litellm.disable_aiohttp_transport = True

        result = litellm.completion(
            model="opencode_zen/gpt-5.4",
            messages=[{"role": "user", "content": "test"}],
        )

        assert result is not None
        auth = respx_mock.calls[0].request.headers["Authorization"]
        assert auth == "Bearer sk-module-key"
        litellm.opencode_zen_api_key = None

    def test_responses_bridge_sends_correct_body(self, respx_mock):
        """The request body should use the responses API format."""

        def capture_request(request):
            body = json.loads(request.read())
            # Responses API uses "input" not "messages" for new models
            # but the bridge converts between formats
            assert "model" in body
            return Response(200, json=_make_responses_response(body.get("model", "gpt-5.5"), "ok"))

        respx_mock.post(ZEN_RESPONSE_ENDPOINT).mock(side_effect=capture_request)

        litellm.api_key = "sk-fake"
        litellm.disable_aiohttp_transport = True

        result = litellm.completion(
            model="opencode_zen/gpt-5.4-pro",
            messages=[{"role": "user", "content": "check body shape"}],
        )

        assert result is not None

    def test_non_responses_model_does_not_use_responses_endpoint(self, respx_mock, monkeypatch):
        """A model on the chat arm (grok-4.5) should hit /v1/chat/completions, not /v1/responses."""
        chat_url = "https://opencode.ai/zen/v1/chat/completions"
        respx_mock.post(chat_url).mock(
            return_value=Response(200, json={"choices": [{"message": {"role": "assistant", "content": "chat ok"}}], "model": "grok-4.5"})
        )

        litellm.api_key = "sk-fake"
        litellm.disable_aiohttp_transport = True

        result = litellm.completion(
            model="opencode_zen/grok-4.5",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is not None
        # Verify it went to chat, not responses
        assert len(respx_mock.calls) > 0
        call_path = respx_mock.calls[0].request.url.path
        assert "/chat/completions" in call_path
        assert "/responses" not in call_path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _load_cost_map(monkeypatch):
    """Ensure the local model_cost map is loaded for every test."""
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


# ---------------------------------------------------------------------------
# Cost-map entries
# ---------------------------------------------------------------------------


class TestCostMap:
    """Cost-map entries for 26 opencode_zen responses models."""

    def _check_base_entry(self, model_key):
        """Return the cost-map entry for a model, asserting it exists."""
        assert model_key in litellm.model_cost, f"{model_key} must be in cost-map"
        entry = litellm.model_cost[model_key]
        return entry

    # --- gpt-5.6 series ---

    def test_gpt_5_6_sol(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.6-sol")
        assert entry["mode"] == "responses"
        assert entry["litellm_provider"] == "opencode_zen"
        assert entry["input_cost_per_token"] == 5e-06
        assert entry["output_cost_per_token"] == 3e-05

    def test_gpt_5_6_terra(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.6-terra")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 2e-06
        assert entry["output_cost_per_token"] == 1.2e-05

    def test_gpt_5_6_luna(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.6-luna")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 2e-07
        assert entry["output_cost_per_token"] == 1.2e-06

    # --- gpt-5.5 series ---

    def test_gpt_5_5(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.5")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 5e-06
        assert entry["output_cost_per_token"] == 3e-05

    def test_gpt_5_5_pro(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.5-pro")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 3e-05
        assert entry["output_cost_per_token"] == 0.00018

    # --- gpt-5.4 series ---

    def test_gpt_5_4(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.4")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 2.5e-06
        assert entry["output_cost_per_token"] == 1.5e-05

    def test_gpt_5_4_pro(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.4-pro")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 3e-05
        assert entry["output_cost_per_token"] == 0.00018

    def test_gpt_5_4_mini(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.4-mini")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 7.5e-07

    def test_gpt_5_4_nano(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.4-nano")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 2e-07

    # --- gpt-5.3 series ---

    def test_gpt_5_3_codex_spark(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.3-codex-spark")
        assert entry["mode"] == "responses"

    def test_gpt_5_3_codex(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.3-codex")
        assert entry["mode"] == "responses"

    # --- gpt-5.2 series ---

    def test_gpt_5_2(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.2")
        assert entry["mode"] == "responses"

    def test_gpt_5_2_codex(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.2-codex")
        assert entry["mode"] == "responses"

    # --- gpt-5.1 series ---

    def test_gpt_5_1(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.1")
        assert entry["mode"] == "responses"

    def test_gpt_5_1_codex_max(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.1-codex-max")
        assert entry["mode"] == "responses"

    def test_gpt_5_1_codex(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.1-codex")
        assert entry["mode"] == "responses"

    def test_gpt_5_1_codex_mini(self):
        entry = self._check_base_entry("opencode_zen/gpt-5.1-codex-mini")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 2.5e-07

    # --- gpt-5 base ---

    def test_gpt_5(self):
        entry = self._check_base_entry("opencode_zen/gpt-5")
        assert entry["mode"] == "responses"

    def test_gpt_5_codex(self):
        entry = self._check_base_entry("opencode_zen/gpt-5-codex")
        assert entry["mode"] == "responses"

    def test_gpt_5_nano(self):
        entry = self._check_base_entry("opencode_zen/gpt-5-nano")
        assert entry["mode"] == "responses"
        assert entry["input_cost_per_token"] == 5e-08

    # --- grok ---

    def test_grok_build_0_1(self):
        entry = self._check_base_entry("opencode_zen/grok-build-0.1")
        assert entry["mode"] == "responses"

    # --- gemini 3.6 ---

    def test_gemini_3_6_flash(self):
        entry = self._check_base_entry("opencode_zen/gemini-3.6-flash")
        assert entry["mode"] == "responses"

    # --- gemini 3.5 ---

    def test_gemini_3_5_flash_lite(self):
        entry = self._check_base_entry("opencode_zen/gemini-3.5-flash-lite")
        assert entry["mode"] == "responses"

    def test_gemini_3_5_flash(self):
        entry = self._check_base_entry("opencode_zen/gemini-3.5-flash")
        assert entry["mode"] == "responses"

    # --- gemini 3.1 ---

    def test_gemini_3_1_pro(self):
        entry = self._check_base_entry("opencode_zen/gemini-3.1-pro")
        assert entry["mode"] == "responses"

    # --- gemini 3.0 ---

    def test_gemini_3_flash(self):
        entry = self._check_base_entry("opencode_zen/gemini-3-flash")
        assert entry["mode"] == "responses"

    # --- count ---

    def test_all_26_models_have_responses_mode(self):
        """All 26 models must have mode=responses."""
        responses_models = [
            "opencode_zen/gpt-5.6-sol",
            "opencode_zen/gpt-5.6-terra",
            "opencode_zen/gpt-5.6-luna",
            "opencode_zen/gpt-5.5",
            "opencode_zen/gpt-5.5-pro",
            "opencode_zen/gpt-5.4",
            "opencode_zen/gpt-5.4-pro",
            "opencode_zen/gpt-5.4-mini",
            "opencode_zen/gpt-5.4-nano",
            "opencode_zen/gpt-5.3-codex-spark",
            "opencode_zen/gpt-5.3-codex",
            "opencode_zen/gpt-5.2",
            "opencode_zen/gpt-5.2-codex",
            "opencode_zen/gpt-5.1",
            "opencode_zen/gpt-5.1-codex-max",
            "opencode_zen/gpt-5.1-codex",
            "opencode_zen/gpt-5.1-codex-mini",
            "opencode_zen/gpt-5",
            "opencode_zen/gpt-5-codex",
            "opencode_zen/gpt-5-nano",
            "opencode_zen/grok-build-0.1",
            "opencode_zen/gemini-3.6-flash",
            "opencode_zen/gemini-3.5-flash-lite",
            "opencode_zen/gemini-3.5-flash",
            "opencode_zen/gemini-3.1-pro",
            "opencode_zen/gemini-3-flash",
        ]
        for model in responses_models:
            entry = self._check_base_entry(model)
            assert entry["mode"] == "responses", f"{model} mode must be responses, got {entry['mode']}"
            assert entry["litellm_provider"] == "opencode_zen"
        assert len(responses_models) == 26

    def test_grok_4_5_stays_on_chat_arm(self):
        """grok-4.5 must remain on chat arm, not be taken over by responses bridge."""
        entry = self._check_base_entry("opencode_zen/grok-4.5")
        assert entry["mode"] == "chat", (
            "grok-4.5 must remain on chat arm; responses arm is only for the 26 models above"
        )

    def test_chat_models_stay_on_chat(self):
        """Non-responses opencode_zen models should retain mode=chat."""
        for model in [
            "opencode_zen/glm-5",
            "opencode_zen/deepseek-v4-pro",
            "opencode_zen/minimax-m3",
        ]:
            entry = self._check_base_entry(model)
            assert entry["mode"] == "chat"


# ---------------------------------------------------------------------------
# Takeover-before-dispatch — ensure dispatch is NOT consulted
# ---------------------------------------------------------------------------


class TestTakeoverBeforeDispatch:
    """Verify the responses bridge takes over before provider dispatch."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        """Clean state for every test."""
        monkeypatch.setattr(litellm, "opencode_zen_api_key", None)
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)
        monkeypatch.setattr(litellm, "opencode_api_key", None)
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
        litellm.in_memory_llm_clients_cache.flush_cache()

    def test_responses_bridge_takes_over_before_dispatch(self, respx_mock):
        """
        opencode_zen/gpt-5.5 must hit /v1/responses and NEVER reach the
        chat-completions dispatch chain.
        """
        responses_called = respx_mock.post(ZEN_RESPONSE_ENDPOINT).mock(
            return_value=Response(200, json=_make_responses_response("gpt-5.5", "bridge"))
        )
        # Set up a guard: if the chat completions endpoint is hit, fail the test
        chat_guard = respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(500, json={"error": "DISPATCH_SHOULD_NOT_BE_REACHED"})
        )

        litellm.api_key = "sk-fake"
        litellm.disable_aiohttp_transport = True

        result = litellm.completion(
            model="opencode_zen/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is not None
        # The responses endpoint was called
        assert responses_called.called
        # The chat guard was NOT called
        assert not chat_guard.called

    def test_responses_bridge_takes_over_before_messages_dispatch(self, respx_mock):
        """
        Verify the responses bridge takes over before the messages dispatch.
        If the messages endpoint is hit, the test fails.
        """
        responses_called = respx_mock.post(ZEN_RESPONSE_ENDPOINT).mock(
            return_value=Response(200, json=_make_responses_response("gpt-5.4", "bridge"))
        )
        messages_guard = respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(500, json={"error": "MESSAGES_DISPATCH_SHOULD_NOT_BE_REACHED"})
        )

        litellm.api_key = "sk-fake"
        litellm.disable_aiohttp_transport = True

        result = litellm.completion(
            model="opencode_zen/gpt-5.4",
            messages=[{"role": "user", "content": "test"}],
        )

        assert result is not None
        assert responses_called.called
        assert not messages_guard.called

    def test_cost_map_mode_field_is_required_for_takeover(self, monkeypatch, respx_mock):
        """
        If we remove the 'mode': 'responses' from a model entry, the
        responses bridge should NOT take over and the chat endpoint would
        be hit instead. This is the regression test: if someone removes
        the mode field, the behavior changes.
        """
        # Save original cost map
        original_cost = litellm.get_model_cost_map(url="")

        # Temporarily corrupt gpt-5.5 entry: change mode to "chat"
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {**original_cost, "opencode_zen/gpt-5.5": {**original_cost["opencode_zen/gpt-5.5"], "mode": "chat"}},
        )

        # Set up both endpoints
        responses_mock = respx_mock.post(ZEN_RESPONSE_ENDPOINT).mock(
            return_value=Response(500, json={"error": "BRIDGE_SHOULD_NOT_CALL_WITH_CHAT_MODE"})
        )
        chat_mock = respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "chat fallback"}}], "model": "gpt-5.5"}
            )
        )

        litellm.api_key = "sk-fake"
        litellm.disable_aiohttp_transport = True

        _result = litellm.completion(
            model="opencode_zen/gpt-5.5",
            messages=[{"role": "user", "content": "verify mode dependency"}],
        )

        # With mode=chat, it goes to chat endpoint, not responses
        assert chat_mock.called
        assert not responses_mock.called

        # Restore original cost map
        monkeypatch.setattr(litellm, "model_cost", original_cost)
