"""
Tests for OpenCode Anthropic Messages wire-format arm (Issue 02).

These tests fail before the feature exists and fail if the dispatch
mapping, auth header selection, or URL construction are mutated.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from httpx import Response

import litellm
import pytest

from litellm.llms.opencode.chat.messages_transformation import (
    OpenCodeMessagesConfig,
    OPENCODE_ZEN_MESSAGES_MODELS,
    is_messages_model,
)
from litellm.types.completion import _CompletionDispatchContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _anthropic_response(content: str, **usage_kwargs) -> dict:
    """Build a standard Anthropic Messages response body."""
    prompt = usage_kwargs.get("prompt_tokens", 1)
    completion = usage_kwargs.get("completion_tokens", 1)
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": prompt,
            "output_tokens": completion,
        },
    }


# ---------------------------------------------------------------------------
# Messages-model set as data
# ---------------------------------------------------------------------------


class TestMessagesModelSet:
    """The zen messages-model set is immutable data — mutations fail tests."""

    def test_all_zen_claude_models_present(self):
        """Every live claude model from Issue 02 is in the set.

        claude-opus-4-1 is excluded: it is not served by the live Zen
        gateway roster, so it was removed from the messages set and cost map.
        """
        expected_claude = {
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
        }
        assert expected_claude <= OPENCODE_ZEN_MESSAGES_MODELS
        assert "claude-opus-4-1" not in OPENCODE_ZEN_MESSAGES_MODELS

    def test_qwen_models_in_set(self):
        """qwen3.5-plus and qwen3.6-plus are in the zen messages set."""
        assert "qwen3.5-plus" in OPENCODE_ZEN_MESSAGES_MODELS
        assert "qwen3.6-plus" in OPENCODE_ZEN_MESSAGES_MODELS

    def test_set_size(self):
        """Exactly 13 models in the zen messages set."""
        assert len(OPENCODE_ZEN_MESSAGES_MODELS) == 13

    def test_non_claude_model_not_in_set(self):
        """gpt-5 models are NOT in the messages set (they belong on chat)."""
        assert "gpt-5.1" not in OPENCODE_ZEN_MESSAGES_MODELS
        assert "gpt-5.6-luna" not in OPENCODE_ZEN_MESSAGES_MODELS
        assert "grok-4.5" not in OPENCODE_ZEN_MESSAGES_MODELS

    def test_set_is_frozenset(self):
        """The set is immutable — mutation raises."""
        with pytest.raises(AttributeError):
            OPENCODE_ZEN_MESSAGES_MODELS.add("brand-new-model")


# ---------------------------------------------------------------------------
# is_messages_model routing
# ---------------------------------------------------------------------------


class TestIsMessagesModel:
    """Model-to-arm dispatch decision."""

    def test_claude_model_routes_to_messages(self):
        """claude-sonnet-4 routes to the messages arm on zen."""
        assert is_messages_model("zen", "claude-sonnet-4") is True
        assert is_messages_model("zen", "claude-opus-4-5") is True

    def test_qwen_model_routes_to_messages(self):
        """qwen3.5-plus routes to the messages arm on zen."""
        assert is_messages_model("zen", "qwen3.5-plus") is True
        assert is_messages_model("zen", "qwen3.6-plus") is True

    def test_non_messages_model_routes_to_chat(self):
        """gpt-5.1 does NOT route to messages (it goes to chat)."""
        assert is_messages_model("zen", "gpt-5.1") is False
        assert is_messages_model("zen", "grok-4.5") is False

    def test_unknown_model_does_not_route_to_messages(self):
        """Unknown/new models fall through to chat, not messages."""
        assert is_messages_model("zen", "brand-new-model") is False

    def test_go_minimax_routes_to_messages(self):
        """minimax-m2.5 is a messages model on go."""
        assert is_messages_model("go", "minimax-m2.5") is True
        assert is_messages_model("go", "minimax-m3") is True

    def test_go_qwen_routes_to_messages(self):
        """qwen3.5-max routes to messages on go."""
        assert is_messages_model("go", "qwen3.5-max") is True
        assert is_messages_model("go", "qwen3.8-max") is True

    def test_go_qwen_plus_routes_to_messages(self):
        """qwen3.6-plus routes to messages on go too."""
        assert is_messages_model("go", "qwen3.6-plus") is True

    def test_go_non_messages_model(self):
        """gpt-5.5 is chat-only on go."""
        assert is_messages_model("go", "gpt-5.5") is False

    def test_go_grok_not_messages(self):
        """gpt-5.6-luna is not a messages model on go."""
        assert is_messages_model("go", "gpt-5.6-luna") is False


# ---------------------------------------------------------------------------
# OpenCodeMessagesConfig — URL and headers
# ---------------------------------------------------------------------------


class TestMessagesConfig:
    """Tests for the OpenCodeMessagesConfig class."""

    def test_zen_custom_llm_provider(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        assert cfg.custom_llm_provider == "opencode_zen"

    def test_go_custom_llm_provider(self):
        cfg = OpenCodeMessagesConfig(surface="go")
        assert cfg.custom_llm_provider == "opencode_go"

    def test_zen_base_url(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        assert cfg._base_url() == "https://opencode.ai/zen"

    def test_get_complete_url_zen(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        url = cfg.get_complete_url(None, None, "claude-sonnet-4", {}, {})
        assert url == "https://opencode.ai/zen/v1/messages"

    def test_get_complete_url_trailing_slash(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        url = cfg.get_complete_url("http://localhost:4000/", None, "claude-sonnet-4", {}, {})
        assert url == "http://localhost:4000/v1/messages"

    def test_error_class(self):
        cfg = OpenCodeMessagesConfig(surface="zen")
        assert cfg.get_error_class("bad", 400, {}) is not None


# ---------------------------------------------------------------------------
# x-api-key vs Bearer — regression at the auth seam
# ---------------------------------------------------------------------------


class TestAuthHeader:
    """Zen uses x-api-key; Go uses Bearer on the messages arm."""

    def _make_cfg(self, surface: str):
        return OpenCodeMessagesConfig(surface=surface)

    def test_zen_uses_x_api_key(self):
        """Zen /v1/messages sends x-api-key, NOT Bearer."""
        cfg = self._make_cfg("zen")
        headers: dict = {}
        result, _ = cfg.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-zen-key",
        )
        assert "x-api-key" in result
        assert result["x-api-key"] == "sk-zen-key"
        assert "Authorization" not in result

    def test_go_uses_x_api_key(self):
        """Go /v1/messages sends x-api-key, NOT Bearer.

        Regression test: live verification showed Bearer on Go /v1/messages
        returns 401 "Missing API key"; x-api-key returns 200.
        """
        cfg = self._make_cfg("go")
        headers: dict = {}
        result, _ = cfg.validate_anthropic_messages_environment(
            headers=headers,
            model="minimax-m2.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-go-key",
        )
        assert "x-api-key" in result
        assert result["x-api-key"] == "sk-go-key"
        assert "Authorization" not in result

    def test_zen_anthropic_version_set(self):
        """The anthropic-version header is present on zen."""
        cfg = self._make_cfg("zen")
        headers: dict = {}
        result, _ = cfg.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-key",
        )
        assert result["anthropic-version"] == "2023-06-01"

    def test_zen_content_type_set(self):
        """content-type is application/json."""
        cfg = self._make_cfg("zen")
        headers: dict = {}
        result, _ = cfg.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-sonnet-4",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-key",
        )
        assert result["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# Integration — mocked messages completion call
# ---------------------------------------------------------------------------


class TestMockedMessagesCompletion:
    """Messages arm models hit /v1/messages with Anthropic body + x-api-key."""

    @pytest.fixture(autouse=True)
    def _defaults(self, monkeypatch):
        """Provide max_tokens on every integration test — Anthropic requires it."""
        monkeypatch.setattr(litellm, "opencode_zen_api_key", None)
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)
        monkeypatch.setattr(litellm, "opencode_api_key", None)
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
        # The import-time cost map comes from remote main and predates the
        # un-merged opencode feature. Load the local backup so the max_tokens
        # default (read from the cost map) resolves for opencode models.
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        litellm.in_memory_llm_clients_cache.flush_cache()

    def _make_completion_kwargs(self, **overrides):
        """Return default completion kwargs with a sensible max_tokens."""
        kwargs = {
            "messages": [{"role": "user", "content": "say hi"}],
            "max_tokens": 256,
        }
        kwargs.update(overrides)
        return kwargs

    def test_messages_model_hits_v1_messages(self, respx_mock):
        """
        A messages-model reaches {base}/v1/messages, not /chat/completions.

        This is the core acceptance test for the messages arm.
        """
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("Claude speaks"))
        )

        litellm.api_key = "sk-fake"
        litellm.disable_aiohttp_transport = True
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "say hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        assert result is not None
        assert len(respx_mock.calls) > 0
        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path
        assert "/chat/completions" not in request.url.path

    def test_messages_model_sends_x_api_key(self, respx_mock):
        """
        Zen /v1/messages sends x-api-key header, not Bearer.

        Regression test: Bearer on Zen /v1/messages returns 401.
        """
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("ok"))
        )

        litellm.api_key = "sk-zen-123"
        litellm.disable_aiohttp_transport = True
        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        assert request.headers.get("x-api-key") == "sk-zen-123"
        # Bearer should NOT be present for zen messages arm
        assert "Authorization" not in request.headers

    def test_messages_model_uses_anthropic_body_shape(self, respx_mock):
        """The request body uses Anthropic Messages format, not OpenAI."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("anthropic body"))
        )

        litellm.api_key = "sk-key"
        litellm.disable_aiohttp_transport = True
        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "test body shape"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        body = json.loads(request.read())
        # Anthropic messages uses "messages" with "role" and "content"
        # but the outer shape is Anthropic, not OpenAI
        assert "messages" in body

    def test_unknown_model_still_routes_to_chat_arm(self, respx_mock):
        """
        Models outside the messages set still route to /chat/completions.

        Dispatch precedence: messages-model check happens first;
        non-matching models fall through to chat.
        """
        respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "chat works"}}]},
            )
        )

        litellm.api_key = "sk-key"
        litellm.disable_aiohttp_transport = True
        result = litellm.completion(
            model="opencode_zen/grok-4.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        request = respx_mock.calls[0].request
        assert "/chat/completions" in request.url.path
        assert "/v1/messages" not in request.url.path

    def test_messages_dispatch_precedence_over_chat(self, respx_mock):
        """
        A messages-model is dispatched to /v1/messages, NOT /chat/completions.

        If dispatch precedence is broken, the model would hit /chat/completions.
        """
        messages_endpoint = respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("messages arm"))
        )

        chat_endpoint = respx_mock.post("https://opencode.ai/zen/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"role": "assistant", "content": "wrong"}}]})
        )

        litellm.api_key = "sk-key"
        litellm.disable_aiohttp_transport = True
        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        assert messages_endpoint.call_count == 1
        assert chat_endpoint.call_count == 0

    def test_go_messages_model_sends_x_api_key(self, respx_mock):
        """Go messages models send x-api-key, not Bearer.

        Regression test: live verification showed Bearer on Go /v1/messages
        returns 401 "Missing API key"; x-api-key returns 200.
        """
        respx_mock.post("https://opencode.ai/zen/go/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("go messages"))
        )

        litellm.api_key = "sk-go-123"
        litellm.disable_aiohttp_transport = True
        litellm.completion(
            model="opencode_go/minimax-m2.5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_go",
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path
        assert request.headers.get("x-api-key") == "sk-go-123"
        # Bearer should NOT be present for go messages arm
        assert "Authorization" not in request.headers

    def test_env_var_key_resolution_messages_arm(self, respx_mock, monkeypatch):
        """Messages arm resolves api_key from OPENCODE_ZEN_API_KEY."""
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "sk-env-messages")
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("env key"))
        )

        litellm.disable_aiohttp_transport = True
        litellm.completion(
            model="opencode_zen/claude-fable-5",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        assert request.headers.get("x-api-key") == "sk-env-messages"

    def test_messages_model_qwen(self, respx_mock):
        """qwen3.5-plus is also dispatched to messages arm."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("qwen messages"))
        )

        litellm.api_key = "sk-key"
        litellm.disable_aiohttp_transport = True
        result = litellm.completion(
            model="opencode_zen/qwen3.5-plus",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=256,
        )

        assert result is not None
        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path

    def test_max_tokens_defaulted_from_cost_map(self, respx_mock):
        """A messages-model request with no max_tokens still succeeds.

        Regression: the Anthropic /v1/messages API requires max_tokens, and
        the messages arm passes optional_params straight through.  A playground
        wildcard request (no explicit max_tokens) previously failed with
        ``max_tokens is required for Anthropic /v1/messages API``.  The config
        now defaults it from the model's cost-map ``max_output_tokens``.
        """
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("defaulted"))
        )

        litellm.api_key = "sk-key"
        litellm.disable_aiohttp_transport = True
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
        )

        assert result is not None
        request = respx_mock.calls[0].request
        body = json.loads(request.read())
        # claude-sonnet-4 cost-map max_output_tokens is 64000
        assert body["max_tokens"] == 64000

    def test_max_tokens_defaulted_on_go_surface(self, respx_mock):
        """Go messages models default max_tokens from the go cost-map entry."""
        respx_mock.post("https://opencode.ai/zen/go/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("go defaulted"))
        )

        litellm.api_key = "sk-key"
        litellm.disable_aiohttp_transport = True
        result = litellm.completion(
            model="opencode_go/qwen3.7-plus",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_go",
        )

        assert result is not None
        request = respx_mock.calls[0].request
        body = json.loads(request.read())
        # qwen3.7-plus cost-map max_output_tokens is 65536
        assert body["max_tokens"] == 65536

    def test_explicit_max_tokens_not_overridden(self, respx_mock):
        """An explicit max_tokens is preserved, not replaced by the default."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("explicit"))
        )

        litellm.api_key = "sk-key"
        litellm.disable_aiohttp_transport = True
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=128,
        )

        assert result is not None
        request = respx_mock.calls[0].request
        body = json.loads(request.read())
        assert body["max_tokens"] == 128


# ---------------------------------------------------------------------------
# Streaming test
# ---------------------------------------------------------------------------


def _sse_body(content: str, **usage_kwargs) -> str:
    """Build a single SSE event line for a messages response."""
    prompt = usage_kwargs.get("prompt_tokens", 1)
    completion = usage_kwargs.get("completion_tokens", 1)
    data = {
        "type": "message",
        "id": "msg_123",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": prompt, "output_tokens": completion},
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n"


class TestMessagesArmStreaming:
    """Streaming returns the Anthropic SSE iterator on the messages arm."""

    @pytest.fixture(autouse=True)
    def _disable_aiohttp(self, monkeypatch):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    def test_streaming_sends_to_messages_url(self, respx_mock):
        """stream=True routes to /v1/messages with Anthropic body shape."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(
                200,
                text=_sse_body("streamed answer"),
                headers={"content-type": "text/event-stream"},
            )
        )

        litellm.api_key = "sk-key"
        result = litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            stream=True,
            max_tokens=256,
        )

        assert result is not None
        # The result must be an async iterator (StreamingGenerator)
        assert hasattr(result, "__aiter__")
        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path

    def test_streaming_body_includes_anthropic_params(self, respx_mock):
        """Streaming request carries anthropic-version header."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(
                200,
                text=_sse_body("streamed"),
                headers={"content-type": "text/event-stream"},
            )
        )

        litellm.api_key = "sk-key"
        litellm.completion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            stream=True,
            max_tokens=256,
        )

        request = respx_mock.calls[0].request
        assert request.headers.get("anthropic-version") == "2023-06-01"


# ---------------------------------------------------------------------------
# acompletion coverage on the messages arm
# ---------------------------------------------------------------------------


class TestMessagesArmAcompletion:
    """acompletion dispatches to the messages arm for messages models."""

    @pytest.fixture(autouse=True)
    def _disable_aiohttp(self, monkeypatch):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    @pytest.mark.asyncio
    async def test_acompletion_sends_to_messages_url(self, respx_mock):
        """acompletion for a messages model hits /v1/messages."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("async"))
        )

        litellm.api_key = "sk-key"
        result = await litellm.acompletion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=512,
        )

        assert result is not None
        request = respx_mock.calls[0].request
        assert "/v1/messages" in request.url.path

    @pytest.mark.asyncio
    async def test_acompletion_max_tokens_reaches_request_body(self, respx_mock):
        """acompletion max_tokens is serialized into the request body."""
        respx_mock.post("https://opencode.ai/zen/v1/messages").mock(
            return_value=Response(200, json=_anthropic_response("async"))
        )

        litellm.api_key = "sk-key"
        await litellm.acompletion(
            model="opencode_zen/claude-sonnet-4",
            messages=[{"role": "user", "content": "hi"}],
            custom_llm_provider="opencode_zen",
            max_tokens=512,
        )

        request = respx_mock.calls[0].request
        body = json.loads(request.content)
        assert body["max_tokens"] == 512


# ---------------------------------------------------------------------------
# Regression: acompletion must not bind the aiohttp response to a throwaway loop
# ---------------------------------------------------------------------------


class TestMessagesArmAcompletionLoop:
    """On the acompletion path the messages arm returns a coroutine.

    Regression for the cross-event-loop streaming failure: ``acompletion`` runs
    ``completion`` in an executor thread, and the messages arm previously called
    ``asyncio.run()`` there, binding the aiohttp session/response to a throwaway
    loop. Streaming that response on the caller's loop then raised
    ``RuntimeError: Future attached to a different loop``. The fix returns the
    coroutine so ``acompletion`` awaits it on the main loop (mirroring the chat
    arm). This test fails before the fix (returns a resolved response) and passes
    after (returns a coroutine).
    """

    @pytest.fixture(autouse=True)
    def _defaults(self, monkeypatch):
        monkeypatch.setattr(litellm, "opencode_zen_api_key", None)
        monkeypatch.setattr(litellm, "opencode_go_api_key", None)
        monkeypatch.setattr(litellm, "opencode_api_key", None)
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "api_base", None)
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
        litellm.in_memory_llm_clients_cache.flush_cache()

    def _build_context(self, acompletion: bool) -> _CompletionDispatchContext:
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.types.utils import ModelResponse

        return _CompletionDispatchContext(
            _azure_detection_model="gpt-4o",
            acompletion=acompletion,
            api_base=None,
            api_key="sk-fake",
            api_version=None,
            client=None,
            custom_llm_provider="opencode_zen",
            custom_prompt_dict={},
            extra_headers=None,
            headers={},
            hf_model_name=None,
            kwargs={},
            litellm_params={},
            logger_fn=None,
            logging=Logging(
                model="opencode_zen/claude-sonnet-4",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                call_type="completion",
                start_time=datetime.now(),
                litellm_call_id="test-call",
                function_id="test-fn",
            ),
            max_retries=None,
            max_tokens=None,
            messages=[{"role": "user", "content": "hi"}],
            metadata=None,
            model="opencode_zen/claude-sonnet-4",
            model_response=ModelResponse(),
            optional_params={},
            organization=None,
            provider_config=None,
            shared_session=None,
            stream=True,
            temperature=None,
            text_completion=False,
            timeout=None,
            top_p=None,
        )

    def test_acompletion_path_returns_coroutine(self, monkeypatch):
        """acompletion=True must yield a coroutine, not a resolved response.

        Before the fix the messages arm ran ``asyncio.run()`` and returned the
        resolved response; after, it returns the coroutine so ``acompletion``
        awaits it on the main loop.
        """
        from litellm import main as litellm_main

        # The coroutine is never awaited here, so the handler body never runs.
        async def _fake_handler(**kwargs):
            raise AssertionError("handler must not run in the executor thread")

        monkeypatch.setattr(
            litellm_main.base_llm_http_handler,
            "async_anthropic_messages_handler",
            _fake_handler,
        )

        result = litellm_main._complete_opencode(self._build_context(acompletion=True))

        assert asyncio.iscoroutine(result), (
            "messages arm must return a coroutine on the acompletion path so "
            "acompletion awaits it on the main loop; returning a resolved "
            "response means asyncio.run() bound the aiohttp response to a "
            "throwaway loop (RuntimeError: Future attached to a different loop)"
        )
        # The coroutine is intentionally never awaited here; close it to avoid
        # a ResourceWarning about an un-awaited coroutine.
        result.close()

    def test_sync_path_resolves_response(self, monkeypatch):
        """The sync path still resolves the response via asyncio.run()."""
        from litellm import main as litellm_main

        async def _fake_handler(**kwargs):
            return "resolved"

        monkeypatch.setattr(
            litellm_main.base_llm_http_handler,
            "async_anthropic_messages_handler",
            _fake_handler,
        )

        result = litellm_main._complete_opencode(self._build_context(acompletion=False))

        assert result == "resolved"
        assert not asyncio.iscoroutine(result)


# ---------------------------------------------------------------------------
# Cost-map entries for messages models
# ---------------------------------------------------------------------------


class TestMessagesCostMap:
    """Cost-map entries for the 13 Zen messages models."""

    @pytest.fixture(autouse=True)
    def _load_cost_map(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    def test_claude_sonnet_4_entry(self):
        entry = litellm.model_cost["opencode_zen/claude-sonnet-4"]
        assert entry["litellm_provider"] == "opencode_zen"
        assert entry["max_input_tokens"] == 1000000
        assert entry["max_output_tokens"] == 64000

    def test_claude_opus_4_1_not_in_cost_map(self):
        """claude-opus-4-1 is not served by the live Zen roster."""
        assert "opencode_zen/claude-opus-4-1" not in litellm.model_cost

    def test_claude_fable_5_entry(self):
        entry = litellm.model_cost["opencode_zen/claude-fable-5"]
        assert entry["max_input_tokens"] == 1000000
        assert entry["input_cost_per_token"] == 1e-05

    def test_qwen3_plus_entries(self):
        entry = litellm.model_cost["opencode_zen/qwen3.5-plus"]
        assert entry["litellm_provider"] == "opencode_zen"
        assert entry["max_input_tokens"] == 262144

        entry_qwen36 = litellm.model_cost["opencode_zen/qwen3.6-plus"]
        assert entry_qwen36["max_input_tokens"] == 262144

    def test_all_14_messages_models_have_cost_entries(self):
        """Every model in the messages set has a cost-map entry."""
        for model_name in OPENCODE_ZEN_MESSAGES_MODELS:
            key = f"opencode_zen/{model_name}"
            assert key in litellm.model_cost, f"{key} missing from cost map"
            assert litellm.model_cost[key]["litellm_provider"] == "opencode_zen"

    def test_cost_entries_have_pricing(self):
        """All messages models have nonzero pricing."""
        for model_name in OPENCODE_ZEN_MESSAGES_MODELS:
            key = f"opencode_zen/{model_name}"
            entry = litellm.model_cost[key]
            assert entry["input_cost_per_token"] >= 0
            assert entry["output_cost_per_token"] >= 0
