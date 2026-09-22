"""
what: tests for the real-model adapter, with a stub in place of HTTP.
why:  this is the one adapter that costs money, so it is also the one whose
      tests must never touch the network (rule 5). What is being checked is the
      envelope: the right URL, the right headers, the prompt in the right place,
      and the reply handed back UNCHANGED.
how:  post_fn is injected, and an autouse fixture clears the four environment
      variables first -- otherwise a developer's real .env would leak into the
      tests and change what they prove.
"""

import pytest

from agent.adapters.llm_openai import LLMUnavailable, OpenAIAdapter
from agent.ports import LLMPort

LLM_ENV = ["LLM_BASE_URL", "LLM_MODEL", "OPENAI_API_KEY", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET"]


@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch):
    """Start every test from an empty environment, whatever the machine holds."""
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)


def reply(content: str, total_tokens: int = 42) -> dict:
    """One /chat/completions response, in the shape both providers return."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"total_tokens": total_tokens},
    }


def adapter_with(captured: list, content: str = '{"ok": true}', **kwargs) -> OpenAIAdapter:
    def post(url: str, headers: dict, payload: dict) -> dict:
        captured.append({"url": url, "headers": headers, "payload": payload})
        return reply(content)

    defaults = {"model": "llm-luis", "api_key": "sk-test-key-1234567890"}
    defaults.update(kwargs)
    return OpenAIAdapter(post_fn=post, **defaults)


def test_the_adapter_satisfies_the_port():
    # Given:    an adapter with a stubbed transport
    # Expected: it passes isinstance against LLMPort
    # Why:      if it does not fit, run.py cannot swap MockLLM for it in one line
    assert isinstance(adapter_with([]), LLMPort)


def test_the_reply_is_returned_untouched():
    # Given:    a model that wraps its JSON in chat, as they do
    # Expected: decide() returns that whole string, prose included
    # Why:      cleaning it here would hide the failure the graph must see and retry
    messy = 'Sure! Here you go:\n{"action": "conclude"}'
    assert adapter_with([], content=messy).decide("prompt") == messy


def test_the_request_carries_the_model_and_the_prompt():
    # Given:    one decide() call
    # Expected: the chat-completions URL, the model, the prompt as the user message
    # Why:      the prompt reaching the model is the one thing this adapter exists to do
    captured = []
    adapter_with(captured).decide("INCIDENT RB-002 on lab-victim")
    request = captured[0]
    assert request["url"].endswith("/chat/completions")
    assert request["payload"]["model"] == "llm-luis"
    assert request["payload"]["messages"][-1] == {
        "role": "user",
        "content": "INCIDENT RB-002 on lab-victim",
    }


def test_an_api_key_becomes_a_bearer_header():
    # Given:    an OpenAI-style key
    # Expected: Authorization: Bearer <key>
    # Why:      the OpenAI path of the same adapter
    captured = []
    adapter_with(captured).decide("p")
    assert captured[0]["headers"]["Authorization"] == "Bearer sk-test-key-1234567890"


def test_gateway_credentials_become_cloudflare_headers(monkeypatch):
    # Given:    the course gateway's two CF_ACCESS_* variables and no API key
    # Expected: both CF-Access headers, and the request still goes out
    # Why:      one adapter, two providers, decided by environment alone
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "cf-id")
    monkeypatch.setenv("CF_ACCESS_CLIENT_SECRET", "cf-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://ia.pezcol.dev/v1")
    monkeypatch.setenv("LLM_MODEL", "llm-luis")
    captured = []

    def post(url, headers, payload):
        captured.append({"url": url, "headers": headers})
        return reply("{}")

    OpenAIAdapter(post_fn=post).decide("p")
    assert captured[0]["headers"]["CF-Access-Client-Id"] == "cf-id"
    assert captured[0]["headers"]["CF-Access-Client-Secret"] == "cf-secret"
    assert captured[0]["url"].startswith("https://ia.pezcol.dev/v1")


def test_no_credentials_fails_at_construction():
    # Given:    no key and no gateway headers
    # Expected: LLMUnavailable before any call is attempted
    # Why:      a missing key must stop the run at start-up, not mid-investigation
    with pytest.raises(LLMUnavailable):
        OpenAIAdapter(model="llm-luis", post_fn=lambda *args: reply("{}"))


def test_no_model_fails_at_construction():
    # Given:    credentials but no model name
    # Expected: LLMUnavailable
    # Why:      the provider would answer 404 with a confusing message instead
    with pytest.raises(LLMUnavailable):
        OpenAIAdapter(api_key="sk-x", post_fn=lambda *args: reply("{}"))


def test_an_unreadable_reply_is_an_error_not_a_guess():
    # Given:    a 200 response with no choices in it
    # Expected: LLMUnavailable
    # Why:      no text means transport trouble; the graph's retry cannot fix that
    adapter = OpenAIAdapter(
        model="llm-luis", api_key="sk-x", post_fn=lambda *args: {"error": "overloaded"}
    )
    with pytest.raises(LLMUnavailable):
        adapter.decide("p")


def test_calls_and_tokens_are_counted():
    # Given:    two calls, each reporting 42 tokens
    # Expected: calls == 2 and tokens == 84
    # Why:      USD 40 for the whole project makes "how much did that cost" a daily question
    adapter = adapter_with([])
    adapter.decide("one")
    adapter.decide("two")
    assert adapter.calls == 2
    assert adapter.tokens == 84


def test_the_key_is_masked_in_describe():
    # Given:    an adapter holding a key
    # Expected: describe() shows a prefix and suffix only
    # Why:      this line goes into terminals and screenshots
    described = adapter_with([]).describe()
    assert "sk-test-key-1234567890" not in described
    assert "sk-tes" in described


def test_temperature_is_omitted_unless_asked_for():
    # Given:    an adapter built with no temperature, then one built with 0
    # Expected: the field is absent, then present
    # Why:      gpt-5.6-luna answers 400 to any temperature but its default; older
    #           models accept 0. Omitting it is the form every provider accepts.
    captured = []
    adapter_with(captured).decide("p")
    assert "temperature" not in captured[0]["payload"]

    captured_two = []
    adapter_with(captured_two, temperature=0).decide("p")
    assert captured_two[0]["payload"]["temperature"] == 0
