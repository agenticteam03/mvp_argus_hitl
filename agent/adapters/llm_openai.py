"""
what: an LLMPort adapter that asks a real model, over HTTP.
why:  one adapter for every provider the project uses -- the team's own OpenAI
      key, the course gateway at ia.pezcol.dev, and a LOCAL QWEN server --
      because all three speak the same /chat/completions dialect. Which one runs
      is decided by environment variables, never by an edit to this file and
      never by a key pasted into source.

      "OpenAI" in the class name is a DIALECT, not a vendor. Ollama, vLLM,
      llama.cpp and LM Studio all serve POST /v1/chat/completions with the same
      request and reply shape, which is why pointing this adapter at a Qwen on
      Luis's server needs no new code at all -- only LLM_BASE_URL, LLM_MODEL,
      and permission to send no credential (see LLM_REQUIRE_AUTH below).
how:  POST /chat/completions with temperature 0, and hand back the assistant's
      message content UNTOUCHED.

      Untouched is the important word. This adapter does not strip prose, does
      not hunt for the first { in the reply, does not "fix" anything. If the
      model wraps its JSON in chat, the graph must see that and retry -- that is
      the failure Step 1 built the net for, and quietly repairing it here would
      hide the lesson and the bug.

      Note what is NOT here: no allowlist, no step bound, no tool execution. The
      model proposes; Python decides, elsewhere (rule 1).
"""

import os
from collections.abc import Callable
from typing import Any

DEFAULT_BASE_URL = "https://api.openai.com/v1"
# 60 s is generous for a hosted API and tight for a local one: a 7B Qwen
# answering on CPU can take minutes on a prompt this size. Overridable through
# LLM_TIMEOUT rather than edited here, because the right value depends on the
# machine the model runs on, not on this code.
DEFAULT_TIMEOUT = 60.0

# Short on purpose. Every RULE the agent enforces lives in code -- the allowlist
# in registry.py, the bound in route(), the schema in models.py. A system prompt
# is a request, and this project's whole argument is that a request is not a
# permission.
DEFAULT_SYSTEM_PROMPT = (
    "You are a container diagnostics agent. Reply with exactly one JSON object "
    "matching the schema you are given, and no other text."
)


class LLMUnavailable(RuntimeError):
    """The provider could not be reached, refused us, or answered nonsense."""


class OpenAIAdapter:
    """A real model behind LLMPort. Satisfies LLMPort."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        post_fn: Callable[[str, dict, dict], dict] | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        require_auth: bool | None = None,
        response_format: str | None = None,
    ) -> None:
        # Arguments win over environment, environment wins over the default.
        # run.py can therefore override anything without new env vars, and the
        # tests can construct an adapter with no environment at all.
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or ""
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
        cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
        self._timeout = _first_number(timeout, os.environ.get("LLM_TIMEOUT"))
        self._system_prompt = system_prompt
        # Asks the PROVIDER for JSON; it does not touch what comes back. A small
        # local model is far likelier than a hosted one to wrap its answer in
        # prose or -- on Qwen3 -- in a <think> block, and every such reply costs
        # a step. Constraining the server is configuration; editing the reply
        # afterwards would be us hiding the failure the graph is built to see.
        # Unset by default, because not every provider accepts the field.
        self._response_format = response_format or os.environ.get("LLM_RESPONSE_FORMAT") or ""
        # Sent ONLY when set. Measured: gpt-5.6-luna answers
        # 400 "temperature does not support 0 with this model; only the default
        # (1) is supported", while older models accept 0 happily. Omitting the
        # field is the one form every provider accepts, and repeatability here
        # comes from the schema and the validation, not from the sampler.
        env_temperature = os.environ.get("LLM_TEMPERATURE", "")
        if temperature is None and env_temperature:
            temperature = float(env_temperature)
        self._temperature = temperature
        self._post = post_fn if post_fn is not None else self._http_post

        # Fail at construction, not at the first call. run.py builds this object
        # before the graph starts, so a missing key stops the run before any
        # work is done rather than halfway through an investigation.
        if not self.model:
            raise LLMUnavailable("No model configured: set LLM_MODEL in .env")
        # A local model usually has no credential to give, so "no key" has to be
        # sayable -- but only OUT LOUD. The default stays True: for a paid
        # provider a missing key is a typo that should stop the run before it
        # starts, not a 401 discovered halfway through an investigation. Setting
        # LLM_REQUIRE_AUTH=false is a statement that this endpoint wants no
        # credential, not a way to silence an error.
        if require_auth is None:
            require_auth = _as_bool(os.environ.get("LLM_REQUIRE_AUTH"), default=True)
        self.require_auth = require_auth
        if require_auth and not self._api_key and not cf_id:
            raise LLMUnavailable(
                "No credentials: set OPENAI_API_KEY, or the gateway's "
                "CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET, in .env -- "
                "or, for a local model that needs none, LLM_REQUIRE_AUTH=false"
            )

        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"
        if cf_id:
            # Cloudflare Access sits in front of the course gateway and checks
            # these two headers before the request ever reaches the model.
            self._headers["CF-Access-Client-Id"] = cf_id
            self._headers["CF-Access-Client-Secret"] = cf_secret

        # Public counters, for budget watching. USD 40 for the whole project
        # means "how many calls and how many tokens" is not a detail.
        self.calls = 0
        self.tokens = 0

    def decide(self, prompt: str) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._response_format:
            # The OpenAI spelling, which Ollama and vLLM both accept:
            # {"type": "json_object"}. Sent only when configured, for the same
            # reason temperature is -- a provider that does not know the field
            # answers 400, and the whole run would die on a nicety.
            payload["response_format"] = {"type": self._response_format}
        body = self._post(f"{self.base_url}/chat/completions", self._headers, payload)
        self.calls += 1
        usage = body.get("usage") or {}
        self.tokens += int(usage.get("total_tokens") or 0)
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            # A reply we cannot even find the text in is a transport-level
            # problem, not a bad answer -- the graph's retry cannot help, so it
            # is raised rather than returned.
            raise LLMUnavailable(f"Unexpected response shape from {self.base_url}: {body}") from exc

    def describe(self) -> str:
        """Configuration as one line, with the key masked. For logs and demos."""
        if not self._api_key:
            key = "(none)"
        elif len(self._api_key) < 12:
            key = "(set, short)"
        else:
            key = f"{self._api_key[:6]}...{self._api_key[-4:]}"
        gateway = "CF-Access-Client-Id" in self._headers
        # Everything that decides where a request goes and what it looks like,
        # on one line. When a local model answers 404 or returns prose, this is
        # the line that says whether the URL, the model name or the JSON mode
        # was the thing that was wrong.
        return (
            f"model={self.model} base_url={self.base_url} api_key={key} "
            f"cf_headers={gateway} auth_required={self.require_auth} "
            f"response_format={self._response_format or '(unset)'} "
            f"timeout={self._timeout:.0f}s"
        )

    def _http_post(self, url: str, headers: dict, payload: dict) -> dict:
        import httpx

        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:  # timeouts, DNS, refused connections
            raise LLMUnavailable(f"Could not reach {url}: {exc}") from exc
        if response.status_code >= 400:
            # The body carries the reason -- wrong model name, no credit, bad
            # key -- and truncating it keeps a stack trace readable.
            raise LLMUnavailable(f"{url} returned {response.status_code}: {response.text[:300]}")
        return response.json()


def _as_bool(raw: str | None, default: bool) -> bool:
    """Read a boolean out of the environment without surprising anyone.

    Every environment variable is a string, and the string "false" is truthy in
    Python -- so a plain `if os.environ.get("X")` would read LLM_REQUIRE_AUTH=false
    as True and produce exactly the failure the flag exists to prevent. The
    accepted spellings are listed rather than guessed at, and anything else falls
    back to the default instead of being silently treated as off.
    """
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _first_number(*candidates: object) -> float:
    """The first candidate that is a usable number: argument, then env, then default.

    Separate from the constructor so a mistyped LLM_TIMEOUT ("3 minutes") falls
    through to the default instead of crashing with a ValueError that names
    float() rather than the variable the human actually got wrong.
    """
    for candidate in candidates:
        if candidate is None or candidate == "":
            continue
        try:
            return float(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return DEFAULT_TIMEOUT
