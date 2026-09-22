"""
what: tests for everything that had to become configurable before this project
      could run on a machine none of us is sitting in front of -- the model
      provider, the credential requirement, the Prometheus address, and the
      three files Stage 4 and Stage 5 need to outlive a container.
why:  every one of these was a literal that happened to be correct on a Windows
      laptop and is wrong inside a container. "localhost" means the container
      itself; a relative path is destroyed when `run --rm` finishes; a local
      Qwen has no API key to give. Each fails SILENTLY rather than loudly, which
      is exactly the kind of failure that arrives during a demo.

      The checkpoint path is the sharpest of them. The human gate works by
      saving the run and exiting, so if that file does not survive the
      container, the pause is gone before anyone can answer it -- and the
      symptom is the cheerful line "Nothing is waiting for an answer".
how:  monkeypatch the environment and assert what the code does with it. No
      network, no Docker, no model (rule 5): everything here is construction,
      payload shape, or a file in tmp_path.
"""

import pytest

from agent import audit
from agent.adapters.llm_mock import MockLLM
from agent.adapters.llm_openai import LLMUnavailable, OpenAIAdapter, _as_bool, _first_number
from agent.adapters.memory_sqlite import SqliteMemory
from agent.adapters.metrics_prometheus import DEFAULT_URL, PrometheusMetrics
from agent.run import build_llm

# Every variable any of this reads. Cleared before each test so a developer's
# own .env or shell cannot change what these tests prove -- the same trick
# test_llm_openai.py uses, and for the same reason.
DEPLOY_ENV = [
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_REQUIRE_AUTH",
    "LLM_RESPONSE_FORMAT",
    "LLM_TIMEOUT",
    "OPENAI_API_KEY",
    "CF_ACCESS_CLIENT_ID",
    "CF_ACCESS_CLIENT_SECRET",
    "PROMETHEUS_URL",
    "LAB_AGENT_DB",
    "LAB_AGENT_AUDIT",
    "LAB_AGENT_CHECKPOINTS",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Start every test from an empty environment, whatever the machine holds."""
    for name in DEPLOY_ENV:
        monkeypatch.delenv(name, raising=False)


def reply(content: str = '{"ok": true}') -> dict:
    return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 1}}


def capture() -> tuple[list, object]:
    """A post_fn that records what it was asked to send."""
    sent: list[dict] = []

    def post(url: str, headers: dict, payload: dict) -> dict:
        sent.append({"url": url, "headers": headers, "payload": payload})
        return reply()

    return sent, post


# ---------------------------------------------------------------------------
# 1. A local model has no credential to give
# ---------------------------------------------------------------------------


def test_a_missing_key_still_stops_the_run_by_default():
    # Given:    no key, no gateway headers, and LLM_REQUIRE_AUTH unset
    # Expected: LLMUnavailable at construction, exactly as before
    # Why:      for a PAID provider a missing key is a typo, and it must fail at
    #           start-up rather than 401 halfway through an investigation
    with pytest.raises(LLMUnavailable):
        OpenAIAdapter(model="qwen2.5", post_fn=lambda *a: reply())


def test_require_auth_false_lets_a_local_model_answer_with_no_key(monkeypatch):
    # Given:    LLM_REQUIRE_AUTH=false, a base URL on the LAN, and no credential
    # Expected: the adapter constructs and sends, with no Authorization header
    # Why:      this is the whole Qwen path -- Ollama and vLLM want no key, and
    #           without this flag the adapter refuses to start at all
    monkeypatch.setenv("LLM_REQUIRE_AUTH", "false")
    monkeypatch.setenv("LLM_BASE_URL", "http://10.0.0.36:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b-instruct")
    sent, post = capture()

    OpenAIAdapter(post_fn=post).decide("prompt")

    assert sent[0]["url"] == "http://10.0.0.36:11434/v1/chat/completions"
    assert "Authorization" not in sent[0]["headers"]


def test_the_string_false_is_not_true():
    # Given:    the spellings an .env file realistically contains
    # Expected: "false" and "0" are False; "true" is True; empty falls back
    # Why:      every env var is a STRING, and bool("false") is True in Python.
    #           A plain `if os.environ.get(...)` would read LLM_REQUIRE_AUTH=false
    #           as "yes, require auth" -- the exact failure the flag prevents
    assert _as_bool("false", default=True) is False
    assert _as_bool("0", default=True) is False
    assert _as_bool("true", default=False) is True
    assert _as_bool("", default=True) is True


# ---------------------------------------------------------------------------
# 2. Asking a small local model for JSON
# ---------------------------------------------------------------------------


def test_response_format_is_omitted_unless_configured(monkeypatch):
    # Given:    an adapter with LLM_RESPONSE_FORMAT unset
    # Expected: no response_format field in the payload
    # Why:      a provider that does not know the field answers 400, and the run
    #           would die on a nicety rather than on anything real
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    sent, post = capture()

    OpenAIAdapter(post_fn=post).decide("p")

    assert "response_format" not in sent[0]["payload"]


def test_response_format_is_sent_when_configured(monkeypatch):
    # Given:    LLM_RESPONSE_FORMAT=json_object, as .env.example suggests for Qwen
    # Expected: payload carries {"type": "json_object"}
    # Why:      constraining the SERVER is configuration; a local model that
    #           wraps its answer in prose costs a step, and there are only five
    monkeypatch.setenv("LLM_REQUIRE_AUTH", "false")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "json_object")
    sent, post = capture()

    OpenAIAdapter(post_fn=post).decide("p")

    assert sent[0]["payload"]["response_format"] == {"type": "json_object"}


def test_the_reply_is_still_returned_untouched(monkeypatch):
    # Given:    json_object requested, and a Qwen3 that emits <think> anyway
    # Expected: decide() returns the whole string, think-block included
    # Why:      asking the provider for JSON must not become us CLEANING the
    #           answer. If the model disobeys, the graph has to see it, record it
    #           and retry -- that is what the ValidationError branch is for
    monkeypatch.setenv("LLM_REQUIRE_AUTH", "false")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "json_object")
    messy = '<think>the container died</think>\n{"action": "conclude"}'

    assert OpenAIAdapter(post_fn=lambda *a: reply(messy)).decide("p") == messy


# ---------------------------------------------------------------------------
# 3. Timeouts, because a local model is slow
# ---------------------------------------------------------------------------


def test_the_timeout_can_be_raised_from_the_environment(monkeypatch):
    # Given:    LLM_TIMEOUT=180
    # Expected: the adapter reports 180 s
    # Why:      a 7B Qwen on one vCPU can take minutes; 60 s would turn a slow
    #           answer into a transport error and end the run
    monkeypatch.setenv("LLM_REQUIRE_AUTH", "false")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5")
    monkeypatch.setenv("LLM_TIMEOUT", "180")

    assert "timeout=180s" in OpenAIAdapter(post_fn=lambda *a: reply()).describe()


def test_a_mistyped_timeout_falls_back_instead_of_crashing():
    # Given:    "3 minutes" where a number was expected
    # Expected: the 60 s default, not a ValueError
    # Why:      a typo in .env should not crash the composition root with a
    #           message that names float() rather than the variable
    assert _first_number("3 minutes", None) == 60.0


# ---------------------------------------------------------------------------
# 4. Where the agent's own infrastructure lives
# ---------------------------------------------------------------------------


def test_prometheus_url_comes_from_the_environment(monkeypatch):
    # Given:    PROMETHEUS_URL as docker-compose.yml sets it
    # Expected: the adapter queries the service name, not localhost
    # Why:      inside a container localhost IS the container -- the agent would
    #           get connection refused and blame Prometheus
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus:9090")

    # _url, not a public accessor: there is none, and inventing one just to be
    # asserted on would be test-shaped design. The address is what matters.
    assert PrometheusMetrics(query_fn=lambda q: [])._url == "http://prometheus:9090"


def test_an_explicit_url_still_beats_the_environment(monkeypatch):
    # Given:    both an argument and an environment variable
    # Expected: the argument wins
    # Why:      argument > environment > default is the order every adapter here
    #           uses; a test pinning one address must not be quietly redirected
    monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus:9090")

    assert PrometheusMetrics(url="http://localhost:9091")._url == "http://localhost:9091"


def test_prometheus_falls_back_to_the_laptop_default():
    # Given:    no PROMETHEUS_URL at all
    # Expected: localhost:9090
    # Why:      `python -m agent.run` on a dev machine must keep working exactly
    #           as it did before any of this was configurable
    assert PrometheusMetrics(query_fn=lambda q: [])._url == DEFAULT_URL


# ---------------------------------------------------------------------------
# 5. The three files that must outlive the container
# ---------------------------------------------------------------------------


def test_episodic_memory_can_be_moved_onto_a_volume(monkeypatch, tmp_path):
    # Given:    LAB_AGENT_DB pointing outside the working directory
    # Expected: SqliteMemory writes there
    # Why:      `run --rm` destroys the container filesystem, so a relative path
    #           resets memory after every run -- NullMemory in SQLite's clothes
    target = tmp_path / "volume" / "lab_agent.db"
    target.parent.mkdir()
    monkeypatch.setenv("LAB_AGENT_DB", str(target))

    SqliteMemory().record({"kind": "RB-002", "container": "lab-victim"})

    assert target.exists()


def test_the_audit_trail_reads_its_path_when_it_writes_not_when_imported(monkeypatch, tmp_path):
    # Given:    LAB_AGENT_AUDIT set AFTER agent.audit was imported
    # Expected: the record still lands at the new path
    # Why:      run.py calls load_dotenv() inside main(), long after this module
    #           is imported. Reading the variable at import time would make
    #           LAB_AGENT_AUDIT in .env silently do nothing while the same
    #           variable in docker-compose.yml worked -- a bug that only appears
    #           where you cannot reproduce it
    target = tmp_path / "audit.jsonl"
    monkeypatch.setenv("LAB_AGENT_AUDIT", str(target))

    audit.write({"event": "attempting", "tool": "restart_container"})

    assert target.exists()
    assert audit.read_all()[0]["tool"] == "restart_container"


def test_an_explicit_audit_path_still_wins(monkeypatch, tmp_path):
    # Given:    an environment variable and an explicit path argument
    # Expected: the argument wins
    # Why:      gate_demo.py passes its own trail file so a demonstration never
    #           writes into the real evidence
    monkeypatch.setenv("LAB_AGENT_AUDIT", str(tmp_path / "env.jsonl"))
    explicit = tmp_path / "demo.jsonl"

    audit.write({"event": "attempting"}, path=str(explicit))

    assert explicit.exists()
    assert not (tmp_path / "env.jsonl").exists()


def test_the_checkpoint_file_can_be_moved_onto_a_volume(monkeypatch, tmp_path):
    # Given:    LAB_AGENT_CHECKPOINTS set after agent.checkpoint was imported
    # Expected: the SQLite file appears at that path
    # Why:      THE Stage 5 deployment trap. The gate saves the run and exits;
    #           the approval is a second container. If this file dies with the
    #           first one, the pause is unanswerable -- and the message is
    #           "Nothing is waiting for an answer", not an error
    from agent.checkpoint import sqlite_checkpointer

    target = tmp_path / "lab_checkpoints.db"
    monkeypatch.setenv("LAB_AGENT_CHECKPOINTS", str(target))

    sqlite_checkpointer()

    assert target.exists()


# ---------------------------------------------------------------------------
# 6. The provider switch in the composition root
# ---------------------------------------------------------------------------


def test_the_default_provider_is_free_and_offline():
    # Given:    LLM_PROVIDER unset, as on a freshly cloned repo
    # Expected: MockLLM
    # Why:      the first run on a shared machine should prove the pipeline, not
    #           spend money; USD 40 is the budget for the entire project
    assert isinstance(build_llm(rehearsing=False, resuming=False), MockLLM)


def test_openai_selects_the_http_adapter(monkeypatch):
    # Given:    LLM_PROVIDER=openai and a local Qwen's settings
    # Expected: an OpenAIAdapter aimed at that endpoint
    # Why:      "openai" names a DIALECT here; the same branch serves the paid
    #           API, the course gateway and Luis's Qwen
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_REQUIRE_AUTH", "false")
    monkeypatch.setenv("LLM_BASE_URL", "http://10.0.0.36:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b-instruct")

    llm = build_llm(rehearsing=False, resuming=False)

    assert isinstance(llm, OpenAIAdapter)
    assert llm.base_url == "http://10.0.0.36:11434/v1"


def test_a_misspelled_provider_is_refused_not_guessed(monkeypatch):
    # Given:    LLM_PROVIDER=opneai
    # Expected: SystemExit
    # Why:      falling back to the mock would print a complete, plausible trace
    #           that no model ever saw -- the most expensive kind of success
    monkeypatch.setenv("LLM_PROVIDER", "opneai")

    with pytest.raises(SystemExit):
        build_llm(rehearsing=False, resuming=False)


def test_rehearse_beats_the_provider_and_asks_for_a_restart(monkeypatch):
    # Given:    LLM_PROVIDER=openai but --rehearse in effect
    # Expected: a MockLLM whose first reply requests restart_container
    # Why:      the gate demo must be free and deterministic. Since Stage 4 a
    #           real model usually declines to restart -- right behaviour, and a
    #           demo that shows nothing
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.6-luna")

    llm = build_llm(rehearsing=True, resuming=False)

    assert isinstance(llm, MockLLM)
    assert "restart_container" in llm.decide("prompt")


def test_a_resumed_rehearsal_concludes_instead_of_asking_again():
    # Given:    --rehearse on a run being resumed after approval
    # Expected: the first reply concludes; it does not ask for another restart
    # Why:      act_node re-enters reason after the tool runs. Replaying the
    #           restart request would make the agent restart the container it
    #           has just restarted, and loop until the step bound stopped it
    llm = build_llm(rehearsing=True, resuming=True)

    first = llm.decide("prompt")
    assert '"action": "conclude"' in first
    assert "restart_container" not in first
