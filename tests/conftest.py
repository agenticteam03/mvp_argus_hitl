"""
what: one fixture, applied automatically to every test: the tool registry points
      at the fake tools.
why:  from Stage 1 the real tools talk to Docker. get_container_logs reads a
      real container, and restart_container really restarts one. A suite that
      did that would need Docker Desktop running, would be slow, and would
      restart containers as a side effect of `pytest`. Rule 5 says the suite
      stays offline and free, forever -- so the tools get swapped at the
      boundary, the same way MockLLM stands in for a model.
how:  an autouse fixture plus monkeypatch.setitem on REGISTRY, undone after each
      test. ALLOWED is NOT touched: the policy under test stays the real one,
      and the refusal tests still exercise the real dispatcher -- they never
      reach a tool anyway, because dispatch() raises before calling one.

      conftest.py is pytest's own convention: fixtures defined here apply to
      every test in this folder without being imported anywhere.
"""

import pytest

from agent.tools import container_fake
from agent.tools.registry import REGISTRY

FAKE_TOOLS = {
    "get_container_stats": container_fake.get_container_stats,
    "get_container_logs": container_fake.get_container_logs,
    "restart_container": container_fake.restart_container,
}


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    """Point every tool name at its fake, for the duration of one test."""
    for name, function in FAKE_TOOLS.items():
        monkeypatch.setitem(REGISTRY, name, function)


@pytest.fixture(autouse=True)
def audit_to_tmp(tmp_path, monkeypatch):
    """Send every test's audit trail to its own temporary file.

    Without this, running pytest would append to the real audit.jsonl -- the
    file that is supposed to be evidence of what the agent actually did. A test
    suite forging entries into it would make it worthless.

    It sets the ENVIRONMENT VARIABLE rather than the module's DEFAULT_PATH, and
    that distinction is load-bearing since the paths became deployable. audit.py
    resolves argument > LAB_AGENT_AUDIT > DEFAULT_PATH, so patching the global
    alone would be silently overridden wherever the variable is set -- including
    inside the agent container, where docker-compose.yml sets it, and where
    `docker compose run --rm agent python -m pytest` is the documented way to
    verify a deployment. The suite would then forge entries into the real trail:
    the exact failure this fixture exists to prevent.

    Returns the path, so a test that wants to read what was audited can ask for
    this fixture by name.
    """
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("LAB_AGENT_AUDIT", str(path))
    return str(path)
