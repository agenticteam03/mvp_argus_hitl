"""
what: the three real tools -- two reads and one write -- against the local
      Docker daemon.
why:  Stage 0's fakes proved the plumbing; these do the work. This is also the
      file the allowlist exists for. restart_container really restarts a
      container, and from here on the only thing between a model's request and
      that happening is the check in registry.py.
how:  one Docker client, created on first use, shared by the three functions.
      Every tool returns a one-line observation STRING, because observations are
      pasted into the next prompt and a model reads text.

      get_container_stats reuses DockerMetrics instead of repeating its event
      reading. That is Docker-specific code calling Docker-specific code, not a
      port violation: this module IS the Docker implementation. Notice what it
      means from Stage 2 on -- the DETECTOR's numbers will come from Prometheus
      while this tool still reads Docker live. That is deliberate, and it is
      what a human does too: alert from monitoring, confirm on the machine.
"""

from typing import Any

from agent.adapters.metrics_docker import DockerMetrics

_client: Any = None


def _docker() -> Any:
    """The Docker client, created on first use.

    Lazy on purpose: registry.py imports this module at start-up, and pytest
    imports registry.py. Connecting to the daemon at import time would mean no
    test could even be COLLECTED without Docker Desktop running.
    """
    global _client
    if _client is None:
        import docker

        _client = docker.from_env()
    return _client


def get_container_stats(name: str) -> str:
    """Read tool. Current state, restart count, memory, and how it last died."""
    stats = DockerMetrics(client=_docker()).container_stats(name)
    used_mib = stats["memory_usage_bytes"] // (1024 * 1024)
    limit_mib = stats["memory_limit_bytes"] // (1024 * 1024)
    # Same field order and names as the Stage 0 fake, so the trace reads the
    # same whether the numbers are real or canned.
    return (
        f"{name}: running={stats['running']} restart_count={stats['restart_count']} "
        f"last_exit_code={stats['last_exit_code']} oom_killed={stats['oom_killed']} "
        f"memory={used_mib}MiB/{limit_mib}MiB"
    )


def get_container_logs(name: str, lines: int = 50) -> str:
    """Read tool. The last `lines` lines the container wrote."""
    container = _docker().containers.get(name)
    # tail=N asks the daemon for the last N lines instead of the whole history.
    raw = container.logs(tail=lines, stdout=True, stderr=True)
    # errors="replace" because container output is arbitrary bytes, and a tool
    # that raises UnicodeDecodeError mid-investigation is worse than a few "?".
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        # The expected result for an OOM kill: SIGKILL cannot be caught, so the
        # process never gets to write anything. RB-002 sends you to dmesg.
        return f"{name}: last {lines} log lines: (no output)"
    return f"{name}: last {lines} log lines:\n{text}"


def restart_container(name: str) -> str:
    """WRITE tool. Really restarts the container. Gated by the allowlist.

    The observation stays factual -- what was done, and the counts before and
    after. It deliberately does NOT append "a restart is not a fix": that is the
    runbook's judgement, and the runbook reaches the model through DocsPort in
    Stage 4. A tool that argues with the model is a prompt hiding in a tool.
    """
    container = _docker().containers.get(name)
    before = container.attrs.get("RestartCount", 0)
    # timeout=5: SIGTERM first, SIGKILL five seconds later if it is still there.
    container.restart(timeout=5)
    container.reload()  # attrs are a snapshot; refresh them after the restart
    after = container.attrs.get("RestartCount", 0)
    return f"restarted {name} (RestartCount {before} -> {after}, status {container.status})"
