"""
what: a MetricsPort adapter that fills the contract from the real Docker daemon.
why:  same seven keys as metrics_fake.py, from a real container. Nothing in
      graph.py, detectors.py or the tests knows the difference -- that is the
      whole point of the port, and Stage 2 will do it again with Prometheus.
how:  three sources, because one is not enough:

      1. container.attrs      -> running, restart_count
      2. container.stats()    -> memory usage and limit
      3. the daemon's EVENTS  -> how it last died (exit code, OOM or not)

      Source 3 is the surprise, and it came from measuring rather than guessing.
      Docker rebuilds State when it starts a container, so on a live lab-victim
      that had already restarted nine times, `docker inspect` read:

          status=running restarts=9 exit=0 oom=false

      The restart count survives; the exit code and the OOM flag do not. An
      adapter that read only State would report a healthy container, because a
      looping container is running almost all of the time -- ours dies once
      every 26 seconds and is back within a fraction of a second.

      The daemon keeps the real record as `oom` and `die` events, and the die
      event carries exitCode=137. That is exactly what RB-002 tells a human to
      check ("docker events --since 1h --filter event=die --filter event=oom"),
      so the adapter checks it too.
"""

import time
from typing import Any

# How far back to look for the last death. Ten minutes is long enough to cover
# several cycles of a looping container and short enough that an OOM from this
# morning is not reported as current news.
DEFAULT_EVENT_WINDOW_SECONDS = 600


class DockerMetrics:
    """Real container stats from the local Docker daemon. Satisfies MetricsPort."""

    def __init__(
        self,
        client: Any = None,
        event_window_seconds: int = DEFAULT_EVENT_WINDOW_SECONDS,
    ) -> None:
        # The client is injectable so tests can pass a stub and stay offline
        # (rule 5). docker is imported HERE, not at module level, so importing
        # this file never needs a running daemon.
        if client is None:
            import docker

            client = docker.from_env()
        self._client = client
        self._window = event_window_seconds

    def container_stats(self, name: str) -> dict:
        container = self._client.containers.get(name)
        attrs = container.attrs
        state = attrs.get("State", {})
        usage, limit = self._memory(container)
        exit_code, oom_killed = self._last_death(name, state)
        # Exactly the seven keys agent/ports.py promises. No extras: a key that
        # only the Docker adapter returns is a key a detector might come to rely
        # on, and then Prometheus cannot replace it in Stage 2.
        return {
            "name": name,
            "running": bool(state.get("Running", False)),
            "restart_count": int(attrs.get("RestartCount", 0)),
            "last_exit_code": exit_code,
            "oom_killed": oom_killed,
            "memory_usage_bytes": usage,
            "memory_limit_bytes": limit,
        }

    def _memory(self, container: Any) -> tuple[int, int]:
        """Current memory use and the cgroup limit, in bytes."""
        # stream=False takes ONE sample. The default streams forever, which would
        # hang the agent on its first tool call.
        stats = container.stats(stream=False)
        memory = stats.get("memory_stats", {})
        usage = int(memory.get("usage", 0))
        # `docker stats` subtracts the page cache before printing a number, and a
        # figure that disagrees with `docker stats` starts an argument about
        # which one is lying. On cgroup v2 that field is called inactive_file.
        cache = int(memory.get("stats", {}).get("inactive_file", 0))
        return max(usage - cache, 0), int(memory.get("limit", 0))

    def _last_death(self, name: str, state: dict) -> tuple[int | None, bool]:
        """How this container last died: (exit code, was it an OOM kill).

        Reads the daemon's event log rather than State, for the reason in the
        file header: State is reset every time the container starts.
        """
        now = int(time.time())
        events = self._client.events(
            since=now - self._window,
            until=now,  # bounded, so this returns instead of streaming forever
            filters={"container": name, "event": ["die", "oom"]},
            decode=True,  # dicts instead of raw JSON bytes
        )

        exit_code: int | None = None
        oom_killed = False
        for event in events:
            # Docker 25+ names this field Action; older daemons called it status.
            # Read whichever is present rather than pinning the adapter to one
            # engine version -- measured on Docker Desktop 29.6.1, which sends
            # Action and no status at all, so reading only status found nothing
            # and reported a healthy container.
            action = event.get("Action") or event.get("status")
            if action == "oom":
                # An oom event anywhere in the window counts. Pairing each oom
                # with its die would be more precise and, for a container that
                # dies every 26 seconds, would change nothing.
                oom_killed = True
            elif action == "die":
                code = event.get("Actor", {}).get("Attributes", {}).get("exitCode")
                if code is not None:
                    exit_code = int(code)  # the attribute arrives as a string

        # A container that is stopped RIGHT NOW has a State worth trusting: it
        # was written at the moment of death and nothing has restarted it since.
        # This is the fallback for a death older than the event window.
        if exit_code is None and not state.get("Running", False):
            exit_code = state.get("ExitCode")
            oom_killed = bool(state.get("OOMKilled", False))

        return exit_code, oom_killed
