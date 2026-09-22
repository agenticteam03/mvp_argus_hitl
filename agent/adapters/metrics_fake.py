"""
what: a MetricsPort adapter that returns canned numbers instead of asking Docker.
why:  Stage 0 must run with no Docker, no network and no waiting. Equally
      important, it must run with numbers we CHOSE -- reproducing a real OOM
      takes 13 seconds and produces slightly different values every time, which
      is fine for a demo and useless for a test.
how:  a dict matching the MetricsPort contract in agent/ports.py, key for key.
      Getting that contract right here is what makes Stage 1 a one-line swap:
      metrics_docker.py will fill the same keys from the Docker SDK, and neither
      the detectors nor the graph will notice the difference.

      Note the class imports nothing from ports.py. It satisfies MetricsPort by
      HAVING container_stats, not by declaring that it does.
"""

# A container mid-OOM-loop: 10 MB/s allocation against a 128 MiB cap, killed and
# restarted three times. These are the numbers RB-002's reproduction recipe
# produces on a real host.
OOM_VICTIM = {
    "name": "lab-victim",
    "running": True,  # true right now, because it just restarted again
    "restart_count": 3,
    "last_exit_code": 137,
    "oom_killed": True,
    "memory_usage_bytes": 127 * 1024 * 1024,
    "memory_limit_bytes": 128 * 1024 * 1024,
}

# The control case. Kept next to the incident so a test can prove the detector
# distinguishes them, rather than always saying yes.
HEALTHY_VICTIM = {
    "name": "lab-victim",
    "running": True,
    "restart_count": 0,
    "last_exit_code": None,
    "oom_killed": False,
    "memory_usage_bytes": 40 * 1024 * 1024,
    "memory_limit_bytes": 128 * 1024 * 1024,
}


class FakeMetrics:
    """Canned container stats. Satisfies MetricsPort."""

    def __init__(self, stats: dict | None = None) -> None:
        # Copied, not aliased: a caller that later edits the dict it passed in
        # must not silently change what this adapter reports.
        self._stats = dict(stats if stats is not None else OOM_VICTIM)

    def container_stats(self, name: str) -> dict:
        # The requested name wins, so asking about any container returns the
        # canned condition. Crude, but honest for a fake: Stage 1's real adapter
        # will simply 404 on a container that does not exist, and that
        # difference is worth meeting when it is real rather than simulated.
        return {**self._stats, "name": name}
