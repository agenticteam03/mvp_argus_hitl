"""
what: tests for the Docker metrics adapter, with a stub client instead of Docker.
why:  the mapping is where the bugs live -- Docker's JSON is deep, oddly named,
      and (as Step 3 measured) sometimes resets the very fields you need. That
      logic must be testable without a daemon, or the suite stops being offline
      and free the moment Stage 1 lands (rule 5).
how:  StubClient returns canned Docker JSON in the real shape. The interesting
      fixture is a LOOPING container: State says running, exit 0, oom false,
      while the daemon's events say exit 137 and oom. That combination is not
      invented for the test -- it is what a real lab-victim reported.
"""

from agent.adapters.metrics_docker import DockerMetrics
from agent.detectors import detect
from agent.ports import MetricsPort

# The seven keys agent/ports.py requires of every metrics adapter.
CONTRACT_KEYS = {
    "name",
    "running",
    "restart_count",
    "last_exit_code",
    "oom_killed",
    "memory_usage_bytes",
    "memory_limit_bytes",
}

# What `docker inspect lab-victim` really returned at t+26s, after nine restarts.
LOOPING_ATTRS = {
    "RestartCount": 9,
    "State": {"Status": "running", "Running": True, "ExitCode": 0, "OOMKilled": False},
}

# What `container.stats(stream=False)` returns: usage includes the page cache.
LOOPING_STATS = {
    "memory_stats": {
        "usage": 90 * 1024 * 1024,
        "limit": 128 * 1024 * 1024,
        "stats": {"inactive_file": 2 * 1024 * 1024},
    }
}

# What `docker events` recorded for the same container, newest last. This is the
# real shape from Docker Desktop 29.6.1, copied field for field: the action is in
# "Action", and exitCode arrives as a STRING.
LOOPING_EVENTS = [
    {"Type": "container", "Action": "oom", "Actor": {"Attributes": {"name": "lab-victim"}}},
    {
        "Type": "container",
        "Action": "die",
        "Actor": {"Attributes": {"name": "lab-victim", "exitCode": "137", "execDuration": "25"}},
    },
]


class StubContainer:
    def __init__(self, attrs: dict, stats: dict) -> None:
        self.attrs = attrs
        self._stats = stats

    def stats(self, stream: bool = True) -> dict:
        assert stream is False, "the adapter must take one sample, not a stream"
        return self._stats


class StubClient:
    """Stands in for docker.DockerClient: only the three calls the adapter makes."""

    def __init__(self, attrs: dict, stats: dict, events: list) -> None:
        self._container = StubContainer(attrs, stats)
        self._events = events
        self.containers = self  # so client.containers.get(name) works

    def get(self, name: str) -> StubContainer:
        return self._container

    def events(self, since=None, until=None, filters=None, decode=False) -> list:
        return self._events


def looping_adapter() -> DockerMetrics:
    return DockerMetrics(client=StubClient(LOOPING_ATTRS, LOOPING_STATS, LOOPING_EVENTS))


def test_the_adapter_satisfies_the_port():
    # Given:    a DockerMetrics built on a stub client
    # Expected: it passes isinstance against MetricsPort
    # Why:      if it does not fit the port, the one-line swap in run.py is a lie
    assert isinstance(looping_adapter(), MetricsPort)


def test_it_returns_exactly_the_contract_keys():
    # Given:    a looping container
    # Expected: the seven contract keys, no more and no fewer
    # Why:      an extra key is one a detector might use, and Prometheus could not supply
    assert set(looping_adapter().container_stats("lab-victim")) == CONTRACT_KEYS


def test_the_death_comes_from_events_not_from_state():
    # Given:    State says running/exit 0/oom false; events say oom and exit 137
    # Expected: last_exit_code 137 and oom_killed True
    # Why:      Docker resets State on every start, so State alone hides the OOM loop
    stats = looping_adapter().container_stats("lab-victim")
    assert stats["last_exit_code"] == 137
    assert stats["oom_killed"] is True
    assert stats["running"] is True  # it really is running, between two deaths
    assert stats["restart_count"] == 9


def test_a_looping_container_reaches_the_detector_as_rb_002():
    # Given:    the real shape of a looping container, mapped by the adapter
    # Expected: detect() returns an RB-002 incident for it
    # Why:      the end of the chain: real Docker JSON in, the same incident out
    incident = detect(looping_adapter().container_stats("lab-victim"))
    assert incident is not None
    assert incident.kind == "RB-002"


def test_state_is_the_fallback_when_the_container_is_stopped():
    # Given:    a stopped container, no events left in the window
    # Expected: the exit code and OOM flag come from State
    # Why:      State is trustworthy exactly when nothing has restarted it since
    stopped = {
        "RestartCount": 2,
        "State": {"Status": "exited", "Running": False, "ExitCode": 137, "OOMKilled": True},
    }
    adapter = DockerMetrics(client=StubClient(stopped, LOOPING_STATS, []))
    stats = adapter.container_stats("lab-victim")
    assert stats["running"] is False
    assert stats["last_exit_code"] == 137
    assert stats["oom_killed"] is True


def test_memory_excludes_the_page_cache():
    # Given:    usage 90 MiB of which 2 MiB is inactive_file
    # Expected: 88 MiB reported
    # Why:      so the number agrees with `docker stats` instead of quietly differing
    stats = looping_adapter().container_stats("lab-victim")
    assert stats["memory_usage_bytes"] == 88 * 1024 * 1024
    assert stats["memory_limit_bytes"] == 128 * 1024 * 1024


def test_it_also_understands_the_older_event_key():
    # Given:    events using the legacy "status" key instead of "Action"
    # Expected: the same 137 / True result
    # Why:      Docker renamed that field; an adapter pinned to one engine version
    #           reports a healthy container on the other one, silently
    legacy_events = [
        {"status": "oom", "Actor": {"Attributes": {"name": "lab-victim"}}},
        {"status": "die", "Actor": {"Attributes": {"name": "lab-victim", "exitCode": "137"}}},
    ]
    adapter = DockerMetrics(client=StubClient(LOOPING_ATTRS, LOOPING_STATS, legacy_events))
    stats = adapter.container_stats("lab-victim")
    assert stats["last_exit_code"] == 137
    assert stats["oom_killed"] is True
