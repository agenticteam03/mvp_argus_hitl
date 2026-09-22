"""
what: the four interfaces the agent talks to the outside world through --
      metrics, model, documents, memory.
why:  this is the file the whole six-stage plan turns on. The graph, the
      detectors and the tests are written against these names and never against
      a concrete class, so Stage 2 can move the metrics source from Docker to
      Prometheus by editing ONE line in run.py. If swapping an adapter ever
      forces an edit to graph.py, detectors.py or a test, the port is wrong --
      fix the port, not the caller.
how:  Protocol gives structural typing: a class satisfies MetricsPort by having
      a container_stats method with the right shape, NOT by inheriting from it.
      Adapters therefore never import this file. That is the point -- the
      dependency arrow runs from the graph to the port, and from the adapter to
      nothing.

      @runtime_checkable enables isinstance() checks, which the tests use to
      demonstrate the idea. Be honest about its limit: at runtime it verifies
      that the METHOD NAMES exist, and nothing about their signatures or return
      types. Static checking is what catches the rest.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsPort(Protocol):
    """Where the agent gets numbers about a container.

    Implemented by metrics_fake (Stage 0), metrics_docker (Stage 1) and
    metrics_prometheus (Stage 2). Returns a plain dict rather than a typed
    object so that a new source can add keys without breaking old callers.

    THE CONTRACT. Every adapter must return at least these keys, because
    agent/detectors.py reads them by name:

        name                 str        the container
        running              bool       is it up right now
        restart_count        int        Docker's RestartCount
        last_exit_code       int|None   None if it has never exited
        oom_killed           bool       Docker's State.OOMKilled
        memory_usage_bytes   int
        memory_limit_bytes   int        0 when no limit is set

    Write this down and mean it. In Stage 2 the Prometheus exporter will have to
    publish last_exit_code and oom_killed as gauges purely because this contract
    says so -- a source that cannot supply a field does not get to quietly drop
    it, or the detectors silently weaken. That negotiation IS the ports lesson.
    """

    def container_stats(self, name: str) -> dict: ...


@runtime_checkable
class LLMPort(Protocol):


   

    def decide(self, prompt: str) -> str: ...













@runtime_checkable
class DocsPort(Protocol):
    """Semantic memory: retrieval over the runbooks. Real in Stage 4.

    k defaults to 3 because a prompt that carries ten chunks of runbook is
    mostly noise -- and because the Stage 4 lesson is that retrieval quality is
    dominated by which corpus you point at, not by how many chunks you take.
    """

    def search(self, query: str, k: int = 3) -> list[str]: ...


@runtime_checkable
class MemoryPort(Protocol):
    """Episodic memory: what has already happened. Real in Stage 4.

    recent_incidents is what lets the agent say "third OOM on this container in
    ten minutes" -- a sentence that is impossible for a stateless script, and
    the reason this port exists from Stage 0 even while it returns [].
    """

    def recent_incidents(self, kind: str, hours: int) -> list[dict]: ...
    def record(self, incident: dict) -> None: ...
