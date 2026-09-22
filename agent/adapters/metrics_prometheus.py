"""
what: a MetricsPort adapter that answers from Prometheus instead of Docker.
why:  the Stage 2 payoff. The same seven keys, the same detector, the same
      graph and the same tests -- an entirely different source of truth, chosen
      by one line in run.py.

      It also brings a hazard Docker did not have. A live Docker read either
      works or raises; Prometheus will cheerfully answer with a five-minute-old
      number in exactly the same tone as a fresh one. Its instant queries look
      back up to five minutes by default, so an exporter that died four minutes
      ago still "reports" a healthy container.
how:  one instant query per read. A label selector fetches all six
      lab_container_* series for the container in a single round trip --
      __name__ can be matched like any other label -- and then the numbers are
      translated back into the contract's types (-1 becomes None, 1.0 becomes
      True), undoing the flattening the exporter had to do.

      Two situations are raised as errors rather than returned as plausible
      dicts: no series at all, and samples older than max_staleness_seconds.
      Both would otherwise read as "healthy container", which is the worst lie a
      monitoring agent can tell.
"""

import os
import time
from collections.abc import Callable

# The laptop's answer. Inside a container it is the WRONG answer: localhost
# there means the container itself, not the host and certainly not another
# service. PROMETHEUS_URL overrides it, and docker-compose.yml sets that to
# http://prometheus:9090 -- the compose SERVICE name and the port Prometheus
# listens on inside its own container, never the port published on the host.
DEFAULT_URL = "http://localhost:9090"
# Six times the 5 s scrape interval. Long enough to survive a missed scrape or
# two, short enough that a dead exporter is noticed within half a minute.
DEFAULT_MAX_STALENESS_SECONDS = 30.0

# Prometheus metric name -> contract key. The exporter builds the same mapping
# in the other direction; between them they are the whole translation layer.
FIELD_BY_METRIC = {
    "lab_container_running": "running",
    "lab_container_restart_count": "restart_count",
    "lab_container_last_exit_code": "last_exit_code",
    "lab_container_oom_killed": "oom_killed",
    "lab_container_memory_usage_bytes": "memory_usage_bytes",
    "lab_container_memory_limit_bytes": "memory_limit_bytes",
}


class MetricsUnavailable(RuntimeError):
    """Prometheus could not answer with fresh numbers for this container."""


class PrometheusMetrics:
    """Container stats read from Prometheus. Satisfies MetricsPort."""

    def __init__(
        self,
        url: str | None = None,
        query_fn: Callable[[str], list[dict]] | None = None,
        max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS,
        timeout: float = 5.0,
    ) -> None:
        # Argument wins over environment, environment wins over the default --
        # the same order llm_openai.py uses, so there is one rule to remember
        # rather than one per adapter. run.py therefore stays a single line even
        # though the address now differs between the laptop and the VM.
        self._url = (url or os.environ.get("PROMETHEUS_URL") or DEFAULT_URL).rstrip("/")
        # Injectable so the tests can hand over canned Prometheus JSON and stay
        # offline, the same trick DockerMetrics uses with its client.
        self._query = query_fn if query_fn is not None else self._http_query
        self._max_staleness = max_staleness_seconds
        self._timeout = timeout

    def container_stats(self, name: str) -> dict:
        series = self._query(f'{{__name__=~"lab_container_.*", container="{name}"}}')
        if not series:
            raise MetricsUnavailable(
                f"Prometheus has no lab_container_* series for {name!r}: the exporter "
                "may never have seen this container, or the scrape is failing."
            )

        values: dict[str, float] = {}
        oldest_sample: float | None = None
        for item in series:
            field = FIELD_BY_METRIC.get(item.get("metric", {}).get("__name__", ""))
            if field is None:
                continue  # some other lab_container_ metric we do not need
            # The API returns [timestamp, "value"] -- the value is a STRING.
            timestamp, raw = item["value"]
            values[field] = float(raw)
            oldest_sample = min(float(timestamp), oldest_sample or float(timestamp))

        missing = set(FIELD_BY_METRIC.values()) - set(values)
        if missing:
            raise MetricsUnavailable(
                f"Prometheus is missing {sorted(missing)} for {name!r}: the exporter "
                "publishes fewer fields than the contract requires."
            )

        # The staleness check. Without it, a dead exporter is indistinguishable
        # from a healthy container for five minutes.
        age = time.time() - float(oldest_sample)
        if age > self._max_staleness:
            raise MetricsUnavailable(
                f"the newest samples for {name!r} are {age:.0f}s old (limit "
                f"{self._max_staleness:.0f}s): the exporter is probably dead."
            )

        # Undo the exporter's flattening: floats back into the contract's types.
        return {
            "name": name,
            "running": bool(values["running"]),
            "restart_count": int(values["restart_count"]),
            # -1 was our stand-in for "never exited", because Prometheus cannot
            # store None. It is not a real exit code, so nothing else means it.
            "last_exit_code": (
                None if values["last_exit_code"] == -1 else int(values["last_exit_code"])
            ),
            "oom_killed": bool(values["oom_killed"]),
            "memory_usage_bytes": int(values["memory_usage_bytes"]),
            "memory_limit_bytes": int(values["memory_limit_bytes"]),
        }

    def _http_query(self, promql: str) -> list[dict]:
        """One instant query against the same HTTP API you used from PowerShell."""
        import httpx

        response = httpx.get(
            f"{self._url}/api/v1/query", params={"query": promql}, timeout=self._timeout
        )
        response.raise_for_status()
        body = response.json()
        # Prometheus answers 200 with status="error" for a bad query, so the
        # HTTP status alone is not enough to trust the body.
        if body.get("status") != "success":
            raise MetricsUnavailable(f"Prometheus rejected the query: {body.get('error')}")
        return body["data"]["result"]
