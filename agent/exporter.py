"""
what: a small Prometheus exporter -- the process that holds numbers in memory
      and hands them out over HTTP when Prometheus asks.
why:  mvp_plan.md rules out cAdvisor on purpose. It needs /sys and
      /var/lib/docker mounts that behave differently inside Docker Desktop's
      WSL2 VM, and debugging that costs an evening and teaches nothing about
      agents. Thirty lines of our own behave identically on every machine and
      show what an exporter actually is.
how:  one Gauge per contract field, labelled by container. A loop reads each
      lab-* container through DockerMetrics -- the same adapter Stage 1 uses, so
      there is exactly ONE implementation of "how do I read a container",
      including the die/oom event trick -- and copies the values into the gauges.
      start_http_server serves them at :9101/metrics (see DEFAULT_PORT below
      for why that port and not 8000).

      Notice what this forces. Prometheus can only ever know what is published
      here, so the MetricsPort contract in ports.py decides what this file must
      expose. A field forgotten here is a field the Stage 2 adapter cannot
      return, and the detector would quietly stop firing.
"""

import time

from prometheus_client import Gauge, start_http_server

from agent.adapters.metrics_docker import DockerMetrics

# One gauge per contract field, keyed BY the contract key, so publish() can copy
# the adapter's dict across without a translation table that could drift.
#
# "container" is a label rather than part of the name: in Prometheus the name
# says what is measured and the labels say which thing it was measured on.
#
# restart_count, not restarts_total: by convention a _total suffix means a
# counter that only ever rises, and Stage 1 measured RestartCount dropping to 0
# after a manual restart. The suffix would lie to whoever writes the PromQL.
GAUGES = {
    "running": Gauge(
        "lab_container_running", "1 while the container is up", ["container"]
    ),
    "restart_count": Gauge(
        "lab_container_restart_count", "Docker's RestartCount", ["container"]
    ),
    "last_exit_code": Gauge(
        "lab_container_last_exit_code",
        "Exit code of the last death, -1 when unknown",
        ["container"],
    ),
    "oom_killed": Gauge(
        "lab_container_oom_killed",
        "1 if the last death was an OOM kill",
        ["container"],
    ),
    "memory_usage_bytes": Gauge(
        "lab_container_memory_usage_bytes",
        "Memory in use, page cache excluded",
        ["container"],
    ),
    "memory_limit_bytes": Gauge(
        "lab_container_memory_limit_bytes", "The cgroup memory limit", ["container"]
    ),
}


def publish(stats: dict) -> None:
    """Copy one container's stats dict into the gauges."""
    container = stats["name"]
    for key, gauge in GAUGES.items():
        gauge.labels(container).set(_as_number(stats[key]))


def _as_number(value) -> float:
    """Prometheus stores float64 and nothing else: no booleans, no None, no text.

    True becomes 1 and False 0 (a bool IS an int in Python). An unknown exit code
    becomes -1, which is not a real exit code, so it cannot be mistaken for one.
    Squeezing the contract through a numeric-only store is the cost of this
    stage, and unpacking it again is the Stage 2 adapter's job.
    """
    if value is None:
        return -1.0
    return float(value)


# 9101, not 8000. Measured on the team's laptop: a uvicorn application already
# owned 8000, and Docker Desktop had bound it too. Worse, on Windows a second
# process is allowed to bind a port that is already taken, so the exporter
# printed "listening" while the scrapes went to the other program, which
# answered 404. 9101 sits beside node_exporter's 9100, where exporters live.
DEFAULT_PORT = 9101


def _self_check(port: int) -> None:
    """Fetch our own /metrics and complain loudly if someone else answered.

    Starting is not the same as serving. On Windows a process may bind a port
    another program already holds, and the scrapes then reach whichever one the
    OS picks -- which is how an exporter can print "listening" while Prometheus
    reads 404s from an unrelated web app.
    """
    import urllib.request

    try:
        body = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=3).read()
    except Exception as exc:  # noqa: BLE001 - any failure here means the same thing
        print(f"WARNING: could not scrape myself on port {port}: {exc}", flush=True)
        return
    if b"lab_container_" not in body:
        print(
            f"WARNING: something else is answering on port {port} -- "
            "its reply has no lab_container_ metrics in it. Pick another port.",
            flush=True,
        )


def main(port: int = DEFAULT_PORT, interval_seconds: int = 5) -> None:
    """Serve /metrics and refresh the numbers forever."""
    import docker

    client = docker.from_env()
    metrics = DockerMetrics(client=client)
    # Starts a background thread and returns immediately; the loop below is what
    # keeps the process alive.
    start_http_server(port)
    print(f"exporter on http://localhost:{port}/metrics - Ctrl+C to stop", flush=True)
    _self_check(port)

    while True:
        # all=True so a container that is dead right now is still reported. The
        # name filter is a substring match on the same lab- prefix the allowlist
        # uses -- this exporter has no business publishing anything else.
        containers = client.containers.list(all=True, filters={"name": "lab-"})
        for container in containers:
            publish(metrics.container_stats(container.name))
        print(
            f"{time.strftime('%H:%M:%S')} published {len(containers)} container(s)",
            flush=True,
        )
        # Scrapes are pulls, so this interval only decides how stale the numbers
        # can be when Prometheus arrives. Nothing is pushed anywhere.
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
