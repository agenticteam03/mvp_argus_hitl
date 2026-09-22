"""
what: tests for the exporter's mapping -- no Docker, no HTTP server.
why:  the exporter is where the contract turns into numbers, and numbers are all
      Prometheus can store. A field missing here is a field the Stage 2 adapter
      cannot return, and the detector would quietly stop firing. So the first
      test is simply: does every contract field have a gauge?
how:  call publish() with a stats dict, then read the values back out of
      prometheus_client's default registry -- the same values a scrape would get.
"""

from prometheus_client import REGISTRY

from agent.exporter import GAUGES, publish

# The MetricsPort contract, minus "name", which is a LABEL rather than a gauge.
CONTRACT_FIELDS = {
    "running",
    "restart_count",
    "last_exit_code",
    "oom_killed",
    "memory_usage_bytes",
    "memory_limit_bytes",
}

LOOPING = {
    "name": "lab-victim",
    "running": True,
    "restart_count": 88,
    "last_exit_code": 137,
    "oom_killed": True,
    "memory_usage_bytes": 127 * 1024 * 1024,
    "memory_limit_bytes": 128 * 1024 * 1024,
}


def scraped(metric: str, container: str = "lab-victim"):
    """What a scrape of this metric would return for one container."""
    return REGISTRY.get_sample_value(metric, {"container": container})


def test_every_contract_field_has_a_gauge():
    # Given:    the exporter's gauge table
    # Expected: exactly the contract fields, minus name
    # Why:      Prometheus knows only what is published; a gap here breaks Stage 2 silently
    assert set(GAUGES) == CONTRACT_FIELDS


def test_publish_writes_what_a_scrape_would_read():
    # Given:    a looping container's stats
    # Expected: the same numbers come back out of the registry
    # Why:      proves the gauges are actually set, not merely declared
    publish(LOOPING)
    assert scraped("lab_container_restart_count") == 88
    assert scraped("lab_container_last_exit_code") == 137
    assert scraped("lab_container_memory_limit_bytes") == 128 * 1024 * 1024


def test_booleans_become_one_and_zero():
    # Given:    running True and oom_killed True, then both False
    # Expected: 1.0 then 0.0
    # Why:      Prometheus stores float64 only -- no booleans, no strings
    publish(LOOPING)
    assert scraped("lab_container_running") == 1.0
    assert scraped("lab_container_oom_killed") == 1.0
    publish({**LOOPING, "running": False, "oom_killed": False})
    assert scraped("lab_container_running") == 0.0
    assert scraped("lab_container_oom_killed") == 0.0


def test_an_unknown_exit_code_becomes_minus_one():
    # Given:    a container that has never exited (last_exit_code None)
    # Expected: -1 published
    # Why:      None cannot be stored; -1 is not a real exit code, so it cannot be misread
    publish({**LOOPING, "last_exit_code": None})
    assert scraped("lab_container_last_exit_code") == -1.0


def test_each_container_gets_its_own_labelled_series():
    # Given:    two containers published through the same gauges
    # Expected: each keeps its own value
    # Why:      that is what the label is for -- one metric, many things measured
    publish(LOOPING)
    publish({**LOOPING, "name": "lab-other", "restart_count": 3})
    assert scraped("lab_container_restart_count", "lab-victim") == 88
    assert scraped("lab_container_restart_count", "lab-other") == 3
