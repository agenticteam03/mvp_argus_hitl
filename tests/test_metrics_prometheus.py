"""
what: tests for the Prometheus adapter, with canned API responses instead of a
      running Prometheus.
why:  two jobs are being checked. The translation back from floats into the
      contract's types, which is where the quiet bugs live (-1 must become None,
      1.0 must become True). And the two refusals: no data and stale data must
      raise, never return a dict that reads as "healthy".
how:  query_fn is injected, so nothing here touches HTTP. The fixtures copy the
      real API shape, where a sample is [timestamp, "value"] and the value is a
      string.
"""

import time

import pytest

from agent.adapters.metrics_prometheus import MetricsUnavailable, PrometheusMetrics
from agent.detectors import detect
from agent.ports import MetricsPort

CONTRACT_KEYS = {
    "name",
    "running",
    "restart_count",
    "last_exit_code",
    "oom_killed",
    "memory_usage_bytes",
    "memory_limit_bytes",
}


def sample(metric: str, value, age_seconds: float = 0.0) -> dict:
    """One series as the Prometheus HTTP API returns it."""
    return {
        "metric": {"__name__": metric, "container": "lab-victim", "job": "lab-exporter"},
        "value": [time.time() - age_seconds, str(value)],
    }


def looping(age_seconds: float = 0.0) -> list[dict]:
    """A container in an OOM restart loop, as six Prometheus series."""
    return [
        sample("lab_container_running", 1.0, age_seconds),
        sample("lab_container_restart_count", 88.0, age_seconds),
        sample("lab_container_last_exit_code", 137.0, age_seconds),
        sample("lab_container_oom_killed", 1.0, age_seconds),
        sample("lab_container_memory_usage_bytes", 133169152.0, age_seconds),
        sample("lab_container_memory_limit_bytes", 134217728.0, age_seconds),
    ]


def adapter_for(series: list[dict], **kwargs) -> PrometheusMetrics:
    return PrometheusMetrics(query_fn=lambda promql: series, **kwargs)


def test_the_adapter_satisfies_the_port():
    # Given:    a PrometheusMetrics built on a canned query function
    # Expected: it passes isinstance against MetricsPort
    # Why:      if it does not fit the port, run.py cannot swap to it in one line
    assert isinstance(adapter_for(looping()), MetricsPort)


def test_it_returns_exactly_the_contract_keys():
    # Given:    six series for one container
    # Expected: the seven contract keys, no more and no fewer
    # Why:      the same contract Docker fills -- that is what makes them interchangeable
    assert set(adapter_for(looping()).container_stats("lab-victim")) == CONTRACT_KEYS


def test_floats_are_translated_back_into_the_contract_types():
    # Given:    Prometheus floats: 1.0 for running, 137.0 for the exit code
    # Expected: True for running, int 137, int restart count
    # Why:      the exporter flattened everything to float64; someone must unflatten it
    stats = adapter_for(looping()).container_stats("lab-victim")
    assert stats["running"] is True
    assert stats["oom_killed"] is True
    assert stats["last_exit_code"] == 137
    assert stats["restart_count"] == 88


def test_minus_one_becomes_none_again():
    # Given:    a container that has never exited, published as -1
    # Expected: last_exit_code is None
    # Why:      -1 was only ever a stand-in for None, which Prometheus cannot store
    series = [s for s in looping() if s["metric"]["__name__"] != "lab_container_last_exit_code"]
    series.append(sample("lab_container_last_exit_code", -1.0))
    assert adapter_for(series).container_stats("lab-victim")["last_exit_code"] is None


def test_a_looping_container_reaches_the_detector_as_rb_002():
    # Given:    the Prometheus view of a looping container
    # Expected: detect() returns an RB-002 incident
    # Why:      the whole point of Stage 2 -- new source, same verdict, unchanged detector
    incident = detect(adapter_for(looping()).container_stats("lab-victim"))
    assert incident is not None
    assert incident.kind == "RB-002"


def test_no_series_is_an_error_not_a_healthy_container():
    # Given:    Prometheus knows nothing about this container
    # Expected: MetricsUnavailable
    # Why:      absence of data is not evidence of health -- the worst lie a monitor can tell
    with pytest.raises(MetricsUnavailable):
        adapter_for([]).container_stats("lab-victim")


def test_stale_samples_are_an_error():
    # Given:    every sample is 120 s old (a dead exporter)
    # Expected: MetricsUnavailable naming the age
    # Why:      instant queries look back 5 minutes, so stale data arrives sounding fresh
    with pytest.raises(MetricsUnavailable) as err:
        adapter_for(looping(age_seconds=120)).container_stats("lab-victim")
    assert "120s old" in str(err.value)


def test_a_missing_field_is_an_error():
    # Given:    the exporter publishes everything except oom_killed
    # Expected: MetricsUnavailable naming the missing field
    # Why:      a contract half-filled would silently stop the detector from firing
    series = [s for s in looping() if s["metric"]["__name__"] != "lab_container_oom_killed"]
    with pytest.raises(MetricsUnavailable) as err:
        adapter_for(series).container_stats("lab-victim")
    assert "oom_killed" in str(err.value)


def test_the_query_asks_for_one_container():
    # Given:    a read for lab-victim
    # Expected: the PromQL selector names that container
    # Why:      without the label the adapter would average over every container
    asked = []

    def spy(promql: str) -> list[dict]:
        asked.append(promql)
        return looping()

    PrometheusMetrics(query_fn=spy).container_stats("lab-victim")
    assert 'container="lab-victim"' in asked[0]
    assert "lab_container_" in asked[0]
