"""
what: the specification for agent/detectors.py, written before the detector.
why:  two of these tests are lifted straight out of RB-002 and are the reason
      the detector is not a one-line `exit_code == 137`. The runbook's own DO NOT
      list says: "Never conclude 'OOM' from exit code 137 alone" -- 137 is
      SIGKILL, which an external `docker kill` produces just as readily as the
      kernel OOM killer. A detector that ignores that is confidently wrong, and
      confidently wrong is the worst thing an ops agent can be.
how:  `stats()` builds a healthy container, and each test overrides exactly one
      field. One abnormal condition per test, so a failure names its own cause.

      These run against dicts, never against Docker. That is the point of the
      purity rule: detectors take a metrics dict and return an incident or None,
      with no I/O, no network and no clock -- so they are testable at zero cost
      and behave identically whether the numbers came from Docker or Prometheus.
"""

import pytest

from agent.detectors import GRACEFUL_EXIT_CODE, OOM_EXIT_CODE, detect
from agent.models import Incident


def stats(**overrides) -> dict:
    """A healthy lab-victim, matching the MetricsPort contract in ports.py."""
    baseline = {
        "name": "lab-victim",
        "running": True,
        "restart_count": 0,
        "last_exit_code": None,
        "oom_killed": False,
        "memory_usage_bytes": 40 * 1024 * 1024,
        "memory_limit_bytes": 128 * 1024 * 1024,
    }
    baseline.update(overrides)
    return baseline


def test_healthy_container_is_not_an_incident():
    # Given:    the healthy baseline, unchanged
    # Expected: None
    # Why:      the control case -- a detector that always fires is useless
    assert detect(stats()) is None


def test_detects_oom_from_exit_137():
    # Given:    exit 137 + OOMKilled true + 3 restarts + memory at the limit
    # Expected: an RB-002 Incident for lab-victim
    # Why:      the one incident this whole MVP exists to handle
    incident = detect(
        stats(last_exit_code=OOM_EXIT_CODE, oom_killed=True, restart_count=3,
              memory_usage_bytes=127 * 1024 * 1024)
    )
    assert isinstance(incident, Incident)
    assert incident.kind == "RB-002"
    assert incident.container == "lab-victim"


def test_ignores_exit_143():
    # Given:    exit 143 (SIGTERM) with 3 restarts
    # Expected: None
    # Why:      143 is a polite stop, e.g. a deploy -- not a fault
    assert detect(stats(last_exit_code=GRACEFUL_EXIT_CODE, restart_count=3)) is None


def test_exit_137_without_oom_confirmation_is_not_an_incident():
    # Given:    exit 137 but OOMKilled false
    # Expected: None
    # Why:      RB-002's DO NOT list -- `docker kill` also gives 137; unconfirmed is not an OOM
    assert detect(stats(last_exit_code=OOM_EXIT_CODE, oom_killed=False, restart_count=3)) is None


def test_a_single_death_is_not_yet_a_loop():
    # Given:    a confirmed OOM kill, but 0 restarts
    # Expected: None
    # Why:      it died and stayed dead -- a problem, but not a restart loop
    assert detect(stats(last_exit_code=OOM_EXIT_CODE, oom_killed=True, restart_count=0)) is None


def test_incident_carries_the_evidence_it_reasoned_from():
    # Given:    a confirmed OOM loop
    # Expected: restart count, exit code and OOMKilled copied into evidence
    # Why:      the prompt, the trace and the audit record must cite the same numbers
    incident = detect(stats(last_exit_code=OOM_EXIT_CODE, oom_killed=True, restart_count=3))
    assert incident is not None
    assert incident.evidence["restart_count"] == 3
    assert incident.evidence["last_exit_code"] == OOM_EXIT_CODE
    assert incident.evidence["oom_killed"] is True


def test_detector_does_not_mutate_its_input():
    # Given:    a metrics dict, and a copy taken before calling detect()
    # Expected: the dict is unchanged afterwards
    # Why:      purity -- a detector that edits its input makes the order of detectors matter
    metrics = stats(last_exit_code=OOM_EXIT_CODE, oom_killed=True, restart_count=3)
    before = dict(metrics)
    detect(metrics)
    assert metrics == before


def test_detect_is_a_pure_function_of_its_argument():
    # Given:    the same metrics dict, passed in twice
    # Expected: two equal results
    # Why:      no clock, no counter, no hidden state -- same input, same output
    metrics = stats(last_exit_code=OOM_EXIT_CODE, oom_killed=True, restart_count=3)
    first, second = detect(metrics), detect(metrics)
    assert first == second
