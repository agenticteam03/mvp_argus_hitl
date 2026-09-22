"""
what: tests for the SQLite episodic memory.
why:  the whole point of this adapter is that it remembers ACROSS processes and
      forgets what is old. Both claims need a test, and the second one needs a
      row planted in the past, because a test cannot wait 25 hours.
how:  pytest's tmp_path gives each test its own database file, so nothing here
      touches the real lab_agent.db and the tests can run in any order.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

from agent.adapters.memory_sqlite import SqliteMemory
from agent.ports import MemoryPort

OOM = {
    "kind": "RB-002",
    "container": "lab-victim",
    "diagnosis": "OOM restart loop",
    "confidence": 0.9,
    "action_taken": None,
}


def memory_in(tmp_path) -> SqliteMemory:
    return SqliteMemory(path=str(tmp_path / "test.db"))


def test_the_adapter_satisfies_the_port(tmp_path):
    # Given:    a SqliteMemory on a temporary file
    # Expected: it passes isinstance against MemoryPort
    # Why:      NullMemory and this must be interchangeable in run.py
    assert isinstance(memory_in(tmp_path), MemoryPort)


def test_what_is_recorded_can_be_recalled(tmp_path):
    # Given:    one recorded RB-002 incident
    # Expected: recent_incidents returns it with its fields intact
    # Why:      the minimum claim: writing then reading works
    memory = memory_in(tmp_path)
    memory.record(OOM)
    rows = memory.recent_incidents("RB-002", hours=24)
    assert len(rows) == 1
    assert rows[0]["container"] == "lab-victim"
    assert rows[0]["diagnosis"] == "OOM restart loop"
    assert rows[0]["confidence"] == 0.9


def test_a_different_kind_is_not_returned(tmp_path):
    # Given:    an RB-002 incident
    # Expected: asking for RB-001 returns nothing
    # Why:      recall must answer about ONE incident class, or the count is meaningless
    memory = memory_in(tmp_path)
    memory.record(OOM)
    assert memory.recent_incidents("RB-001", hours=24) == []


def test_it_survives_a_new_process(tmp_path):
    # Given:    an incident recorded, then a brand-new SqliteMemory on that file
    # Expected: the incident is still there
    # Why:      this is the ONE thing NullMemory could not do -- the reason for Stage 4
    path = str(tmp_path / "durable.db")
    SqliteMemory(path=path).record(OOM)
    assert len(SqliteMemory(path=path).recent_incidents("RB-002", hours=24)) == 1


def test_incidents_older_than_the_window_are_left_out(tmp_path):
    # Given:    one row planted 30 hours ago and one recorded now
    # Expected: a 24-hour window returns only the recent one
    # Why:      "three times in ten minutes" is a pattern; "three times this year" is not
    path = str(tmp_path / "old.db")
    memory = SqliteMemory(path=path)
    memory.record(OOM)
    long_ago = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    # Written directly, because the adapter always stamps "now" -- which is
    # correct behaviour, and exactly why the past has to be planted by hand.
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO incidents (ts, kind, container, diagnosis, confidence, action_taken) "
            "VALUES (?, 'RB-002', 'lab-victim', 'old one', 0.5, NULL)",
            (long_ago,),
        )
    assert len(memory.recent_incidents("RB-002", hours=24)) == 1
    assert len(memory.recent_incidents("RB-002", hours=48)) == 2


def test_a_run_with_no_diagnosis_can_still_be_recorded(tmp_path):
    # Given:    an incident where the model never produced a valid decision
    # Expected: the row is written, with diagnosis NULL
    # Why:      record_node writes exactly this after the step bound stops a run
    memory = memory_in(tmp_path)
    memory.record({"kind": "RB-002", "container": "lab-victim"})
    rows = memory.recent_incidents("RB-002", hours=24)
    assert rows[0]["diagnosis"] is None
    assert rows[0]["confidence"] is None


def test_rows_come_back_oldest_first(tmp_path):
    # Given:    two incidents on different containers
    # Expected: they return in the order they were written
    # Why:      the prompt says "prior incidents"; a jumbled order reads as noise
    memory = memory_in(tmp_path)
    memory.record({**OOM, "container": "lab-victim"})
    memory.record({**OOM, "container": "lab-other"})
    containers = [row["container"] for row in memory.recent_incidents("RB-002", hours=24)]
    assert containers == ["lab-victim", "lab-other"]
