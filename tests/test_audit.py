"""
what: tests for the audit trail.
why:  this file exists to be readable after something went wrong, so the tests
      are about durability and completeness: lines are appended not replaced,
      every line carries a time, awkward values do not stop a write, and a
      truncated final line does not hide the good records before it.
how:  tmp_path gives each test its own file, so nothing here touches the real
      audit.jsonl and the tests can run in any order.
"""

import json

from agent import audit


def test_a_record_is_written_and_read_back(tmp_path):
    # Given:    one record written to a fresh file
    # Expected: read_all returns it with its fields intact
    # Why:      the minimum claim of an audit trail
    path = str(tmp_path / "audit.jsonl")
    audit.write({"event": "attempting", "tool": "restart_container"}, path=path)
    records = audit.read_all(path)
    assert len(records) == 1
    assert records[0]["tool"] == "restart_container"


def test_every_record_is_stamped(tmp_path):
    # Given:    a record with no timestamp of its own
    # Expected: a UTC "ts" field is added
    # Why:      "when" is half of what an audit answers
    path = str(tmp_path / "audit.jsonl")
    written = audit.write({"event": "attempting"}, path=path)
    assert written["ts"].endswith("+00:00")
    assert audit.read_all(path)[0]["ts"] == written["ts"]


def test_the_caller_cannot_forge_the_timestamp(tmp_path):
    # Given:    a record that supplies its own ts
    # Expected: the written ts is the real one, not the supplied one
    # Why:      a record that lies about its own time is worse than no record
    path = str(tmp_path / "audit.jsonl")
    written = audit.write({"ts": "1999-01-01T00:00:00+00:00", "event": "x"}, path=path)
    assert not written["ts"].startswith("1999")


def test_writes_append_and_keep_their_order(tmp_path):
    # Given:    three records written one after another
    # Expected: three lines, oldest first
    # Why:      an audit that overwrites is a log of the last thing only
    path = str(tmp_path / "audit.jsonl")
    for step in ("one", "two", "three"):
        audit.write({"event": step}, path=path)
    assert [record["event"] for record in audit.read_all(path)] == ["one", "two", "three"]


def test_an_awkward_value_does_not_stop_the_write(tmp_path):
    # Given:    a record holding an object json cannot serialise
    # Expected: it is written as text, not raised
    # Why:      failing to record an action because a value was odd is the worst outcome
    path = str(tmp_path / "audit.jsonl")

    class Odd:
        def __str__(self) -> str:
            return "an odd object"

    audit.write({"event": "attempting", "decision": Odd()}, path=path)
    assert audit.read_all(path)[0]["decision"] == "an odd object"


def test_a_truncated_last_line_does_not_hide_the_good_ones(tmp_path):
    # Given:    two complete records, then a half-written third
    # Expected: read_all returns the two complete ones
    # Why:      a crash mid-write looks exactly like this, and the earlier
    #           records are the ones you came to read
    path = str(tmp_path / "audit.jsonl")
    audit.write({"event": "one"}, path=path)
    audit.write({"event": "two"}, path=path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"event": "thr')  # the process died here
    assert [record["event"] for record in audit.read_all(path)] == ["one", "two"]


def test_a_missing_file_is_an_empty_history(tmp_path):
    # Given:    a path that does not exist
    # Expected: an empty list, not an exception
    # Why:      the first run of a fresh install has no trail yet
    assert audit.read_all(str(tmp_path / "nothing.jsonl")) == []


def test_the_file_is_plain_readable_jsonl(tmp_path):
    # Given:    two records
    # Expected: the raw file is two lines, each parsing on its own
    # Why:      the format has to be greppable and tailable during an incident
    path = str(tmp_path / "audit.jsonl")
    audit.write({"event": "one"}, path=path)
    audit.write({"event": "two"}, path=path)
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["event"] == "two"
