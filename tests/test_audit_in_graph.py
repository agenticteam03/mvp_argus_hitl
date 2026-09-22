"""
what: tests that the graph audits an action BEFORE it takes it.
why:  rule 4 is an ordering claim, and ordering is exactly what a comment cannot
      prove. These tests read the file back and check WHICH LINE CAME FIRST,
      including the case that matters most: a tool that blows up mid-call.
how:  the autouse fixture in conftest.py points the audit at a temporary file,
      and each test reads it with audit.read_all().
"""

import json

import pytest

from agent import audit
from agent.adapters.llm_mock import DEFAULT_SCRIPT, MockLLM
from agent.tools.registry import REGISTRY
from tests.test_graph_mock import FORBIDDEN_REQUEST, run


def events(path: str) -> list[str]:
    """Just the event names, in the order they were written."""
    return [record["event"] for record in audit.read_all(path)]


def test_an_action_is_audited_before_it_runs(audit_to_tmp):
    # Given:    a normal run, which calls get_container_stats once
    # Expected: "attempting" is written, then "completed"
    # Why:      rule 4 as an ordering test -- after-the-fact logging would only
    #           ever record the actions that finished
    run()
    assert events(audit_to_tmp) == ["attempting", "completed"]


def test_the_attempt_survives_a_tool_that_explodes(audit_to_tmp, monkeypatch):
    # Given:    a tool that raises something the graph does not catch
    # Expected: the run dies, and "attempting" is still on disk, alone
    # Why:      THE rule-4 test. An action that crashes mid-flight must still
    #           leave a trace, or the only unexplained actions are the dangerous ones
    def explodes(name: str) -> str:
        raise RuntimeError("the daemon went away mid-call")

    monkeypatch.setitem(REGISTRY, "get_container_stats", explodes)
    with pytest.raises(RuntimeError):
        run()
    assert events(audit_to_tmp) == ["attempting"]


def test_a_refusal_is_recorded_without_an_attempt(audit_to_tmp):
    # Given:    a model asking to restart postgres, then concluding
    # Expected: ONE record, "refused", naming the tool, the args and the reason
    # Why:      from Stage 5 the allowlist is checked before anything else, so a
    #           forbidden call is never "attempted" -- and no human is woken to
    #           approve something the code was always going to reject. The
    #           refusal record still carries everything a review needs.
    run(llm=MockLLM([FORBIDDEN_REQUEST, DEFAULT_SCRIPT[1]]))
    records = audit.read_all(audit_to_tmp)
    assert events(audit_to_tmp) == ["refused"]
    assert records[0]["tool"] == "restart_container"
    assert records[0]["args"] == {"name": "postgres"}
    assert records[0]["refused_by"] == "allowlist"
    assert "lab-" in records[0]["reason"]


def test_the_record_says_why_not_just_what(audit_to_tmp):
    # Given:    a normal run
    # Expected: the attempt carries confidence, reasoning and the step number
    # Why:      "the container restarted" is Docker's fact; "the agent restarted
    #           it because X, at confidence 0.6" is only knowable from here
    run()
    attempt = audit.read_all(audit_to_tmp)[0]
    assert attempt["confidence"] == 0.6
    assert "Restart count is climbing" in attempt["reasoning"]
    assert attempt["step"] == 0
    assert attempt["container"] == "lab-victim"


def test_a_read_is_marked_as_not_a_write(audit_to_tmp):
    # Given:    a run whose only tool call is get_container_stats
    # Expected: the attempt carries write=False and approval "not required"
    # Why:      one flag separates "what did it look at" from "what did it
    #           change"; the write=True side is tested in test_human_gate.py,
    #           because from Stage 5 a real write needs an approval to get there
    run()
    attempt = audit.read_all(audit_to_tmp)[0]
    assert attempt["write"] is False
    assert attempt["approval"] == "not required"


def test_the_trail_is_still_one_json_object_per_line(audit_to_tmp):
    # Given:    a full run through the graph
    # Expected: every line parses on its own
    # Why:      the format has to survive `tail -f` and `grep` during an incident
    run()
    for line in open(audit_to_tmp, encoding="utf-8").read().strip().splitlines():
        assert json.loads(line)["ts"]
