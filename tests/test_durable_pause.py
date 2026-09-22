"""
what: tests that a paused run survives the process that started it.
why:  Step 3 proved a write waits for a human. It waited in memory, so "wait"
      meant "wait until this terminal closes" -- useless for a real approval,
      which arrives minutes or hours later, often from somebody else.
how:  each test builds a graph, pauses it, then throws that graph away and
      builds a COMPLETELY NEW one over the same database file. Everything the
      second graph knows about the run it has to read off the disk. That is a
      process boundary in all the ways that matter here, minus the killing.
"""

import json

from langgraph.types import Command

from agent import audit
from agent.adapters.docs_null import NullDocs
from agent.adapters.llm_mock import DEFAULT_SCRIPT, MockLLM
from agent.adapters.memory_null import NullMemory
from agent.adapters.metrics_fake import FakeMetrics
from agent.checkpoint import sqlite_checkpointer, start_fresh
from agent.graph import build_graph, initial_state
from agent.models import AgentDecision, Incident

ALLOWED_WRITE = json.dumps(
    {
        "action": "use_tool",
        "tool": "restart_container",
        "args": {"name": "lab-victim"},
        "confidence": 0.9,
        "reasoning": "Memory is at the limit; restart to clear the immediate pressure.",
    }
)


def graph_over(db_path: str):
    """A brand-new graph, sharing only the checkpoint file on disk."""
    return build_graph(
        FakeMetrics(),
        MockLLM([ALLOWED_WRITE, DEFAULT_SCRIPT[1]]),
        NullDocs(),
        NullMemory(),
        checkpointer=sqlite_checkpointer(db_path),
    )


def config_for(thread: str = "lab-victim") -> dict:
    return {"configurable": {"thread_id": thread}}


def test_a_pause_is_written_to_disk(tmp_path):
    # Given:    a run paused at the gate
    # Expected: the checkpoint file exists and has content
    # Why:      "paused" has to mean something on disk, or the process owns the run
    db = str(tmp_path / "checkpoints.db")
    graph_over(db).invoke(initial_state("lab-victim"), config_for())
    assert (tmp_path / "checkpoints.db").stat().st_size > 0


def test_a_new_graph_can_see_what_the_old_one_was_waiting_for(tmp_path):
    # Given:    a paused run, then a graph object built from scratch
    # Expected: get_state().next names the node that is waiting -- 'act'
    # Why:      whoever answers needs to find the pending run without the
    #           original process being alive to tell them
    db = str(tmp_path / "checkpoints.db")
    graph_over(db).invoke(initial_state("lab-victim"), config_for())
    assert graph_over(db).get_state(config_for()).next == ("act",)


def test_a_new_graph_can_approve_what_the_old_one_paused(audit_to_tmp, tmp_path):
    # Given:    a run paused by one graph, resumed by a different one
    # Expected: the action runs, and the trail shows the approval
    # Why:      THE Stage 5 claim: kill the process, answer later, the run continues
    db = str(tmp_path / "checkpoints.db")
    graph_over(db).invoke(initial_state("lab-victim"), config_for())
    assert audit.read_all(audit_to_tmp) == []  # nothing done yet

    final = graph_over(db).invoke(Command(resume="approve"), config_for())

    assert [r["event"] for r in audit.read_all(audit_to_tmp)] == ["attempting", "completed"]
    assert any("restarted lab-victim" in line for line in final["history"])


def test_a_new_graph_can_deny_what_the_old_one_paused(audit_to_tmp, tmp_path):
    # Given:    the same pause, answered "deny" by a different graph
    # Expected: a denial in the trail and nothing executed
    # Why:      mvp_plan.md's "done when": approve one run, deny another, and
    #           audit.jsonl contains both
    db = str(tmp_path / "checkpoints.db")
    graph_over(db).invoke(initial_state("lab-victim"), config_for())
    final = graph_over(db).invoke(Command(resume="deny"), config_for())

    assert [r["event"] for r in audit.read_all(audit_to_tmp)] == ["attempting", "denied"]
    assert not any("restarted lab-victim" in line for line in final["history"])


def test_the_state_comes_back_as_the_types_it_went_in_as(tmp_path):
    # Given:    a run whose state holds an Incident and an AgentDecision
    # Expected: after a reload they are still those classes, not dicts
    # Why:      the checkpoint serialiser has an allowlist of types; if ours were
    #           missing, LangGraph would refuse them and the resumed run would
    #           crash on `decision.tool`
    db = str(tmp_path / "checkpoints.db")
    graph_over(db).invoke(initial_state("lab-victim"), config_for())
    values = graph_over(db).get_state(config_for()).values
    assert isinstance(values["incident"], Incident)
    assert isinstance(values["decision"], AgentDecision)


def test_two_containers_do_not_share_a_pause(tmp_path):
    # Given:    two runs on the same file under different thread ids
    # Expected: each keeps its own pending state
    # Why:      thread_id is what makes "the paused run for lab-victim" a thing
    #           you can name; sharing one would let an answer resume the wrong run
    db = str(tmp_path / "checkpoints.db")
    graph_over(db).invoke(initial_state("lab-victim"), config_for("lab-victim"))
    assert graph_over(db).get_state(config_for("lab-other")).next == ()
    assert graph_over(db).get_state(config_for("lab-victim")).next == ("act",)


def test_a_reused_thread_would_inherit_the_previous_run(tmp_path):
    # Given:    three runs on ONE thread id, with nothing cleared between them
    # Expected: the history grows 6 -> 12 -> 18 lines
    # Why:      this documents the hazard rather than the fix. `history` has an
    #           operator.add reducer, and build_prompt sends history to the
    #           model, so without start_fresh every prompt carries every earlier
    #           investigation -- a bill that grows with how often you run it.
    db = str(tmp_path / "reused.db")
    lengths = []
    for _ in range(3):
        final = graph_over(db).invoke(initial_state("lab-victim"), config_for())
        graph_over(db).invoke(Command(resume="deny"), config_for())
        lengths.append(len([line for line in final["history"] if line.startswith("DETECT")]))
    assert lengths == [1, 2, 3]


def test_start_fresh_keeps_one_run_to_one_investigation(tmp_path):
    # Given:    the same three runs, each calling start_fresh first
    # Expected: one DETECT block every time
    # Why:      the fix for the hazard above, and the reason run.py checks for a
    #           pending answer BEFORE clearing anything
    db = str(tmp_path / "cleared.db")
    checkpointer = sqlite_checkpointer(db)
    blocks = []
    for _ in range(3):
        start_fresh(checkpointer, "lab-victim")
        final = graph_over(db).invoke(initial_state("lab-victim"), config_for())
        graph_over(db).invoke(Command(resume="deny"), config_for())
        blocks.append(len([line for line in final["history"] if line.startswith("DETECT")]))
    assert blocks == [1, 1, 1]
