"""
what: tests for the human gate -- the pause before any write.
why:  Stage 3's real run restarted a container on the model's own initiative.
      The allowlist permitted it, correctly: lab-victim IS a lab container. What
      was missing is a person. These tests pin the three outcomes -- approved,
      denied, and never asked -- and the fourth, quieter claim: that a paused
      run has not done anything yet.
how:  an InMemorySaver stands in for the durable checkpointer (Step 4 uses
      SQLite), and a thread_id names the run so it can be resumed. The first
      invoke returns with "__interrupt__" in it; the second resumes with
      Command(resume="approve" | "deny").
"""

import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent import audit
from agent.adapters.docs_null import NullDocs
from agent.adapters.llm_mock import DEFAULT_SCRIPT, MockLLM
from agent.adapters.memory_null import NullMemory
from agent.adapters.metrics_fake import FakeMetrics
from agent.graph import build_graph, initial_state

# A write the allowlist PERMITS: lab-victim matches lab-*. The gate is the only
# thing between this request and a real container restarting.
ALLOWED_WRITE = json.dumps(
    {
        "action": "use_tool",
        "tool": "restart_container",
        "args": {"name": "lab-victim"},
        "confidence": 0.9,
        "reasoning": "Memory is at the limit; restart to clear the immediate pressure.",
    }
)

# The same request aimed at a container the allowlist forbids.
FORBIDDEN_WRITE = json.dumps(
    {
        "action": "use_tool",
        "tool": "restart_container",
        "args": {"name": "postgres"},
        "confidence": 0.7,
        "reasoning": "Restarting the database will clear the memory pressure.",
    }
)


def gated_graph(script):
    """A graph with a checkpointer, plus the config that names this run."""
    graph = build_graph(
        FakeMetrics(),
        MockLLM(script),
        NullDocs(),
        NullMemory(),
        checkpointer=InMemorySaver(),
    )
    return graph, {"configurable": {"thread_id": "test-run"}}


def events(path: str) -> list[str]:
    return [record["event"] for record in audit.read_all(path)]


def test_a_read_is_never_gated(audit_to_tmp):
    # Given:    the default script, whose only tool call is a read
    # Expected: the run finishes in one invoke, with no interrupt
    # Why:      an agent that asks permission to LOOK at something is useless
    graph, config = gated_graph(DEFAULT_SCRIPT)
    final = graph.invoke(initial_state("lab-victim"), config)
    assert "__interrupt__" not in final
    assert events(audit_to_tmp) == ["attempting", "completed"]


def test_a_write_pauses_before_anything_happens(audit_to_tmp):
    # Given:    a model asking to restart lab-victim, which IS allowed
    # Expected: the run stops with an interrupt, and the audit is still EMPTY
    # Why:      the strongest claim in this stage -- a paused run has not acted.
    #           Nothing above interrupt() may have side effects, because the node
    #           runs again from the top when the run resumes.
    graph, config = gated_graph([ALLOWED_WRITE, DEFAULT_SCRIPT[1]])
    paused = graph.invoke(initial_state("lab-victim"), config)
    assert "__interrupt__" in paused
    assert audit.read_all(audit_to_tmp) == []


def test_the_pause_tells_the_human_what_and_why(audit_to_tmp):
    # Given:    a paused write
    # Expected: the payload carries the tool, the target, the confidence and the
    #           model's own reasoning
    # Why:      a prompt that says "approve y/n" with no context trains people to
    #           say yes; this is the difference between a gate and a formality
    graph, config = gated_graph([ALLOWED_WRITE, DEFAULT_SCRIPT[1]])
    paused = graph.invoke(initial_state("lab-victim"), config)
    payload = paused["__interrupt__"][0].value
    assert payload["tool"] == "restart_container"
    assert payload["args"] == {"name": "lab-victim"}
    assert payload["confidence"] == 0.9
    assert "clear the immediate pressure" in payload["reasoning"]


def test_approving_lets_the_action_run(audit_to_tmp):
    # Given:    a paused write, resumed with "approve"
    # Expected: attempting (approval recorded) then completed, and the run ends
    # Why:      the gate has to be passable, or the agent can never act at all
    graph, config = gated_graph([ALLOWED_WRITE, DEFAULT_SCRIPT[1]])
    graph.invoke(initial_state("lab-victim"), config)
    final = graph.invoke(Command(resume="approve"), config)

    assert events(audit_to_tmp) == ["attempting", "completed"]
    assert audit.read_all(audit_to_tmp)[0]["approval"] == "approve"
    assert audit.read_all(audit_to_tmp)[0]["write"] is True
    assert any("restarted lab-victim" in line for line in final["history"])


def test_denying_stops_the_action_and_says_so(audit_to_tmp):
    # Given:    the same paused write, resumed with "deny"
    # Expected: attempting then denied, NO completed, and the trace says DENIED
    # Why:      mvp_plan.md's "done when": audit.jsonl contains both, including
    #           the denial. A denial that left no record would be untraceable.
    graph, config = gated_graph([ALLOWED_WRITE, DEFAULT_SCRIPT[1]])
    graph.invoke(initial_state("lab-victim"), config)
    final = graph.invoke(Command(resume="deny"), config)

    assert events(audit_to_tmp) == ["attempting", "denied"]
    denial = audit.read_all(audit_to_tmp)[1]
    assert denial["refused_by"] == "human"
    assert denial["answer"] == "deny"
    assert any("DENIED by a human" in line for line in final["history"])
    # The container was never touched: no completion, no observation from the tool.
    assert not any("restarted lab-victim" in line for line in final["history"])


def test_the_model_reads_the_denial_and_carries_on(audit_to_tmp):
    # Given:    a denied write, followed by the model's next decision
    # Expected: the run reaches a conclusion rather than stopping dead
    # Why:      a denial is an observation like any other: the model should be
    #           able to say "then I cannot fix it" instead of crashing
    graph, config = gated_graph([ALLOWED_WRITE, DEFAULT_SCRIPT[1]])
    graph.invoke(initial_state("lab-victim"), config)
    final = graph.invoke(Command(resume="deny"), config)
    assert final["decision"].action == "conclude"


def test_a_forbidden_write_never_reaches_a_human(audit_to_tmp):
    # Given:    a model asking to restart postgres
    # Expected: no interrupt at all, one "refused" record, run finishes
    # Why:      the allowlist decides first. Waking someone to approve a call the
    #           code will reject anyway teaches them the prompt is noise.
    graph, config = gated_graph([FORBIDDEN_WRITE, DEFAULT_SCRIPT[1]])
    final = graph.invoke(initial_state("lab-victim"), config)
    assert "__interrupt__" not in final
    assert events(audit_to_tmp) == ["refused"]
