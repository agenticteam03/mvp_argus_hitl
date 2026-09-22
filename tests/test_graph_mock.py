"""
what: the agent loop, run end to end against the Stage 0 fakes.
why:  these test the loop's SHAPE -- that it detects, reasons, acts, feeds the
      observation back into the next prompt, concludes, and above all that it
      STOPS. None of them care whether the diagnosis is clever; that is Stage 3's
      problem. All offline, all free, all deterministic, because MockLLM is the
      default in every test (rule 5).
how:  run() builds a graph from fresh adapters and invokes it once. Each test
      then asserts on the final state, on how often MockLLM was consulted, or on
      the history lines -- whose first word names the node that wrote them.
"""

import json

from agent.adapters.docs_null import NullDocs
from agent.adapters.llm_mock import DEFAULT_SCRIPT, MockLLM
from agent.adapters.memory_null import NullMemory
from agent.adapters.metrics_fake import HEALTHY_VICTIM, FakeMetrics
from agent.graph import MAX_STEPS, build_graph, initial_state, route
from agent.models import AgentDecision


def run(llm=None, stats=None, memory=None):
    """One full graph run on lab-victim. Returns (final_state, llm, memory)."""
    if llm is None:
        llm = MockLLM()
    if memory is None:
        memory = NullMemory()
    graph = build_graph(FakeMetrics(stats), llm, NullDocs(), memory)
    final = graph.invoke(initial_state("lab-victim"))
    return final, llm, memory


def nodes(history: list[str]) -> list[str]:
    """The first word of each trace line: the node that wrote it."""
    return [line.split()[0] for line in history]


def test_full_run_follows_the_stage_0_trace():
    # Given:    the default fakes -- an OOM loop, and a look-then-conclude script
    # Expected: DETECT RECALL REASON ACT REASON RECORD; 1 tool call, 2 model calls
    # Why:      mvp_plan.md's "done when" for Stage 0, as an assertion
    final, llm, _ = run()
    assert nodes(final["history"]) == ["DETECT", "RECALL", "REASON", "ACT", "REASON", "RECORD"]
    assert final["decision"].action == "conclude"
    assert final["steps"] == 1
    assert llm.calls == 2


def test_graph_stops_at_max_steps():
    # Given:    a MockLLM that asks for a tool forever
    # Expected: MAX_STEPS tool calls, MAX_STEPS + 1 model calls, then a "step bound" RECORD
    # Why:      rule 3 -- Python ends the loop, whatever the model keeps asking for
    looping = MockLLM(DEFAULT_SCRIPT[:1])
    final, llm, _ = run(llm=looping)
    assert final["steps"] == MAX_STEPS
    assert nodes(final["history"]).count("ACT") == MAX_STEPS
    # One more REASON than ACT: the model is asked once more, wants another
    # tool, and route() sends the run to record instead.
    assert llm.calls == MAX_STEPS + 1
    assert "step bound" in final["history"][-1]


def test_router_refuses_to_act_at_the_bound():
    # Given:    a use_tool decision, with steps one below MAX_STEPS and then at it
    # Expected: "act", then "record"
    # Why:      the same bound, tested on route() alone -- no graph needed
    wants_a_tool = AgentDecision.model_validate_json(DEFAULT_SCRIPT[0])
    assert route({"decision": wants_a_tool, "steps": MAX_STEPS - 1}) == "act"  # type: ignore[typeddict-item]
    assert route({"decision": wants_a_tool, "steps": MAX_STEPS}) == "record"  # type: ignore[typeddict-item]


def test_healthy_container_never_consults_the_model():
    # Given:    healthy container stats
    # Expected: no incident, 0 model calls, a single DETECT line
    # Why:      cost control -- no incident means no model call and no spend
    final, llm, _ = run(stats=HEALTHY_VICTIM)
    assert final["incident"] is None
    assert llm.calls == 0
    assert nodes(final["history"]) == ["DETECT"]


def test_the_incident_reaches_the_prompt():
    # Given:    a normal run
    # Expected: the first prompt contains RB-002 and exit code 137
    # Why:      "we put it in the prompt" is a claim worth checking
    _, llm, _ = run()
    assert "RB-002" in llm.prompts[0]
    assert "137" in llm.prompts[0]


def test_the_observation_reaches_the_next_prompt():
    # Given:    a normal run
    # Expected: the tool output is absent from prompt 1 and present in prompt 2
    # Why:      that IS the loop -- what act returns is what reason reads next
    _, llm, _ = run()
    assert "[FAKE] lab-victim" not in llm.prompts[0]
    assert "[FAKE] lab-victim" in llm.prompts[1]


def test_the_run_is_recorded_in_memory():
    # Given:    a normal run with a fresh NullMemory
    # Expected: one RB-002 record for lab-victim, carrying the final diagnosis
    # Why:      record_node writes episodic memory; Stage 4 makes it survive the process
    final, _, memory = run()
    recorded = memory.recent_incidents("RB-002", hours=24)
    assert len(recorded) == 1
    assert recorded[0]["container"] == "lab-victim"
    assert recorded[0]["diagnosis"] == final["decision"].diagnosis


# The same forbidden request the refusal demo in code.bash uses, so the demo and
# the test can never drift apart.
FORBIDDEN_REQUEST = json.dumps(
    {
        "action": "use_tool",
        "tool": "restart_container",
        "args": {"name": "postgres"},
        "confidence": 0.7,
        "reasoning": "Restarting the database will clear the memory pressure.",
    }
)


def test_a_refused_tool_becomes_an_observation_not_a_crash():
    # Given:    a model that asks to restart "postgres", then concludes
    # Expected: the run finishes, the trace holds a REFUSED line, the model sees it
    # Why:      rule 2 blocks the action, and the model is told why instead of the run dying
    scripted = MockLLM([FORBIDDEN_REQUEST, DEFAULT_SCRIPT[1]])
    final, llm, _ = run(llm=scripted)
    assert any(line.startswith("ACT") and "REFUSED" in line for line in final["history"])
    assert "REFUSED" in llm.prompts[1]
    assert final["decision"].action == "conclude"
    assert final["steps"] == 1  # a refused call still costs a step


# Two replies a real model genuinely produces, and Pydantic must refuse both:
# prose wrapped around the JSON, and a decision with a required field missing.
PROSE_AROUND_JSON = (
    "Sure! Here is my decision: "
    '{"action": "conclude", "diagnosis": "OOM", "confidence": 0.9, "reasoning": "RB-002"}'
)
MISSING_REASONING = json.dumps(
    {"action": "conclude", "diagnosis": "OOM restart loop", "confidence": 0.9}
)


def test_a_malformed_reply_is_retried_not_crashed():
    # Given:    a model that answers with prose around its JSON, then correctly
    # Expected: the run survives, an INVALID line appears, and it still concludes
    # Why:      a real model will do this; crashing on it would end the investigation
    scripted = MockLLM([PROSE_AROUND_JSON, DEFAULT_SCRIPT[1]])
    final, llm, _ = run(llm=scripted)
    assert any(line.startswith("INVALID") for line in final["history"])
    assert final["decision"].action == "conclude"
    assert llm.calls == 2
    assert final["steps"] == 1  # the rejected reply cost a step


def test_the_validation_error_reaches_the_next_prompt():
    # Given:    a reply missing the required "reasoning" field
    # Expected: the next prompt names that field
    # Why:      the model can only correct a mistake it is told about
    scripted = MockLLM([MISSING_REASONING, DEFAULT_SCRIPT[1]])
    _, llm, _ = run(llm=scripted)
    assert "INVALID" in llm.prompts[1]
    assert "reasoning" in llm.prompts[1]


def test_a_model_that_never_validates_is_stopped_by_the_bound():
    # Given:    a model whose every reply is malformed
    # Expected: exactly MAX_STEPS attempts, then a RECORD line, no exception
    # Why:      rule 3 again -- retrying must be bounded by the same counter as acting
    scripted = MockLLM([MISSING_REASONING])
    final, llm, _ = run(llm=scripted)
    assert final["steps"] == MAX_STEPS
    assert llm.calls == MAX_STEPS
    assert final["history"][-1].startswith("RECORD")
    assert final["decision"] is None


def test_the_prompt_lists_each_tool_with_its_parameters():
    # Given:    a normal run
    # Expected: the first prompt shows names AND parameters, e.g. lines=50
    # Why:      a real model invented argument names when it was given names alone
    _, llm, _ = run()
    assert "get_container_logs(name" in llm.prompts[0]
    assert "lines" in llm.prompts[0]


def test_a_tool_call_with_wrong_argument_names_is_refused_not_crashed():
    # Given:    a model that calls get_container_logs(container=, tail=), then concludes
    # Expected: a REFUSED observation naming the real parameters, and a conclusion
    # Why:      this exact call came from a real model; it must cost a step, not the run
    wrong_args = json.dumps(
        {
            "action": "use_tool",
            "tool": "get_container_logs",
            "args": {"container": "lab-victim", "tail": 200},
            "confidence": 0.65,
            "reasoning": "Logs may show what allocates before the kill.",
        }
    )
    final, llm, _ = run(llm=MockLLM([wrong_args, DEFAULT_SCRIPT[1]]))
    refusals = [line for line in final["history"] if line.startswith("ACT") and "REFUSED" in line]
    assert refusals and "name" in refusals[0]
    assert final["decision"].action == "conclude"


def test_a_run_with_no_valid_decision_is_still_recorded():
    # Given:    the same never-valid model
    # Expected: one memory row, with no diagnosis
    # Why:      "we failed to reach a conclusion" is itself a fact worth keeping
    _, _, memory = run(llm=MockLLM([MISSING_REASONING]))
    recorded = memory.recent_incidents("RB-002", hours=24)
    assert len(recorded) == 1
    assert recorded[0]["diagnosis"] is None

