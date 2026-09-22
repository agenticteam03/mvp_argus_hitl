"""
what: tests that the four Stage 0 adapters satisfy their ports and behave.
why:  two claims are worth pinning down here. First, that each fake really does
      fit its port -- if it does not, the "one-line swap" promised in Stage 2
      turns into an afternoon. Second, and more important, that MockLLM's output
      survives the SAME Pydantic validation a real provider's output will face.
      A mock that is easier to satisfy than production hides the very bug it was
      written to catch.
how:  isinstance() against the runtime_checkable Protocols for shape, then
      behavioural assertions. No Docker, no network, no key.
"""

from agent.adapters.docs_null import NullDocs
from agent.adapters.llm_mock import DEFAULT_SCRIPT, MockLLM
from agent.adapters.memory_null import NullMemory
from agent.adapters.metrics_fake import HEALTHY_VICTIM, OOM_VICTIM, FakeMetrics
from agent.detectors import detect
from agent.models import AgentDecision
from agent.ports import DocsPort, LLMPort, MemoryPort, MetricsPort

# The keys agent/ports.py says every metrics adapter must return. Stage 1 and
# Stage 2 adapters will be held to this same set.
CONTRACT_KEYS = {
    "name",
    "running",
    "restart_count",
    "last_exit_code",
    "oom_killed",
    "memory_usage_bytes",
    "memory_limit_bytes",
}


def test_every_adapter_satisfies_its_port():
    # Given:    one instance of each Stage 0 adapter
    # Expected: each passes isinstance() against its own port
    # Why:      an adapter that does not fit its port breaks the one-line swap
    assert isinstance(FakeMetrics(), MetricsPort)
    assert isinstance(MockLLM(), LLMPort)
    assert isinstance(NullDocs(), DocsPort)
    assert isinstance(NullMemory(), MemoryPort)


def test_fake_metrics_honours_the_contract():
    # Given:    FakeMetrics' stats for lab-victim
    # Expected: exactly the seven keys the MetricsPort contract lists
    # Why:      the Docker and Prometheus adapters will be held to the same keys
    stats = FakeMetrics().container_stats("lab-victim")
    assert set(stats) == CONTRACT_KEYS


def test_fake_metrics_feeds_the_detector_an_incident():
    # Given:    FakeMetrics loaded with the OOM fixture
    # Expected: detect() returns an RB-002 incident
    # Why:      the first test of two components wired together
    incident = detect(FakeMetrics(OOM_VICTIM).container_stats("lab-victim"))
    assert incident is not None
    assert incident.kind == "RB-002"


def test_the_healthy_fixture_produces_no_incident():
    # Given:    FakeMetrics loaded with the healthy fixture
    # Expected: None
    # Why:      proves the detector tells cases apart instead of always saying yes
    assert detect(FakeMetrics(HEALTHY_VICTIM).container_stats("lab-victim")) is None


def test_mock_llm_output_passes_the_same_validation_as_a_real_model():
    # Given:    every response in MockLLM's script
    # Expected: each one parses into an AgentDecision
    # Why:      a mock easier to satisfy than production hides the bug it exists to catch
    for response in DEFAULT_SCRIPT:
        decision = AgentDecision.model_validate_json(response)
        assert decision.reasoning


def test_mock_llm_looks_first_then_concludes():
    # Given:    a fresh MockLLM, called twice
    # Expected: first use_tool get_container_stats, then conclude; 2 calls counted
    # Why:      the fixed script the whole Stage 0 trace depends on
    llm = MockLLM()
    first = AgentDecision.model_validate_json(llm.decide("prompt 1"))
    second = AgentDecision.model_validate_json(llm.decide("prompt 2"))
    assert first.action == "use_tool"
    assert first.tool == "get_container_stats"
    assert second.action == "conclude"
    assert llm.calls == 2


def test_mock_llm_repeats_its_last_line_forever():
    # Given:    a one-entry script (always use_tool), called 5 times
    # Expected: use_tool every time
    # Why:      a "model" that never concludes -- exactly what the MAX_STEPS test must stop
    llm = MockLLM(DEFAULT_SCRIPT[:1])
    actions = [AgentDecision.model_validate_json(llm.decide("p")).action for _ in range(5)]
    assert actions == ["use_tool"] * 5


def test_mock_llm_keeps_the_prompts_it_was_given():
    # Given:    one call with a prompt that mentions RB-002
    # Expected: that prompt is stored in llm.prompts
    # Why:      lets the graph tests check what actually reached the model
    llm = MockLLM()
    llm.decide("incident: RB-002 on lab-victim")
    assert "RB-002" in llm.prompts[0]


def test_null_memory_forgets_nothing_within_a_process_and_everything_between():
    # Given:    one recorded RB-002 incident, then a brand-new NullMemory
    # Expected: 1 match for RB-002, 0 for RB-001, 0 in the new instance
    # Why:      memory that dies with the process -- the gap Stage 4's SQLite adapter closes
    memory = NullMemory()
    assert memory.recent_incidents("RB-002", hours=24) == []
    memory.record({"kind": "RB-002", "container": "lab-victim"})
    assert len(memory.recent_incidents("RB-002", hours=24)) == 1
    assert memory.recent_incidents("RB-001", hours=24) == []  # the kind filter is real
    assert NullMemory().recent_incidents("RB-002", hours=24) == []  # a "new process" knows nothing


def test_null_docs_returns_nothing_and_callers_must_cope():
    # Given:    any query
    # Expected: an empty list
    # Why:      real retrievers return [] too, so the graph must handle it from day one
    assert NullDocs().search("why was lab-victim killed", k=3) == []
