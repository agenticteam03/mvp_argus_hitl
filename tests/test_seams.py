"""
what: tests for the two seams written in Step 2 -- Pydantic validation of a
      model response, and structural typing of a port.
why:  these two mechanisms are invisible until something violates them. The
      tests below are the violations, written down once, so that "Pydantic
      catches it" and "an adapter needs no import to satisfy a port" stop being
      claims and become facts you have watched pass.
how:  every test constructs the bad case on purpose and asserts the failure.
      No network, no API key, no cost -- and that is permanent: pytest must run
      offline and free for the entire life of this project.
"""

import json

import pytest
from pydantic import ValidationError

from agent.models import AgentDecision, Incident
from agent.ports import MetricsPort

# A well-formed response, as raw text, because raw text is what LLMPort returns.
GOOD_RESPONSE = json.dumps(
    {
        "action": "use_tool",
        "tool": "get_container_stats",
        "args": {"name": "lab-victim"},
        "confidence": 0.8,
        "reasoning": "Need current memory usage before diagnosing.",
    }
)


def test_well_formed_response_parses():
    # Given:    a valid use_tool response, as raw JSON text
    # Expected: an AgentDecision carrying the same tool and args
    # Why:      the happy path of the model boundary -- text in, typed object out
    decision = AgentDecision.model_validate_json(GOOD_RESPONSE)
    assert decision.action == "use_tool"
    assert decision.tool == "get_container_stats"
    assert decision.args["name"] == "lab-victim"


def test_confidence_above_one_is_rejected():
    # Given:    a conclude response with confidence 1.7
    # Expected: ValidationError (its message is printed, to be read)
    # Why:      models invent numbers; Field(le=1.0) stops one before anything acts on it
    bad = json.dumps(
        {"action": "conclude", "diagnosis": "OOM", "confidence": 1.7, "reasoning": "sure"}
    )
    with pytest.raises(ValidationError) as err:
        AgentDecision.model_validate_json(bad)
    # In Stage 3 this exact text is fed back to the model as an observation.
    print("\nconfidence 1.7 rejected:\n", err.value)


def test_invented_field_is_rejected():
    # Given:    an otherwise valid response with an extra "priority" field
    # Expected: ValidationError
    # Why:      extra="forbid" -- a silently dropped field is a decision you think was made
    bad = json.dumps(
        {
            "action": "conclude",
            "diagnosis": "OOM loop",
            "confidence": 0.9,
            "reasoning": "restart count climbing",
            "priority": "high",  # no such field
        }
    )
    with pytest.raises(ValidationError):
        AgentDecision.model_validate_json(bad)


def test_use_tool_without_a_tool_name_is_rejected():
    # Given:    action "use_tool" with no tool name
    # Expected: ValidationError
    # Why:      every field is fine alone; only the cross-field validator sees the whole is not
    bad = json.dumps({"action": "use_tool", "confidence": 0.5, "reasoning": "let me look"})
    with pytest.raises(ValidationError):
        AgentDecision.model_validate_json(bad)


def test_incident_is_immutable():
    # Given:    an Incident
    # Expected: changing its container raises
    # Why:      frozen=True -- no node may edit a detector's finding after the fact
    incident = Incident(kind="RB-002", container="lab-victim", summary="OOM loop")
    with pytest.raises(Exception):  # dataclasses raise FrozenInstanceError
        incident.container = "postgres"  # type: ignore[misc]


class ThingWithStats:
    """Deliberately imports nothing from agent.ports and inherits from nothing."""

    def container_stats(self, name: str) -> dict:
        return {"name": name}


class ThingWithoutStats:
    def something_else(self) -> None:  # pragma: no cover - never called
        pass


def test_a_port_is_satisfied_by_shape_not_by_inheritance():
    # Given:    a class with container_stats that never imports or inherits MetricsPort
    # Expected: isinstance(..., MetricsPort) is True
    # Why:      structural typing -- the reason swapping the metrics source is one line in run.py
    assert isinstance(ThingWithStats(), MetricsPort)


def test_missing_method_does_not_satisfy_the_port():
    # Given:    a class without container_stats
    # Expected: isinstance(..., MetricsPort) is False
    # Why:      proves the check above is real, not something that always says yes
    assert not isinstance(ThingWithoutStats(), MetricsPort)
