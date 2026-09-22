"""
what: the two data shapes that travel through the graph -- Incident, produced by
      our own detectors, and AgentDecision, produced by a language model.
why:  they sit on opposite sides of a trust boundary, and this file exists to
      make that boundary visible. Data that Python computed is trustworthy and
      needs no validation. Data that a model emitted is untrusted text that
      merely looks like JSON. Two shapes, two tools, one file, on purpose.
how:  Incident is a frozen dataclass: cheap, immutable, unvalidated, because a
      detector cannot hand us a confidence of 1.7. AgentDecision is a Pydantic
      model, because an LLM can and eventually will.

      Note what AgentDecision does NOT do: it never decides whether an action is
      *permitted*. "Well-formed" and "allowed" are separate questions answered by
      separate layers -- the allowlist in agent/tools/registry.py answers the
      second one, in Stage 1. Merging them is the mistake this MVP exists to
      teach you not to make.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True)
class Incident:
    """One thing a detector believes is wrong with one container.

    frozen=True because an incident is a finding, not a workspace: once a
    detector has made its call, no downstream node should be able to quietly
    edit the evidence it was based on.
    """

    kind: str  # the runbook that covers it, e.g. "RB-002"
    container: str
    summary: str  # one plain sentence, for the trace and the prompt
    # The raw numbers the detector reasoned from. Kept so the prompt, the audit
    # record and a human reading the trace all see the same evidence.
    # (frozen protects the field, not the dict's contents -- nobody mutates it.)
    evidence: dict = field(default_factory=dict)


class AgentDecision(BaseModel):
    """One step of the model's reasoning: either call a tool, or conclude.

    This is the only Pydantic model in the project, because this is the only
    place where data crosses in from outside our control.
    """

    # extra="forbid" is a deliberate choice. Pydantic's default is to silently
    # drop fields it does not recognise; we would rather a model that invents
    # "priority": "high" fail loudly and get retried, because a silently ignored
    # field is a decision you think was made and wasn't.
    model_config = ConfigDict(extra="forbid")

    action: Literal["use_tool", "conclude"]
    tool: str | None = None
    args: dict = Field(default_factory=dict)
    diagnosis: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)  # a model will one day return 1.7
    reasoning: str

    @model_validator(mode="after")
    def check_fields_match_the_action(self) -> "AgentDecision":
        """Cross-field rules: each action requires its own payload.

        Field-by-field validation cannot catch action="use_tool" with no tool
        name -- every individual field is fine, the combination is not. Without
        this, that response reaches the dispatcher as tool=None and fails there
        instead, far from the cause.
        """
        if self.action == "use_tool" and not self.tool:
            raise ValueError("action='use_tool' requires a tool name")
        if self.action == "conclude" and not self.diagnosis:
            raise ValueError("action='conclude' requires a diagnosis")
        return self
