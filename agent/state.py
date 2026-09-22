"""
what: AgentState -- the graph's working memory, and the only thing that moves
      between nodes.
why:  in LangGraph a node is a pure-ish function of the form
      state -> partial state update. Nothing else is shared: no globals, no
      instance attributes, no hidden context. Writing that contract down in one
      TypedDict is what makes the loop in graph.py readable, and what makes the
      whole run reconstructable from a checkpoint in Stage 5.
how:  a TypedDict rather than a Pydantic model, because this data never crosses
      the model boundary -- it is ours, and validating it on every node return
      would cost time and teach nothing.

      Nodes return PARTIAL dicts (just the keys they changed) and LangGraph
      merges them in. How it merges depends on the annotation:
        - a plain field is OVERWRITTEN by the returned value (last write wins);
        - a field annotated with a reducer is COMBINED using that function.
      `history` uses operator.add so that returning {"history": ["..."]} appends
      one line to the trace instead of erasing everything written before it.
      That single annotation is the difference between a log and a bug.
"""

import operator
from typing import Annotated, TypedDict

from agent.models import AgentDecision, Incident


class AgentState(TypedDict):
    """Everything the agent knows at one point in the run."""

    # --- set once, at the start of the run -------------------------------
    container: str  # the container under investigation

    # --- filled in by nodes ----------------------------------------------
    metrics: dict  # whatever MetricsPort returned, unmodified
    incident: Incident | None  # what detect_node concluded; None means healthy
    prior_incidents: int  # from MemoryPort; 0 until Stage 4 makes it real
    docs: list[str]  # from DocsPort; [] until Stage 4 makes it real
    decision: AgentDecision | None  # the model's latest validated proposal

    # --- the loop --------------------------------------------------------
    # Appended to, never replaced: every node adds its own line, and the result
    # is the human-readable trace that run.py prints at the end.
    history: Annotated[list[str], operator.add]

    # NOT annotated, so it is overwritten. Both act_node and reason_node return
    # steps + 1 -- it counts every attempt that could repeat, a tool call or a
    # reply that failed validation -- and the router compares it to MAX_STEPS. This counter lives in state and is
    # checked in Python -- it is the hard bound on the reasoning loop, and no
    # prompt wording can talk its way past it.
    steps: int
