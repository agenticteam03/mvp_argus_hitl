"""
what: the agent itself -- five nodes, the edges between them, and the two
      routers that decide where each run goes next.
why:  this file replaces the `while True:` loop in 01_practice/mock_agent.py.
      A hand-written loop hides its control flow inside if / break / continue;
      a graph states it as nodes and edges you can list, draw and test one at a
      time. The payoff arrives in Stage 5: a graph can stop between two nodes,
      save its state, and resume in a different process. A while loop cannot.
how:  build_graph() receives the four adapters as arguments and the nodes close
      over them. This file never imports a concrete adapter -- that is rule 6,
      and it is why run.py can swap FakeMetrics for DockerMetrics without this
      file noticing. The dispatcher, by contrast, IS imported directly, because
      a safety gate must not be swappable (see agent/tools/registry.py).

      Each node returns ONLY the keys it changed; LangGraph merges them into the
      state, appending to `history` because of its reducer (see state.py).

          START
            |
          detect ----- no incident -----> END
            |
          recall
            |
          reason <---------+
            |   ^          |
         route()|          |
          |  |  +----------+     reply did not validate: ask again
          |  |             |
          |  +--> act -----+     use_tool AND steps < MAX_STEPS
          |
          +--> record --> END    conclude, OR the step bound was hit
"""

import inspect
import json

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from agent import audit
from agent.detectors import detect
from agent.models import AgentDecision
from agent.ports import DocsPort, LLMPort, MemoryPort, MetricsPort
from agent.state import AgentState
from agent.tools.registry import REGISTRY, ToolNotAllowed, check, dispatch

# The hard bound on tool calls per run. 01_practice had the same idea as
# `if step > MAX_STEPS: break`; here it is a condition on an edge, checked in
# Python on every pass. LangGraph has its own recursion_limit too, but it is a
# last-resort crash set far too high to protect a budget: measured on the
# installed release, a runaway loop with no bound made 5,003 model calls before
# it fired. That limit is the framework's; this one is ours, tested, and ends
# the run cleanly with a record of what happened.
MAX_STEPS = 5


def initial_state(container: str) -> AgentState:
    """The state before any node has run. Kept in one place so run.py and the
    tests cannot drift apart on what a fresh run looks like."""
    return {
        "container": container,
        "metrics": {},
        "incident": None,
        "prior_incidents": 0,
        "docs": [],
        "decision": None,
        "history": [],
        "steps": 0,
    }


def build_prompt(state: AgentState) -> str:
    """Everything the model is allowed to know, as plain text.

    MockLLM ignores this completely. It is still built and tested now so that
    Stage 3 changes exactly one thing: a model starts reading it.
    """
    incident = state["incident"]
    runbook = "\n\n".join(state["docs"]) if state["docs"] else "(none retrieved)"
    return (
        "You are diagnosing a container incident. Reply with ONE JSON object and nothing else.\n"
        # Generated from the Pydantic model, not typed by hand: the schema the
        # model is shown can never drift from the schema it is validated against.
        f"JSON schema: {json.dumps(AgentDecision.model_json_schema())}\n\n"
        f"INCIDENT {incident.kind} on {incident.container}: {incident.summary}\n"
        f"Evidence: {incident.evidence}\n"
        f"Prior {incident.kind} incidents on this container in the last 24h: {state['prior_incidents']}\n\n"
        # Asking for the citation is asking for PROVENANCE: which paragraph did
        # this conclusion come from? Without it a diagnosis and a guess look
        # identical. Each chunk already carries its [file] tag, so the model has
        # something real to name -- and mvp_plan.md's "done when" for this stage
        # is a diagnosis that cites RB-002.
        f"Runbook excerpts (cite the [file] tag of any section you rely on):\n{runbook}\n\n"
        # The prompt DESCRIBES the tools; registry.py ENFORCES them. Listing a
        # tool here grants nothing -- the same lesson as ALLOWED_TOOLS in
        # 01_practice/llm_reasoner.py.
        f"Tools you may request, with their exact parameters:\n{tool_catalogue()}\n"
        # Informational only. route() enforces the bound whatever the model
        # makes of this line.
        f"Tool calls used: {state['steps']} of {MAX_STEPS}\n\n"
        "Investigation so far:\n" + "\n".join(state["history"]) + "\n"
    )


def tool_catalogue() -> str:
    """One line per tool, with its real signature.

    Read straight from the functions, so the catalogue cannot drift from the
    code. Measured need: given only the NAMES, a real model asked for
    get_container_logs(container=..., tail=...) when the signature is
    (name, lines=50).

    This is information, not permission. registry.py still refuses anything that
    does not fit, whatever the model read here.
    """
    lines = []
    for name in sorted(REGISTRY):
        parameters = inspect.signature(REGISTRY[name]).parameters.values()
        lines.append(f"- {name}({', '.join(str(p) for p in parameters)})")
    return "\n".join(lines)


def route_after_detect(state: AgentState) -> str:
    """No incident, no model call. The cheapest LLM call is the one never made."""
    return "recall" if state["incident"] is not None else "end"


def route(state: AgentState) -> str:
    """The step bound (rule 3). Both conditions, every pass, in Python.

    A model that asks for a tool on its sixth pass is not refused politely in a
    prompt -- it is simply never routed to `act` again.
    """
    decision = state["decision"]
    # The bound first, so every path out of reason is capped by the same number:
    # tool calls, retries after an invalid reply, or any mix of the two.
    if state["steps"] >= MAX_STEPS:
        return "record"
    if decision is None:
        # The last reply did not validate. Ask again -- the error is now in the
        # history, so the next prompt carries it.
        return "reason"
    if decision.action == "use_tool":
        return "act"
    return "record"


def build_graph(
    metrics: MetricsPort,
    llm: LLMPort,
    docs: DocsPort,
    memory: MemoryPort,
    checkpointer=None,
):
    """Wire the five nodes to the four ports and compile the graph.

    checkpointer is what makes the human gate possible: interrupt() has to save
    the state somewhere before it stops, or "pause" would just mean "lose the
    run". None is fine for a graph that never writes -- reads are not gated --
    and act_node raises a clear error if a write is attempted without one.
    """

    def detect_node(state: AgentState) -> dict:
        stats = metrics.container_stats(state["container"])
        incident = detect(stats)
        if incident is None:
            line = f"DETECT   {state['container']}: healthy, nothing to investigate"
        else:
            line = f"DETECT   {incident.kind} | {incident.summary}"
        return {"metrics": stats, "incident": incident, "history": [line]}

    def recall_node(state: AgentState) -> dict:
        incident = state["incident"]
        # The port filters by kind only, exactly as mvp_plan.md defines it;
        # narrowing to this container happens here rather than widening the port.
        prior = [
            i
            for i in memory.recent_incidents(incident.kind, hours=24)
            if i.get("container") == incident.container
        ]
        # k=4, not 3, and the number was measured rather than guessed. Ranked
        # against this incident's summary, RB-002 comes back as:
        #   1 Reproduce safely  2 Confirm  3 Remediation  4 DO NOT
        # At k=3 the DO NOT list -- the one section that contradicts the
        # remediation the model kept proposing -- is left out by one place.
        # Note also what wins first place: the section on how to CREATE this
        # incident, which is useless when diagnosing one. That is corpus noise,
        # and it is the Stage 4 lesson in miniature: the retriever is fine, the
        # corpus is what decides quality.
        chunks = docs.search(incident.summary, k=4)
        line = (
            f"RECALL   {len(prior)} prior {incident.kind} incident(s) on "
            f"{incident.container} in 24h | {len(chunks)} runbook chunk(s)"
        )
        return {"prior_incidents": len(prior), "docs": chunks, "history": [line]}

    def reason_node(state: AgentState) -> dict:
        raw = llm.decide(build_prompt(state))
        # THE TRUST BOUNDARY. `raw` is text from outside our control; after this
        # line it is a validated AgentDecision or an exception. In Stage 0 the
        # exception would crash the run, which is acceptable only because MockLLM
        # never sends bad JSON. Stage 3 catches ValidationError here and feeds the
        # message back as an observation, bounded by the same step counter.
        try:
            decision = AgentDecision.model_validate_json(raw)
        except ValidationError as error:
            # The model wrote something we cannot act on: prose around the JSON,
            # a missing field, a confidence of 1.7. Do not crash, and do not
            # guess what it meant. Hand the error back as an observation -- it
            # goes into history, and history goes into the next prompt -- and
            # charge it a step, so "let me try again" cannot loop forever.
            # Exactly the shape of a refused tool call, one layer up.
            problems = "; ".join(
                f"{'.'.join(str(part) for part in item['loc']) or 'response'}: {item['msg']}"
                for item in error.errors()[:3]  # three is plenty to correct from
            )
            return {
                "decision": None,
                "steps": state["steps"] + 1,
                "history": [
                    f"INVALID  [{state['steps']}/{MAX_STEPS}] the reply did not validate -> {problems}"
                ],
            }
        if decision.action == "use_tool":
            what = f"use_tool {decision.tool} {decision.args}"
        else:
            what = f"conclude: {decision.diagnosis}"
        line = (
            f"REASON   [{state['steps']}/{MAX_STEPS}] {what} "
            f"(confidence {decision.confidence}) | {decision.reasoning}"
        )
        return {"decision": decision, "history": [line]}

    def act_node(state: AgentState) -> dict:
        decision = state["decision"]

        # 1. POLICY FIRST, and check() is pure -- it decides without doing.
        #    A call the allowlist rejects is refused here, before anybody is
        #    woken up: nobody should be asked to approve something the code is
        #    going to refuse anyway.
        try:
            spec = check(decision.tool, decision.args)
        except ToolNotAllowed as refusal:
            # "The agent tried to restart postgres and was stopped" is exactly
            # the line a security review wants, and Docker cannot record it:
            # nothing happened, so nothing happened to be logged anywhere else.
            audit.write(
                {
                    "event": "refused",
                    "container": state["container"],
                    "tool": decision.tool,
                    "args": decision.args,
                    "reason": str(refusal),
                    "refused_by": "allowlist",
                }
            )
            return {
                "steps": state["steps"] + 1,
                "history": [f"ACT      {decision.tool} -> REFUSED: {refusal}"],
            }

        # 2. THE HUMAN GATE -- writes only. Reads flow straight through, because
        #    an agent that asks permission to LOOK at something is useless.
        #
        #    interrupt() saves the state and stops the graph. The run is resumed
        #    later with Command(resume=...), and when it is, THIS NODE RUNS
        #    AGAIN FROM THE TOP: interrupt() then returns the resume value
        #    instead of pausing. So everything above this line must be free of
        #    side effects -- which is why check() is pure and the audit write is
        #    below it, not above.
        approval = "not required"
        if spec.get("write"):
            approval = interrupt(
                {
                    "question": "Approve this write? Resume with 'approve' or 'deny'.",
                    "container": state["container"],
                    "tool": decision.tool,
                    "args": decision.args,
                    "confidence": decision.confidence,
                    # The human deciding needs the model's own reasoning, not
                    # just the command. This is the whole point of the pause.
                    "reasoning": decision.reasoning,
                }
            )

        # 3. RULE 4: the record goes down BEFORE the attempt. If dispatch hangs,
        #    raises something we do not catch, or takes the machine down with it,
        #    the file still says what was about to happen, why, and who allowed
        #    it. Reads are audited too: "what did it look at" is worth answering.
        audit.write(
            {
                "event": "attempting",
                "container": state["container"],
                "tool": decision.tool,
                "args": decision.args,
                "write": bool(spec.get("write")),
                "approval": approval,
                "confidence": decision.confidence,
                # The model's own words -- the field that answers "why on earth
                # did it do that", months later, when nobody remembers.
                "reasoning": decision.reasoning,
                "step": state["steps"],
            }
        )

        if spec.get("write") and approval != "approve":
            # A denial is a decision, and decisions belong in the trail. Note
            # what is NOT here: any call to the tool.
            audit.write(
                {
                    "event": "denied",
                    "container": state["container"],
                    "tool": decision.tool,
                    "args": decision.args,
                    "answer": approval,
                    "refused_by": "human",
                }
            )
            observation = f"DENIED by a human ({approval}): {decision.tool} was not run"
            return {
                "steps": state["steps"] + 1,
                "history": [f"ACT      {decision.tool} -> {observation}"],
            }

        try:
            observation = dispatch(decision.tool, decision.args)
        except ToolNotAllowed as refusal:
            # Belt and braces. dispatch() checks again, and a pause can last
            # long enough for the allowlist to have changed underneath it.
            observation = f"REFUSED: {refusal}"
            audit.write(
                {
                    "event": "refused",
                    "container": state["container"],
                    "tool": decision.tool,
                    "args": decision.args,
                    "reason": str(refusal),
                    "refused_by": "allowlist",
                }
            )
        else:
            audit.write(
                {
                    "event": "completed",
                    "container": state["container"],
                    "tool": decision.tool,
                    "observation": observation,
                }
            )
        return {
            # Incremented even when the call was refused or denied, on purpose:
            # a model that keeps asking for a forbidden tool must still run out
            # of steps, or one refusal becomes an infinite loop.
            "steps": state["steps"] + 1,
            # The observation goes into history, and history goes into the next
            # prompt. That is the whole feedback loop, in one line.
            "history": [f"ACT      {decision.tool} -> {observation}"],
        }

    def record_node(state: AgentState) -> dict:
        incident = state["incident"]
        decision = state["decision"]
        # decision is None when every reply was malformed until the bound hit.
        concluded = decision is not None and decision.action == "conclude"
        # No timestamp here: the graph stays off the clock, and the Stage 4
        # SQLite adapter stamps the time when it stores the row.
        memory.record(
            {
                "kind": incident.kind,
                "container": incident.container,
                # None when the bound stopped the run: an honest "no conclusion"
                # beats recording a diagnosis the model never gave.
                "diagnosis": decision.diagnosis if concluded else None,
                "confidence": decision.confidence if decision is not None else None,
                "action_taken": None,  # nothing gated has run yet; Stage 5 fills this
            }
        )
        if concluded:
            outcome = "concluded"
        elif decision is None:
            outcome = f"STOPPED after {MAX_STEPS} attempts: no reply ever validated"
        else:
            outcome = f"STOPPED by step bound after {MAX_STEPS} steps, no conclusion"
        return {"history": [f"RECORD   {outcome} | saved to episodic memory"]}

    builder = StateGraph(AgentState)
    builder.add_node("detect", detect_node)
    builder.add_node("recall", recall_node)
    builder.add_node("reason", reason_node)
    builder.add_node("act", act_node)
    builder.add_node("record", record_node)

    builder.add_edge(START, "detect")
    builder.add_conditional_edges("detect", route_after_detect, {"recall": "recall", "end": END})
    builder.add_edge("recall", "reason")
    builder.add_conditional_edges(
        # "reason" maps to itself: that is the retry after a malformed reply.
        "reason",
        route,
        {"act": "act", "record": "record", "reason": "reason"},
    )
    builder.add_edge("act", "reason")  # the loop: 01's `continue`, as an edge
    builder.add_edge("record", END)

    return builder.compile(checkpointer=checkpointer)
