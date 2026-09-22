"""
what: the human gate, both answers, in one process.
why:  a list of passing tests does not show what a pause IS. This runs the same
      incident twice -- approved once, denied once -- and prints what the human
      was shown, what the agent then did, and what each answer left in the audit
      trail.
how:  fake tools and MockLLM, so it costs nothing and touches no container. An
      in-memory checkpointer is enough here because both halves happen in one
      process; Step 4 swaps it for SQLite, which is what lets the pause survive
      the process being killed.
"""

import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent import audit
from agent.adapters.docs_null import NullDocs
from agent.adapters.llm_mock import DEFAULT_SCRIPT, MockLLM
from agent.adapters.memory_null import NullMemory
from agent.adapters.metrics_fake import FakeMetrics
from agent.graph import build_graph, initial_state
from agent.tools import container_fake
from agent.tools.registry import REGISTRY

DEMO_TRAIL = "demo_gate.jsonl"

# A write the allowlist PERMITS: lab-victim matches lab-*. Everything the agent
# needs is in place; the only thing left between this and a real restart is a
# person saying yes.
ALLOWED_WRITE = (
    '{"action": "use_tool", "tool": "restart_container", '
    '"args": {"name": "lab-victim"}, "confidence": 0.9, '
    '"reasoning": "Memory is at the limit; restart to clear the immediate pressure."}'
)

LINE = "-" * 78


def _demo_graph():
    """A graph wired to fakes, with a checkpointer so it can pause."""
    # The tools are swapped for fakes here, not in the graph: this demo must not
    # restart anything on your laptop, and the gate is what is being shown.
    REGISTRY.update(
        {
            "get_container_stats": container_fake.get_container_stats,
            "get_container_logs": container_fake.get_container_logs,
            "restart_container": container_fake.restart_container,
        }
    )
    return build_graph(
        FakeMetrics(),
        MockLLM([ALLOWED_WRITE, DEFAULT_SCRIPT[1]]),
        NullDocs(),
        NullMemory(),
        checkpointer=InMemorySaver(),
    )


def _one_run(answer: str, thread: str) -> None:
    graph = _demo_graph()
    # thread_id names the run. Two runs with the same id would resume each
    # other's pause, which is exactly what Step 4 relies on across processes.
    config = {"configurable": {"thread_id": thread}}

    # Where the trail already stood, so the counts below describe THIS run and
    # not the one before it. The first version printed run 1's records under
    # run 2's heading, which made the "nothing yet" claim a lie.
    before = len(audit.read_all(DEMO_TRAIL))

    def written_by_this_run() -> list[str]:
        return [record["event"] for record in audit.read_all(DEMO_TRAIL)[before:]]

    paused = graph.invoke(initial_state("lab-victim"), config)
    request = paused["__interrupt__"][0].value

    print(f"PAUSED. The agent wants to run {request['tool']}({request['args']})")
    print(f"  confidence : {request['confidence']}")
    print(f"  because    : {request['reasoning']}")
    # The strongest claim in the stage: at this moment nothing has been done.
    print(f"  this run has written: {written_by_this_run()}  <- nothing yet")

    print(f"\nA human answers: {answer.upper()}")
    final = graph.invoke(Command(resume=answer), config)

    acted = [line for line in final["history"] if line.startswith("ACT")]
    print(f"  the agent saw : {acted[-1][9:110]}")
    print(f"  this run wrote: {written_by_this_run()}")


def main() -> None:
    # Redirect the trail, the same trick tests/conftest.py uses: this demo must
    # not write into the real audit.jsonl, which is evidence.
    #
    # The ENVIRONMENT VARIABLE, not audit.DEFAULT_PATH. audit.py resolves
    # argument > LAB_AGENT_AUDIT > DEFAULT_PATH, so setting the global alone
    # would be overridden wherever the variable is already set -- which inside
    # the agent container it always is. The demo would then write into the real
    # trail while reading from the demo one, and every run would report that it
    # had written nothing.
    os.environ["LAB_AGENT_AUDIT"] = DEMO_TRAIL
    if os.path.exists(DEMO_TRAIL):
        os.remove(DEMO_TRAIL)

    print(LINE)
    print("RUN 1 - the write is APPROVED")
    print(LINE)
    _one_run("approve", thread="demo-approve")

    print()
    print(LINE)
    print("RUN 2 - the same write is DENIED")
    print(LINE)
    _one_run("deny", thread="demo-deny")

    print()
    print(LINE)
    print("THE TRAIL BOTH RUNS LEFT")
    print(LINE)
    for record in audit.read_all(DEMO_TRAIL):
        detail = record.get("observation") or record.get("answer") or record.get("approval") or ""
        print(f"  {record['ts'][11:19]}  {record['event'].upper():<11} {record['tool']}  {str(detail)[:60]}")


if __name__ == "__main__":
    main()
