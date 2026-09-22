"""
what: the command-line entrypoint -- `python -m agent.run`.
why:  this is the ONLY file that names concrete adapters. Every other file is
      written against the ports, so something has to pick the real classes, and
      keeping that choice in one place is what makes each later stage a
      one-line change here and nowhere else (rule 6).

      In 01_practice, main() in mock_agent.py both chose the parts AND ran the
      loop -- which is why switching to a real model meant copying the whole file
      into llm_agent.py. Here the two jobs are split: run.py chooses, graph.py runs.

      This file deliberately has NO test. From Stage 1 it talks to real Docker,
      and from Stage 3 to a paid model; a test that imported it would drag both
      into pytest, which must stay offline and free (rule 5). The logic it wires
      together is already tested, with fakes, in tests/test_graph_mock.py.
how:  build four adapters, hand them to build_graph(), invoke once on
      lab-victim, then print the history as a trace and a short verdict.
"""

import os
import sys
import time

from dotenv import load_dotenv
from langgraph.types import Command

from agent.adapters.docs_tfidf import TfidfDocs
from agent.adapters.llm_mock import MockLLM
from agent.adapters.llm_openai import OpenAIAdapter
from agent.adapters.memory_sqlite import SqliteMemory
from agent.adapters.metrics_prometheus import PrometheusMetrics
from agent.checkpoint import sqlite_checkpointer, start_fresh
from agent.graph import MAX_STEPS, build_graph, initial_state

CONTAINER = "lab-victim"

# mock is the DEFAULT, which reverses the laptop's behaviour on purpose. On a
# shared machine nobody is sitting in front of, the first run should prove
# Docker, the exporter, Prometheus, the detector, the allowlist, the audit trail
# and the checkpoint file -- for nothing, offline, and with no chance of a
# surprise invoice. Spending then becomes an explicit act: LLM_PROVIDER=openai.
# Rule 5 makes the mock the default in every test; on a machine none of us can
# see, it is a good default for the first run too.
DEFAULT_PROVIDER = "mock"

# The name of this investigation, as far as the checkpointer is concerned. One
# container, one pending run: `--approve` a day later finds it by this name,
# from a different process, on a different day.
THREAD_ID = CONTAINER


# A scripted request to restart the container, used by --rehearse. Everything
# else in that run is real: real Prometheus numbers, the real allowlist, the
# real tool, the real audit file, the real checkpoint. Only the model is faked.
#
# Why this exists: since Stage 4 gave it the runbook, the model usually DECLINES
# to restart -- correctly, because RB-002 says a restart is not a fix. That is
# the right behaviour and a terrible demo. The gate has to be showable on
# demand; the model's mood is not a test plan.
REHEARSAL_SCRIPT = [
    (
        '{"action": "use_tool", "tool": "restart_container", '
        '"args": {"name": "lab-victim"}, "confidence": 0.9, '
        '"reasoning": "Memory is at the limit; restart to clear the immediate pressure."}'
    ),
    (
        '{"action": "conclude", "diagnosis": "lab-victim is in an OOM restart loop; '
        'a restart clears the symptom only.", "confidence": 0.9, '
        '"reasoning": "Exit 137 with OOMKilled true and a climbing RestartCount."}'
    ),
]


def build_llm(rehearsing: bool, resuming: bool):
    """The model adapter this run should use.

    Three sources, in priority order, and each exists for its own reason:

      --rehearse        a scripted restart REQUEST, so the human gate can be
                        demonstrated on demand and for free. Since Stage 4 gave
                        it the runbook, a real model usually declines to restart
                        -- correct behaviour, and a terrible demo.
      LLM_PROVIDER=mock the default script: look, then conclude. Free and
                        offline, and it never reaches the gate, because
                        get_container_stats is a read.
      LLM_PROVIDER=openai
                        a real /chat/completions endpoint. "openai" names a
                        DIALECT, not a vendor -- a Qwen served by Ollama or vLLM
                        arrives through this same branch, with only LLM_BASE_URL
                        and LLM_MODEL changed. See .env.example.

    Every branch returns something satisfying LLMPort, which is the point:
    graph.py cannot tell them apart, so a free rehearsal and a paid
    investigation exercise exactly the same code, gate included.
    """
    if rehearsing:
        # A RESUMED run reasons once more, after the action. It must be handed
        # the conclusion, not the restart request again -- otherwise the agent
        # asks to restart the container it has just restarted, and the demo
        # loops until the step bound saves it.
        return MockLLM(REHEARSAL_SCRIPT[1:] if resuming else REHEARSAL_SCRIPT)

    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider == "mock":
        return MockLLM()
    if provider == "openai":
        return OpenAIAdapter()
    # Fail loudly rather than falling back. A typo like LLM_PROVIDER=opneai that
    # quietly returned MockLLM would produce a complete, plausible trace that no
    # model ever saw -- the most expensive kind of silent success.
    raise SystemExit(
        f"LLM_PROVIDER={provider!r} is not a provider. Use 'mock' (free, offline) "
        "or 'openai' (any /chat/completions endpoint, including a local Qwen)."
    )


def _answer(argv: list[str]) -> str | None:
    """--approve / --deny resume a paused run; nothing starts a new one."""
    if "--approve" in argv:
        return "approve"
    if "--deny" in argv:
        return "deny"
    return None


def _report_pause(request: dict) -> None:
    """Print what the agent wants to do, and how to answer it."""
    print("-" * 80)
    print("PAUSED - a write is waiting for a human")
    print(f"  tool       : {request['tool']}({request['args']})")
    print(f"  confidence : {request['confidence']}")
    print(f"  because    : {request['reasoning']}")
    print()
    print("  Nothing has been done. The run is on disk and will wait.")
    print("  Answer with:   python -m agent.run --approve")
    print("            or:  python -m agent.run --deny")
    print("-" * 80)


def main() -> None:
    # Load .env before any adapter is built, because an adapter reads its
    # configuration from the environment at construction. The composition root
    # is the right place for this: an adapter that quietly loaded a file would
    # behave differently depending on where it was imported from.
    load_dotenv()

    # Read the arguments before building anything: what the run is (a start, an
    # approval, a denial) changes which model the composition root should hand
    # the graph.
    answer = _answer(sys.argv[1:])
    rehearsing = "--rehearse" in sys.argv[1:]

    # ---- the four adapter choices: the only lines later stages edit ---------
    # THE STAGE 2 SWAP, and it is this one line again. FakeMetrics ->
    # DockerMetrics -> PrometheusMetrics: three sources, three technologies, a
    # scrape loop and a time-series database now in the path -- and graph.py,
    # detectors.py and every test have still never been touched.
    metrics = PrometheusMetrics()  # was DockerMetrics(), before that FakeMetrics()
    # THE STAGE 3 SWAP: the line that can start costing money. MockLLM is not
    # gone -- it is still the default in every test (rule 5), and on a shared
    # machine it is the default here too until someone says otherwise.
    llm = build_llm(rehearsing=rehearsing, resuming=answer is not None)
    # THE STAGE 4 SWAP (second half). The model can now read the runbook that
    # forbids the remediation it kept proposing. Costs ~800 prompt tokens a call.
    docs = TfidfDocs()  # was NullDocs(); reads runbooks/RB-002-...md
    # THE STAGE 4 SWAP (first half). Every earlier trace said "0 prior
    # incidents" because NullMemory died with the process. From here the count
    # is real, and the prompt carries it.
    memory = SqliteMemory()  # was NullMemory(); writes ./lab_agent.db

    # THE STAGE 5 ADDITION. Without a checkpointer, interrupt() would have
    # nowhere to put the paused run, and "wait for a human" would mean "wait as
    # long as this terminal stays open".
    checkpointer = sqlite_checkpointer()
    graph = build_graph(metrics, llm, docs, memory, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}

    pending = graph.get_state(config).next  # ('act',) when a write is waiting

    if answer is not None and not pending:
        print(f"Nothing is waiting for an answer on '{THREAD_ID}'.")
        print("Start a run first: python -m agent.run")
        return
    if answer is None and pending:
        # Refusing to start a second investigation while one is mid-approval:
        # two runs on one thread would resume each other's pause.
        print(f"A run on '{THREAD_ID}' is already paused, waiting for an answer.")
        print("Answer it first: python -m agent.run --approve   (or --deny)")
        return

    # Timed around invoke() only: Python and LangGraph start-up are not the
    # agent, and "under a second" in mvp_plan.md is about the agent.
    started = time.perf_counter()
    if answer is None:
        # Clear the thread before a NEW investigation. Reusing it would merge
        # this run into the last one: see start_fresh() for the measurement.
        start_fresh(checkpointer, THREAD_ID)
        final = graph.invoke(initial_state(CONTAINER), config)
    else:
        # Resuming a run this process never started. Everything it knows comes
        # out of the checkpoint file.
        print(f"Resuming '{THREAD_ID}' with: {answer}")
        final = graph.invoke(Command(resume=answer), config)
    elapsed_ms = (time.perf_counter() - started) * 1000

    if "__interrupt__" in final:
        # The run stopped at the gate. Print the trace so far, then the request.
        for line in final["history"]:
            print(line)
        _report_pause(final["__interrupt__"][0].value)
        return

    adapters = ", ".join(type(a).__name__ for a in (metrics, llm, docs, memory))
    print(f"lab-agent | {CONTAINER} | adapters: {adapters}")
    # Printed only when the adapter can describe itself. On the VM this is the
    # line that answers "which model produced that trace" -- a question that
    # gets much harder once three people can point the same image at three
    # different endpoints. describe() masks the key, so it is safe to paste.
    describe = getattr(llm, "describe", None)
    if describe is not None:
        print(f"         provider: {describe()}")
    print("-" * 80)
    for line in final["history"]:
        print(line)
    print("-" * 80)

    # Counted from the trace, not from MockLLM.calls: that attribute exists only
    # on the mock, and this file must keep working when the adapter is real.
    model_calls = sum(1 for line in final["history"] if line.startswith("REASON"))
    decision = final["decision"]
    if final["incident"] is None:
        print("VERDICT  healthy -- no incident, so the model was never consulted")
    elif decision.action == "conclude":
        print(f"VERDICT  {decision.diagnosis} (confidence {decision.confidence})")
    else:
        print(f"VERDICT  no conclusion -- the step bound ended the run after {MAX_STEPS} tool calls")
    # getattr, because only the real adapter counts tokens; MockLLM has none and
    # this file must keep working with either.
    tokens = getattr(llm, "tokens", 0)
    print(
        f"         {final['steps']} tool call(s) | {model_calls} model call(s) | "
        f"{tokens} tokens | {elapsed_ms:.0f} ms"
    )


if __name__ == "__main__":
    main()
