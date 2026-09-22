"""
what: one real model call, printed in full.
why:  first contact with a provider should be ONE deliberate request, not a
      whole agent run. If the base URL, the model name or the credentials are
      wrong, you learn that for the price of a single call. And you get to see
      what a model actually sends back before any of our code has tidied it.
how:  builds the same prompt reason_node would build on its first pass, using
      the FAKE metrics -- this step is about the model, so it needs no Docker
      and no Prometheus -- sends it once, prints the raw text, then says whether
      Pydantic accepts it.

      No test covers this file, on purpose: it costs money. Everything it
      exercises is tested elsewhere with stand-ins -- build_prompt in the graph
      tests, AgentDecision in test_seams, the adapter's envelope in
      test_llm_openai with a stubbed transport.
"""

from dotenv import load_dotenv
from pydantic import ValidationError

from agent.adapters.llm_openai import OpenAIAdapter
from agent.adapters.metrics_fake import OOM_VICTIM, FakeMetrics
from agent.detectors import detect
from agent.graph import build_prompt, initial_state
from agent.models import AgentDecision

LINE = "-" * 78


def probe_prompt(container: str = "lab-victim") -> str:
    """Exactly the prompt the graph would send on its first pass."""
    stats = FakeMetrics(OOM_VICTIM).container_stats(container)
    incident = detect(stats)
    state = initial_state(container)
    # The one history line detect_node would have written, so the model sees a
    # realistic investigation rather than an empty one.
    state.update(
        metrics=stats,
        incident=incident,
        history=[f"DETECT   {incident.kind} | {incident.summary}"],
    )
    return build_prompt(state)


def main(adapter=None) -> None:
    load_dotenv()
    # The adapter is a parameter so this can be driven by a stand-in without
    # spending anything; run.py-style construction is the default.
    adapter = adapter if adapter is not None else OpenAIAdapter()
    prompt = probe_prompt()

    print("CONFIG    ", adapter.describe())
    # ~4 characters per token is a rough English average -- close enough to know
    # whether you are about to send one page or ten.
    print("PROMPT    ", f"{len(prompt)} characters, roughly {len(prompt) // 4} tokens")
    print(LINE)

    raw = adapter.decide(prompt)

    print("RAW REPLY (exactly as the model sent it, including any prose):")
    print(raw)
    print(LINE)

    try:
        decision = AgentDecision.model_validate_json(raw)
    except ValidationError as error:
        print("INVALID    Pydantic refused this reply. In a real run the graph would")
        print("           put the error in the history, ask again, and charge a step:")
        print(f"           {error.errors()[0]}")
    else:
        print("VALID     ", decision)

    print(LINE)
    print(f"COST       {adapter.calls} call(s), {adapter.tokens} tokens reported by the provider")


if __name__ == "__main__":
    main()
