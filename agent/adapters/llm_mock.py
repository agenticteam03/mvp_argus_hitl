"""
what: an LLMPort adapter that replays a fixed script instead of calling a model.
why:  this file is the project's budget control and its test determinism, in one
      class. The team's entire model budget is USD 40; a test suite that calls an
      API burns it invisibly and makes every test flaky as a bonus. MockLLM is
      the default in every test, permanently -- that is non-negotiable rule 5.
how:  decide() returns RAW JSON TEXT, exactly as a real provider would, because
      LLMPort promises text and not a parsed object. The mock therefore has to
      produce output that survives Pydantic validation just like the real thing,
      and a test asserts that it does. A mock that is easier to satisfy than
      production is a mock that hides the bug you wrote it to catch.

      When the script runs out, the LAST entry repeats forever. That is
      deliberate: a one-entry script that always asks for a tool becomes an
      infinite reasoning loop, which is precisely what the MAX_STEPS test in the
      graph router needs something to stop.
"""

import json

# The Stage 0 script: look first, then conclude. Two calls, in order.
DEFAULT_SCRIPT: tuple[str, ...] = (
    # Call 1 - ask for a tool. A real model may well ask for logs instead; Stage
    # 3 is where you find out what it actually does, and that difference is the
    # whole reason the mock is not a stand-in for testing against reality.
    json.dumps(
        {
            "action": "use_tool",
            "tool": "get_container_stats",
            "args": {"name": "lab-victim"},
            "confidence": 0.6,
            "reasoning": "Restart count is climbing. Confirm current memory usage against the limit before diagnosing.",
        }
    ),
    # Call 2 - conclude, now that an observation is in the history.
    json.dumps(
        {
            "action": "conclude",
            "diagnosis": "lab-victim is in an OOM restart loop: it allocates past its 128 MiB cgroup limit, is killed with exit 137, and the restart policy brings it straight back.",
            "confidence": 0.9,
            "reasoning": "Exit 137 with OOMKilled true and a climbing RestartCount is the RB-002 signature. A restart clears the symptom only.",
        }
    ),
)


class MockLLM:
    """Deterministic scripted responses. Satisfies LLMPort."""

    def __init__(self, script: tuple[str, ...] | list[str] | None = None) -> None:
        self._script = list(script if script is not None else DEFAULT_SCRIPT)
        if not self._script:
            raise ValueError("MockLLM needs at least one scripted response")
        self.calls = 0  # public: tests assert on how many times it was consulted
        self.prompts: list[str] = []  # every prompt received, for inspection

    def decide(self, prompt: str) -> str:
        # Prompts are kept rather than discarded so a test can assert that the
        # incident, the runbook chunks and the prior-incident count actually
        # reached the model. "We put it in the prompt" is a claim worth checking.
        self.prompts.append(prompt)
        index = min(self.calls, len(self._script) - 1)  # past the end: repeat the last
        self.calls += 1
        return self._script[index]
