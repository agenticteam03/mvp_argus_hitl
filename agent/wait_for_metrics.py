"""
what: block until Prometheus holds samples the agent will actually accept.
why:  "the scrape target is up" and "there is data fresh enough to reason on"
      are different facts. A run started too early dies in detect_node with
      MetricsUnavailable -- which is correct behaviour, and a wasted minute.
      This is the wait that belongs before a demo, not inside the agent.
how:  it asks the SAME adapter the agent uses, so the freshness rule lives in
      one place. If the adapter is happy, the agent will be too.
"""

import sys
import time

from agent.adapters.metrics_prometheus import MetricsUnavailable, PrometheusMetrics

DEFAULT_CONTAINER = "lab-victim"


def wait(container: str = DEFAULT_CONTAINER, attempts: int = 20, delay: float = 2.0) -> bool:
    """True as soon as the adapter can read the container; False if it never can."""
    metrics = PrometheusMetrics()
    for attempt in range(1, attempts + 1):
        try:
            metrics.container_stats(container)
        except MetricsUnavailable as reason:
            # Printed every few attempts rather than every time: the same line
            # twenty times hides the one that matters.
            if attempt == 1 or attempt % 5 == 0:
                print(f"    waiting ({attempt}/{attempts}): {reason}", flush=True)
            time.sleep(delay)
        else:
            print(f"    fresh samples ready after {attempt} check(s)", flush=True)
            return True
    return False


if __name__ == "__main__":
    container = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONTAINER
    # Exit code, not an exception: code.bash checks $LASTEXITCODE.
    raise SystemExit(0 if wait(container) else 1)
