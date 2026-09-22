"""
what: the deterministic layer that decides whether something is wrong. Pure
      functions: metrics dict in, Incident or None out.
why:  a detector is not an LLM, and that is the whole argument. It is instant,
      free, identical on every run, and testable without a network. The model's
      job begins AFTER this file has said "yes, RB-002" -- explaining the cause
      and choosing what to do. Letting a model decide whether an incident exists
      at all would make every downstream step rest on a probabilistic answer to
      a question that has a factual one.
how:  detect() walks DETECTORS in order and returns the first hit. Adding a
      second incident class later means adding a function and one tuple entry --
      nothing else in the project changes.

      The purity rule is load-bearing, not stylistic: no I/O, no network, no
      clock. A detector that reads the clock cannot be tested without freezing
      time, and one that mutates its input makes the ORDER of detectors matter,
      which is a bug that stays invisible until there are two of them.

      Note what is deliberately NOT here: "three OOMs in ten minutes is a
      pattern". That needs history, history needs a clock and a database, and
      both are banned from this file. It belongs to MemoryPort, in Stage 4.
"""

from agent.models import Incident

# 128 + signal number, the shell convention Docker follows for killed processes.
OOM_EXIT_CODE = 137  # 128 + 9  (SIGKILL)  -- kernel OOM killer, or `docker kill`
GRACEFUL_EXIT_CODE = 143  # 128 + 15 (SIGTERM) -- a deploy stopping it politely

# One restart is enough to call it a loop: the container came back rather than
# staying down, so the restart policy is now cycling it. RB-002 treats a first
# OOM as an incident in its own right ("a fourth OOM kill in three days is a
# pattern, a first one is an incident") -- counting occurrences over time is
# episodic memory's job, not the detector's.
MIN_RESTARTS_FOR_LOOP = 1


def detect_oom_restart_loop(metrics: dict) -> Incident | None:
    """RB-002: a container being OOM-killed and restarted repeatedly.

    Three conditions must hold together, and the second one is the interesting
    one. RB-002's DO NOT list says: "Never conclude 'OOM' from exit code 137
    alone." 137 is SIGKILL, which the kernel OOM killer sends -- and so does a
    human typing `docker kill`. Without OOMKilled=true this is an unexplained
    SIGKILL, a different finding for a different runbook, and reporting it as an
    OOM would be confidently wrong.
    """
    # Indexed, not .get(). These keys are the MetricsPort contract; an adapter
    # that omits one has a bug, and a KeyError here names it immediately. A
    # .get(key, default) would turn that bug into "no incident found", which is
    # the same output as a healthy container and therefore undebuggable.
    name = metrics["name"]
    exit_code = metrics["last_exit_code"]
    oom_killed = metrics["oom_killed"]
    restarts = metrics["restart_count"]

    if exit_code != OOM_EXIT_CODE:
        return None  # includes 143 (graceful), 1 (app error) and None (never exited)
    if not oom_killed:
        return None  # SIGKILL from somewhere else -- unconfirmed, so not RB-002
    if restarts < MIN_RESTARTS_FOR_LOOP:
        return None  # died and stayed dead: bad, but not a loop

    # A NEW dict every time. Copying rather than aliasing metrics is what keeps
    # the "does not mutate its input" test honest even after the Incident is
    # passed around by the graph.
    evidence = {
        "restart_count": restarts,
        "last_exit_code": exit_code,
        "oom_killed": oom_killed,
        "memory_usage_bytes": metrics["memory_usage_bytes"],
        "memory_limit_bytes": metrics["memory_limit_bytes"],
    }

    return Incident(
        kind="RB-002",
        container=name,
        summary=(
            f"{name} was OOM-killed (exit {exit_code}, OOMKilled=true) and has "
            f"restarted {restarts} time(s); memory {_human_bytes(metrics['memory_usage_bytes'])} "
            f"of {_human_bytes(metrics['memory_limit_bytes'])}"
        ),
        evidence=evidence,
    )


# The extension point. A second incident class is a function above and a name
# here -- graph.py, run.py and the tests do not change.
DETECTORS = (detect_oom_restart_loop,)


def detect(metrics: dict) -> Incident | None:
    """Run every detector, return the first incident, or None if all is well.

    First-match-wins is fine while there is one detector and honest while there
    are few. If this ever needs to return several at once, change the return
    type here -- do not make the individual detectors aware of each other.
    """
    for detector in DETECTORS:
        incident = detector(metrics)
        if incident is not None:
            return incident
    return None


def _human_bytes(value: int) -> str:
    """Bytes as MiB, for the one-line summary a human actually reads.

    Kept private and trivial: the raw numbers stay in evidence, so nothing
    downstream ever has to parse this string back into a quantity.
    """
    if not value:
        return "no limit"  # Docker reports 0 when no mem_limit is set
    return f"{value / (1024 * 1024):.0f} MiB"
