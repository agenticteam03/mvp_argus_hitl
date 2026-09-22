"""
what: fake versions of the three tools -- two reads, one write.
why:  these were the Stage 0 stand-ins, and from Stage 1 they have a permanent
      job: they are the TEST DOUBLE for the real tools in container.py.
      tests/conftest.py swaps them into REGISTRY for every test, which is what
      keeps `pytest` from reading real containers and restarting them.
how:  plain functions that return an observation STRING. A string, because an
      observation is written into the history and pasted back into the next
      prompt -- the model reads text, so a tool's output is text.

      The numbers are typed in here, NOT imported from metrics_fake.py, on
      purpose: tools must not depend on adapters -- only run.py chooses those
      (rule 6) -- and Stage 1's container.py will not import them either. They
      match metrics_fake.OOM_VICTIM by hand, which is acceptable for a fake that
      is deleted in Stage 1.
"""


def get_container_stats(name: str) -> str:
    """Read tool. Current memory, restart count and last exit state."""
    return (
        f"[FAKE] {name}: running=True restart_count=3 last_exit_code=137 "
        "oom_killed=True memory=127MiB/128MiB"
    )


def get_container_logs(name: str, lines: int = 50) -> str:
    """Read tool. The last `lines` lines of the container's stdout/stderr.

    Empty on purpose, and realistically so. The kernel kills an over-limit
    process with SIGKILL, which cannot be caught or handled -- the process gets
    no chance to write "I am being killed" anywhere. An OOM death leaves the
    container's own logs silent, which is exactly why RB-002 says to check
    `dmesg` rather than `docker logs` before concluding anything.
    """
    return f"[FAKE] {name}: last {lines} log lines: (no output)"


def restart_container(name: str) -> str:
    """WRITE tool. The only tool that changes anything -- and here it changes nothing."""
    return f"[FAKE] restarted {name} (nothing actually happened)"
