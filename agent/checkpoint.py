"""
what: the durable checkpointer -- where a paused run is kept.
why:  interrupt() stops the graph and saves its state. Until now that state went
      into memory, so "paused" really meant "paused as long as this process
      lives". A human gate that cannot outlive the terminal it ran in is not a
      gate: nobody approves a production restart in the four seconds before the
      script gives up.

      With this, the pause is a row in a file. The process can exit, be killed,
      be deployed over, or move to another machine, and the answer can still
      arrive tomorrow.
how:  SqliteSaver over one connection, plus an explicit list of the types our
      state carries.

      That list is not decoration. LangGraph warns:

          Deserializing unregistered type agent.models.Incident from
          checkpoint. This will be blocked in a future version.

      and it is right to: loading arbitrary types out of a file is how a
      deserialisation vulnerability starts. Naming the two we actually use keeps
      the warning quiet AND keeps the door shut for everything else.
"""

import os
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

# The laptop's answer. On a container this MUST move to a volume, and the
# reason is the whole of Stage 5: the gate works by exiting the process and
# resuming it later from this file. `docker compose run --rm` destroys the
# container's filesystem on exit, so a relative path here means the pause is
# gone before anyone can answer it -- and the failure looks like "nothing is
# waiting for an answer", not like a missing file.
#
# Read at call time, not at import: run.py calls load_dotenv() inside main(),
# after this module has been imported, so a module-level os.environ.get would
# miss anything set in .env.
DEFAULT_PATH = "lab_checkpoints.db"

# Exactly the two types AgentState holds beyond plain JSON: the detector's
# frozen dataclass and the model's validated decision. Anything else in a
# checkpoint file is refused rather than quietly reconstructed.
ALLOWED_TYPES = [
    ("agent.models", "Incident"),
    ("agent.models", "AgentDecision"),
]


def start_fresh(checkpointer: SqliteSaver, thread_id: str) -> None:
    """Forget any earlier run on this thread before starting a new one.

    Without this, a second investigation on the same thread_id MERGES into the
    first. `history` carries an operator.add reducer (state.py), so the previous
    run's lines survive and are sent to the model again in every prompt.

    Measured on three runs of one thread: 6 history lines, then 12, then 18 --
    a prompt, and therefore a bill, that grows linearly with how often the agent
    is run, plus a trace that shows two investigations as if they were one.

    A run waiting for approval must NOT be cleared this way: that is why run.py
    checks for a pending answer first.
    """
    checkpointer.delete_thread(thread_id)


def sqlite_checkpointer(path: str | None = None) -> SqliteSaver:
    """A checkpointer that keeps paused runs in a SQLite file."""
    connection = sqlite3.connect(
        path or os.environ.get("LAB_AGENT_CHECKPOINTS") or DEFAULT_PATH,
        # The graph may touch the connection from a worker thread. Without this
        # sqlite3 refuses with "created in a thread other than the current one",
        # which surfaces as a confusing failure in the middle of a run.
        check_same_thread=False,
    )
    saver = SqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES))
    # Creates the tables on first use. Safe to call again: it is idempotent.
    saver.setup()
    return saver
