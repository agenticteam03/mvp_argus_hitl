"""
what: a MemoryPort adapter that remembers incidents in a SQLite file.
why:  NullMemory forgets everything the moment the process ends, so every run
      began by believing it had never seen this container before. That is why
      every trace so far says "0 prior RB-002 incident(s)". With this adapter
      the count is real, and the third OOM in ten minutes can be described as a
      pattern rather than as news.
how:  one table, one file, no ORM. SQLite ships with Python, needs no server,
      and the file is gitignored (*.db) because it holds run history, not code.

      The clock lives HERE, not in the graph and never in a detector (rule 7).
      record() stamps the row with the time it was written, which is the only
      reason recent_incidents() can answer "in the last N hours" at all.
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone

# Relative to the working directory, which is the laptop's answer. In a
# container started with `run --rm` the working directory is thrown away when
# the run ends, so this default would silently reset episodic memory to empty
# after every investigation -- NullMemory again, wearing SQLite's clothes.
# LAB_AGENT_DB overrides it; docker-compose.yml points it at a named volume.
DEFAULT_PATH = "lab_agent.db"

# Exactly the shape mvp_plan.md specifies. TEXT for the timestamp because ISO
# 8601 in UTC sorts correctly as a string, which keeps the query boring.
SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  container TEXT NOT NULL,
  diagnosis TEXT,
  confidence REAL,
  action_taken TEXT
)
"""


class SqliteMemory:
    """Episodic memory that survives the process. Satisfies MemoryPort."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get("LAB_AGENT_DB") or DEFAULT_PATH
        # Created on first use rather than in a migration step: one table, and
        # IF NOT EXISTS makes opening an existing database a no-op.
        with self._connect() as connection:
            connection.execute(SCHEMA)

    def recent_incidents(self, kind: str, hours: int) -> list[dict]:
        """Every incident of this kind written in the last `hours` hours."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._connect() as connection:
            # Parameters, never string formatting: a container name arrives from
            # outside this process, and a query built by concatenation is how
            # injection happens.
            rows = connection.execute(
                "SELECT ts, kind, container, diagnosis, confidence, action_taken "
                "FROM incidents WHERE kind = ? AND ts >= ? ORDER BY ts",
                (kind, cutoff),
            ).fetchall()
        return [dict(row) for row in rows]

    def record(self, incident: dict) -> None:
        """Write one incident, stamped with the current time."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO incidents (ts, kind, container, diagnosis, confidence, action_taken) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    incident["kind"],
                    incident["container"],
                    # .get() for the three that are genuinely optional: a run
                    # stopped by the step bound has no diagnosis, and nothing
                    # has been acted on until Stage 5.
                    incident.get("diagnosis"),
                    incident.get("confidence"),
                    incident.get("action_taken"),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        # row_factory makes rows behave like dicts, so the port can return dicts
        # without a hand-written column list that would drift from the SELECT.
        connection.row_factory = sqlite3.Row
        return connection
