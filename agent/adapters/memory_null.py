"""
what: a MemoryPort adapter that remembers only for the lifetime of the process.
why:  it makes the absence of memory VISIBLE. Run the agent three times against
      this adapter and prior_incidents is 0 every single time -- the agent has
      no idea it has seen this container before. That is the deficiency Stage 4
      exists to fix, and watching it is more convincing than being told.
how:  an in-process list. record() appends, recent_incidents() filters by kind.
      The `hours` argument is accepted and IGNORED, because filtering by time
      needs a clock and durable timestamps -- which is exactly the work
      memory_sqlite.py will do in Stage 4.

      Everything here dies with the process. That is not a limitation to be
      apologised for; it is the point of the demo.
"""


class NullMemory:
    """In-process, non-durable episodic memory. Satisfies MemoryPort."""

    def __init__(self) -> None:
        self._incidents: list[dict] = []

    def recent_incidents(self, kind: str, hours: int) -> list[dict]:
        # `hours` is deliberately unused: with no persisted timestamps there is
        # nothing to compare against. Honouring the signature while ignoring the
        # argument keeps the port stable so Stage 4 can implement it properly.
        return [i for i in self._incidents if i.get("kind") == kind]

    def record(self, incident: dict) -> None:
        self._incidents.append(dict(incident))  # copied: the caller may reuse its dict
