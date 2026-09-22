"""
what: marks `agent/adapters/` as a package.
why:  every adapter in here implements a Protocol from agent/ports.py. Keeping
      them in one folder makes the swap of Stage 1 -> 2 -> 3 a visible move
      between two files that sit side by side.
how:  empty by design; adapters are imported directly, never re-exported here,
      so nothing hides which concrete class run.py actually chose.
"""
