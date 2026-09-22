"""
what: marks `agent/tools/` as a package.
why:  the allowlist and dispatcher land here in Stage 1. They are the safety
      boundary of the whole project, so they get their own namespace rather
      than being buried among the adapters.
how:  empty by design.
"""
