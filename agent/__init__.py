"""
what: marks `agent/` as an importable Python package.
why:  without this file `python -m agent.run` and `from agent.ports import ...`
      both fail with ModuleNotFoundError, which reads like a broken install
      rather than missing plumbing.
how:  the file stays empty of logic on purpose. Import side effects in a package
      __init__ make import order matter, and import order that matters is a bug
      waiting for Stage 3.
"""
