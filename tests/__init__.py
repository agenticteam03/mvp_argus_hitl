"""
what: marks `tests/` as a package.
why:  this is the load-bearing reason, not tidiness. With this file present,
      pytest walks up from a test module until it finds a directory WITHOUT an
      __init__.py -- the project root -- and puts that on sys.path. That is what
      makes `import agent` work when you run `pytest` from the project root.
      Delete this file and every test fails on import, not on an assertion.
how:  empty by design.
"""
