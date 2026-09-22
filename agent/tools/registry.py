"""
what: the single function through which every tool call in the agent passes.
why:  one choke point means one place to enforce policy. The graph never calls a
      tool directly -- it calls dispatch(), and dispatch() decides what runs.

      This is the safety boundary of the whole project. The model may ask for
      anything; ALLOWED decides what is permitted -- in Python, not in prose. A
      tool named in a prompt is granted nothing; only this table grants.

      Rule 2 lives here: writes may only target lab-* containers. The tests for
      it -- restart_container(name="postgres"), names that merely look like
      lab-*, and a name that is not even a string -- were written and watched
      FAIL against the unguarded version of this file before these checks
      existed. A safety test that has never failed may be asserting nothing.

how:  REGISTRY maps a tool name to a function, and dispatch() calls it with the
      model's args. The gate is deliberately NOT injected into the graph as a
      swappable dependency: metrics sources are adapters and should be
      swappable, but a safety boundary that run.py could replace with a more
      permissive one is not a safety boundary.
"""

import inspect
from collections.abc import Callable

# The one line Stage 1 changed. The fakes are not deleted: tests/conftest.py
# swaps them back in for every test, so pytest never touches Docker (rule 5).
from agent.tools import container

# The POLICY table: which tools may run, and under what conditions. Separate
# from REGISTRY, which only says HOW to run them. Two tables, two jobs -- and
# adding a function to REGISTRY grants nothing until a policy appears here, so a
# tool someone forgets to list is refused rather than quietly permitted.
ALLOWED: dict[str, dict] = {
    "get_container_stats": {"write": False},
    "get_container_logs": {"write": False},
    # The only write in the project, and the only entry with a target rule.
    "restart_container": {"write": True, "name_prefix": "lab-"},
}

REGISTRY: dict[str, Callable[..., str]] = {
    "get_container_stats": container.get_container_stats,
    "get_container_logs": container.get_container_logs,
    "restart_container": container.restart_container,
}


class ToolNotAllowed(Exception):
    """A tool call the dispatcher refuses: a tool that is not on the allowlist,
    or a write aimed at a container outside lab-*.

    A refusal, not a crash. The graph catches it and hands the message to the
    model as an observation -- the same idea as `{"ok": False, "error": ...}`
    in 01_practice's run_tool.
    """


class ToolCallInvalid(ToolNotAllowed):
    """The tool is permitted, but the arguments do not fit its signature.

    A subclass, so everything that catches ToolNotAllowed -- act_node, the
    tests -- keeps working, while the name still tells the truth: this is a
    malformed call, not a forbidden one.

    It exists because a real model produced, on its first call ever:
        get_container_logs(container="lab-victim", tail=200)
    for a tool whose signature is (name, lines=50). Pydantic accepted that --
    `args` is a free dict -- and the allowlist accepted it too, because the tool
    IS permitted. Without this check the call reached Python and died with a
    TypeError, which kills the run instead of teaching the model anything.
    """


def check(tool: str, args: dict) -> dict:
    """Decide whether this call is permitted, and return its policy entry.

    PURE: it answers the question without doing anything. That is what lets the
    graph ask "would this be allowed?" before waking a human to approve it --
    nobody should be asked to authorise a call the code will reject anyway --
    and it is why re-running this after a resumed pause is harmless.

    Raises ToolNotAllowed for a tool that is not listed or a write aimed outside
    lab-*, and ToolCallInvalid for arguments that do not fit the signature.
    """
    spec = ALLOWED.get(tool)
    if spec is None:
        # Fail closed. Anything not named in ALLOWED is refused, including a
        # function that exists in REGISTRY. A model asking for a tool that does
        # not exist is not a bug in the model; it is the reason for this line.
        raise ToolNotAllowed(f"{tool} is not in the allowlist")

    prefix = spec.get("name_prefix")
    if prefix is not None:
        name = args.get("name")
        # isinstance FIRST: {"name": 123} must be refused, not crash the run on
        # AttributeError. A gate that breaks on odd input is not a gate, and odd
        # input is exactly what a language model eventually produces.
        if not isinstance(name, str) or not name.startswith(prefix):
            raise ToolNotAllowed(f"{tool} may only target {prefix}* containers, not {name!r}")

    function = REGISTRY[tool]
    # Per-ARGUMENT validation, the job 01_practice gave validate_arguments().
    # signature().bind() answers "could this function be called with exactly
    # these arguments?" without calling it -- so a wrong name or a missing
    # required argument becomes a refusal the model can read and correct,
    # instead of a TypeError that ends the investigation.
    try:
        inspect.signature(function).bind(**args)
    except TypeError as mismatch:
        accepted = ", ".join(inspect.signature(function).parameters)
        raise ToolCallInvalid(f"{tool} takes ({accepted}) -- {mismatch}") from mismatch

    return spec


def dispatch(tool: str, args: dict) -> str:
    """Run the named tool, if policy allows it, and return its observation.

    Two checks run before any tool code does:
      1. is this tool on the allowlist at all?
      2. if it writes, does its target start with the required prefix?

    That order matters for more than tidiness. Because both checks come before
    REGISTRY[tool](**args), a refused call never executes tool code -- so these
    refusals stay free and offline even in Stage 5, when restart_container
    really restarts a container.
    """
    # The same check() the graph already ran before pausing for approval. Run
    # again here on purpose: dispatch must be safe to call on its own, and a
    # policy that is only enforced by the caller is not enforced at all.
    check(tool, args)
    return REGISTRY[tool](**args)
