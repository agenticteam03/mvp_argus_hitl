"""
what: tests for the tool dispatcher -- first the plumbing, then the safety
      boundary.
why:  the safety tests at the bottom are the safety story of the whole project.
      They were written while the dispatcher was still unguarded and run red
      against it first: a safety test you have never seen fail might be
      asserting nothing at all.
how:  call dispatch() directly with the args shape the model produces, and
      assert either an observation string or a ToolNotAllowed refusal.
"""

import pytest

from agent.tools.registry import REGISTRY, ToolCallInvalid, ToolNotAllowed, dispatch

# ---- plumbing ----------------------------------------------------------------


def test_the_three_tools_are_registered():
    # Given:    the registry
    # Expected: exactly the three tools in scope
    # Why:      a fourth tool is scope creep -- and, from Stage 1, one more thing to gate
    assert set(REGISTRY) == {"get_container_stats", "get_container_logs", "restart_container"}


def test_dispatch_reaches_a_read_tool_and_returns_text():
    # Given:    get_container_stats for lab-victim
    # Expected: a string containing oom_killed=True
    # Why:      observations are text, because they get pasted into the next prompt
    observation = dispatch("get_container_stats", {"name": "lab-victim"})
    assert isinstance(observation, str)
    assert "oom_killed=True" in observation


def test_dispatch_passes_optional_args_through():
    # Given:    get_container_logs with lines=10
    # Expected: the observation mentions 10 lines
    # Why:      the model's args must reach the tool unchanged
    observation = dispatch("get_container_logs", {"name": "lab-victim", "lines": 10})
    assert "last 10 log lines" in observation


def test_the_fake_write_tool_changes_nothing():
    # Given:    restart_container on lab-victim
    # Expected: "nothing actually happened"
    # Why:      Stage 0 writes are fake; this test changes when the real tools arrive
    assert "nothing actually happened" in dispatch("restart_container", {"name": "lab-victim"})


# ---- the safety boundary ---------------------------------------------------------


def test_rejects_unknown_tool():
    # Given:    a tool that is not on the allowlist ("delete_volume")
    # Expected: ToolNotAllowed -- a refusal, not a KeyError crash
    # Why:      a model can ask for anything; only listed tools may ever run
    with pytest.raises(ToolNotAllowed):
        dispatch("delete_volume", {"name": "lab-victim"})


def test_rejects_write_on_non_lab_container():
    # Given:    restart_container aimed at "postgres"
    # Expected: ToolNotAllowed
    # Why:      THE safety boundary (rule 2) -- writes may only touch lab-* containers
    with pytest.raises(ToolNotAllowed):
        dispatch("restart_container", {"name": "postgres"})


def test_rejects_write_on_names_that_only_look_like_lab():
    # Given:    restart_container aimed at "postgres-lab-backup", then "LAB-victim"
    # Expected: ToolNotAllowed for both
    # Why:      the rule is "starts with lab-", exactly -- not "mentions lab" somewhere
    for name in ("postgres-lab-backup", "LAB-victim"):
        with pytest.raises(ToolNotAllowed):
            dispatch("restart_container", {"name": name})


def test_rejects_write_with_a_missing_or_non_text_name():
    # Given:    restart_container with no name at all, then with the number 123
    # Expected: ToolNotAllowed for both
    # Why:      a gate that crashes on odd input is not a gate -- and models send odd input
    for args in ({}, {"name": 123}):
        with pytest.raises(ToolNotAllowed):
            dispatch("restart_container", args)


def test_allows_write_on_a_lab_container():
    # Given:    restart_container aimed at "lab-victim"
    # Expected: an observation, no refusal
    # Why:      a gate that refuses everything is as broken as one that refuses nothing
    assert "restarted lab-victim" in dispatch("restart_container", {"name": "lab-victim"})


def test_rejects_a_call_whose_argument_names_do_not_fit():
    # Given:    get_container_logs(container=..., tail=...) -- what a real model
    #           actually asked for on its first call; the signature is (name, lines)
    # Expected: ToolCallInvalid, before the function is called
    # Why:      otherwise Python raises TypeError and the whole investigation dies
    with pytest.raises(ToolCallInvalid):
        dispatch("get_container_logs", {"container": "lab-victim", "tail": 200})


def test_rejects_a_call_missing_a_required_argument():
    # Given:    get_container_stats with no name at all
    # Expected: ToolCallInvalid
    # Why:      same reason: a refusal the model can read beats a crash it cannot
    with pytest.raises(ToolCallInvalid):
        dispatch("get_container_stats", {})


def test_an_invalid_call_is_still_a_tool_not_allowed():
    # Given:    a call with wrong argument names
    # Expected: it is also caught by `except ToolNotAllowed`
    # Why:      act_node catches the parent class; the subclass only sharpens the message
    with pytest.raises(ToolNotAllowed):
        dispatch("get_container_logs", {"container": "lab-victim"})


def test_read_tools_are_not_limited_to_lab_containers():
    # Given:    get_container_stats aimed at "postgres"
    # Expected: an observation, no refusal
    # Why:      looking is safe; only WRITES carry the lab-* restriction
    assert "postgres" in dispatch("get_container_stats", {"name": "postgres"})
