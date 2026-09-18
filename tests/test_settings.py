"""Verbosity, and the rule that says when a drop is a warning."""

import pytest

import mantispy as mt
from mantispy._core.logging import get_logger, report_drop


@pytest.fixture(autouse=True)
def _restore_verbosity():
    previous = mt.settings.verbosity
    yield
    mt.settings.verbosity = previous


def test_info_is_quiet_by_default_and_audible_at_verbosity_two(capsys):
    mt.settings.verbosity = 1
    get_logger().info("a quiet thing happened")
    assert "a quiet thing" not in capsys.readouterr().err

    mt.settings.verbosity = 2
    get_logger().info("a quiet thing happened")
    assert "a quiet thing" in capsys.readouterr().err


def test_warnings_are_audible_even_at_verbosity_one(capsys):
    mt.settings.verbosity = 1
    get_logger().warning("this one matters")
    assert "this one matters" in capsys.readouterr().err


def test_verbosity_rejects_a_value_it_cannot_map():
    with pytest.raises(ValueError, match="verbosity"):
        mt.settings.verbosity = 7


def test_a_total_drop_warns_while_a_partial_drop_informs(capsys):
    """Losing 2 of 100 features is routine and only informs; losing all 100 warns."""
    mt.settings.verbosity = 1
    report_drop("feature(s)", 2, 100, remedy="pass objects=None to keep every object")
    assert capsys.readouterr().err == ""

    report_drop("feature(s)", 100, 100, remedy="pass objects=None to keep every object")
    captured = capsys.readouterr().err
    assert "100 of 100 feature(s)" in captured
    assert "objects=None" in captured


def test_the_escalation_boundary_is_inclusive(capsys):
    """The rule is `dropped >= max(total // 2, 1)`, so dropping half warns. With `>` this
    case would stay quiet."""
    mt.settings.verbosity = 1
    report_drop("feature(s)", 50, 100)
    assert "50 of 100 feature(s)" in capsys.readouterr().err


def test_the_logger_does_not_double_print_through_the_root(capsys):
    """Without propagate=False, a user who has called logging.basicConfig() (as notebook
    tutorials do) sees every mantispy line twice."""
    mt.settings.verbosity = 1
    assert get_logger().propagate is False
    get_logger().warning("once only")
    assert capsys.readouterr().err.count("once only") == 1
