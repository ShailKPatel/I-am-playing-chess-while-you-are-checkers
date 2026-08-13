"""Every RuleConfig flag must measurably change behaviour. Spec section 7.1 / acceptance criterion 11.

mandatory_capture, maximal_capture, flying_kings and promotion_ends_jump_chain are
covered in test_checkers_rules.py; fifty_move_plies is covered in test_terminal.py.
This file covers layout and first_mover.
"""
from engine.board import Position, Side
from engine.config import RuleConfig


def _checker_count(pos: Position) -> int:
    return sum(1 for ch in pos.board if ch in ("c", "C"))


def _king_count(pos: Position) -> int:
    return sum(1 for ch in pos.board if ch == "C")


def test_layout_changes_checker_count_and_type():
    cfg = RuleConfig(first_mover="chess")
    standard = Position(cfg, layout="standard_24v16")
    light = Position(cfg, layout="light_16v16")
    kings = Position(cfg, layout="kings_12v16")

    assert _checker_count(standard) == 24
    assert _checker_count(light) == 16
    assert _checker_count(kings) == 16
    assert _king_count(standard) == 0
    assert _king_count(light) == 0
    assert _king_count(kings) == 16


def test_first_mover_is_deterministic_when_fixed():
    cfg_chess = RuleConfig(first_mover="chess")
    cfg_checkers = RuleConfig(first_mover="checkers")
    for _ in range(10):
        assert Position(cfg_chess).side_to_move == Side.CHESS
        assert Position(cfg_checkers).side_to_move == Side.CHECKERS


def test_first_mover_random_hits_both_sides():
    cfg = RuleConfig(first_mover="random")
    seen = {Position(cfg).side_to_move for _ in range(60)}
    assert seen == {Side.CHESS, Side.CHECKERS}
