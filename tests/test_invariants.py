"""Engine-wide invariants. Spec section 9.6."""
import random

import pytest

from engine.board import Position, Side
from engine.config import RuleConfig
from engine.legal import in_check


def _random_walk(pos: Position, rng: random.Random, n_plies: int):
    for _ in range(n_plies):
        if pos.result() is not None:
            break
        moves = pos.legal_moves()
        if not moves:
            break
        mv = rng.choice(moves)
        yield mv


def test_make_unmake_restores_key():
    cfg = RuleConfig(first_mover="chess")
    rng = random.Random(7)
    checked = 0
    for _game in range(200):
        pos = Position(cfg)
        for _ply in range(30):
            if pos.result() is not None:
                break
            moves = pos.legal_moves()
            if not moves:
                break
            mv = rng.choice(moves)
            key_before = pos.key()
            undo = pos.make(mv)
            pos.unmake(undo)
            assert pos.key() == key_before
            pos.make(mv)
            checked += 1
    assert checked > 1000


@pytest.mark.slow
def test_make_unmake_restores_key_100k():
    cfg = RuleConfig(first_mover="chess")
    rng = random.Random(8)
    checked = 0
    while checked < 100_000:
        pos = Position(cfg)
        for _ply in range(40):
            if pos.result() is not None:
                break
            moves = pos.legal_moves()
            if not moves:
                break
            mv = rng.choice(moves)
            key_before = pos.key()
            undo = pos.make(mv)
            pos.unmake(undo)
            assert pos.key() == key_before
            pos.make(mv)
            checked += 1
            if checked >= 100_000:
                break


def test_legal_moves_never_leaves_chess_king_attacked():
    cfg = RuleConfig(first_mover="chess", king_capture_immunity=True)
    rng = random.Random(11)
    checked = 0
    for _game in range(100):
        pos = Position(cfg)
        for _ply in range(40):
            if pos.result() is not None:
                break
            moves = pos.legal_moves()
            if not moves:
                break
            was_chess = pos.side_to_move == Side.CHESS
            mv = rng.choice(moves)
            pos.make(mv)
            if was_chess:
                # The chess side just moved; its own king must not be left attacked.
                saved = pos.side_to_move
                pos.side_to_move = Side.CHESS
                assert not in_check(pos)
                pos.side_to_move = saved
            checked += 1
    assert checked > 500


def test_random_game_terminates():
    cfg = RuleConfig(first_mover="chess")
    rng = random.Random(3)
    for _game in range(20):
        pos = Position(cfg)
        plies = 0
        while pos.result() is None and plies < 3000:
            moves = pos.legal_moves()
            if not moves:
                break
            pos.make(rng.choice(moves))
            plies += 1
        assert plies < 3000, "game failed to terminate within bound"
