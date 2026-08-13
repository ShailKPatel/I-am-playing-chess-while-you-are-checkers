"""Serialization round-trip tests. Spec section 7.5 / 9.6."""
import random

from engine.board import Position
from engine.config import RuleConfig
from engine.serialize import serialize, deserialize

N_GAMES = 200
PLIES_PER_GAME = 25


def _fields(pos: Position):
    return (
        tuple(pos.board), pos.side_to_move, frozenset(pos.castling),
        pos.ep_square, pos.halfmove_clock, pos.fullmove_number, pos.jump_from,
    )


def test_round_trip_over_random_positions():
    cfg = RuleConfig(first_mover="chess")
    rng = random.Random(99)
    checked = 0
    for _game in range(N_GAMES):
        pos = Position(cfg)
        for _ply in range(PLIES_PER_GAME):
            moves = pos.legal_moves()
            if not moves or pos.result() is not None:
                break
            mv = rng.choice(moves)
            pos.make(mv)

            state = serialize(pos)
            restored = deserialize(state, cfg)
            assert _fields(pos) == _fields(restored)
            assert serialize(restored) == state
            checked += 1
    assert checked > 1000


def test_deserialize_rejects_wrong_config_hash():
    cfg1 = RuleConfig(mandatory_capture=True)
    cfg2 = RuleConfig(mandatory_capture=False)
    pos = Position(cfg1)
    state = serialize(pos)
    try:
        deserialize(state, cfg2)
        assert False, "expected ValueError on config hash mismatch"
    except ValueError:
        pass
