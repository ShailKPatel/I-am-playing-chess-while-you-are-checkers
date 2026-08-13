"""Differential testing against python-chess. Spec section 9.2 / acceptance criterion 4.

python-chess is a dev-only test oracle; it must never appear in engine/ imports.
"""
import random

import chess as pychess

from engine.board import Position, STANDARD_FEN
from engine.config import RuleConfig
from engine.moves import move_to_uci

TARGET_POSITIONS = 5000
MAX_PLIES = 10


def _legal_uci_set(pos: Position) -> set[str]:
    return {move_to_uci(mv) for mv in pos.legal_moves()}


def test_differential_random_positions():
    cfg = RuleConfig(first_mover="chess")
    rng = random.Random(1234)
    checked = 0
    while checked < TARGET_POSITIONS:
        pos = Position.from_fen(STANDARD_FEN, cfg)
        pyb = pychess.Board()
        for _ply in range(MAX_PLIES):
            ours = _legal_uci_set(pos)
            theirs = {m.uci() for m in pyb.legal_moves}
            assert ours == theirs, (
                f"mismatch at fen={pyb.fen()}\n"
                f"ours-theirs={ours - theirs}\ntheirs-ours={theirs - ours}"
            )
            checked += 1
            if checked >= TARGET_POSITIONS or not ours:
                break
            uci = rng.choice(list(ours))
            mv = next(mv for mv in pos.legal_moves() if move_to_uci(mv) == uci)
            pos.make(mv)
            pyb.push(pychess.Move.from_uci(uci))
