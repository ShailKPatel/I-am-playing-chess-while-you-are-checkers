"""to_planes(): NN input tensor. Spec section 7.6. Unused by anything today;
built now so the future training phase has a stable interface to target."""
from __future__ import annotations

import numpy as np

from engine.board import Position, Side, sq_rf

CHESS_PIECE_ORDER = "PNBRQK"
NUM_PLANES = 19


def to_planes(position: Position) -> np.ndarray:
    """Position -> (19, 8, 8) float32 NN input tensor (spec 7.6)."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    idx = 0

    for piece_char in CHESS_PIECE_ORDER:
        for sq, ch in enumerate(position.board):
            if ch == piece_char:
                r, f = sq_rf(sq)
                planes[idx, r, f] = 1.0
        idx += 1

    for piece_char in ("c", "C"):
        for sq, ch in enumerate(position.board):
            if ch == piece_char:
                r, f = sq_rf(sq)
                planes[idx, r, f] = 1.0
        idx += 1

    for r in range(8):
        for f in range(8):
            planes[idx, r, f] = 1.0 if (r + f) % 2 == 1 else 0.0
    idx += 1

    for right in ("K", "Q", "k", "q"):
        planes[idx, :, :] = 1.0 if right in position.castling else 0.0
        idx += 1

    if position.ep_square is not None:
        r, f = sq_rf(position.ep_square)
        planes[idx, r, f] = 1.0
    idx += 1

    planes[idx, :, :] = 1.0 if position.side_to_move == Side.CHECKERS else 0.0
    idx += 1

    if position.jump_from is not None:
        r, f = sq_rf(position.jump_from)
        planes[idx, r, f] = 1.0
    idx += 1

    planes[idx, :, :] = min(position.halfmove_clock / position.config.fifty_move_plies, 1.0)
    idx += 1

    rep = position.repetition_count()
    planes[idx, :, :] = 1.0 if rep >= 2 else 0.0
    idx += 1
    planes[idx, :, :] = 1.0 if rep >= 3 else 0.0
    idx += 1

    assert idx == NUM_PLANES
    return planes
