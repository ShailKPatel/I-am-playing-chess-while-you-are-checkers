"""to_planes(): NN input tensor. Spec section 7.6. Unused by anything today;
built now so the future training phase has a stable interface to target."""
from __future__ import annotations

import numpy as np

from engine.board import Position, Side, sq_rf
from engine.attacks import checkers_attacking
from engine.legal import legal_moves as _legal_moves
from engine.moves import decode

CHESS_PIECE_ORDER = "PNBRQK"
# Own-piece-type legal-destination ("mobility") map, one plane per type: for
# each type, every square at least one piece of that type can legally move
# to THIS ply. Chess' 6 + checkers' 2 — reuses whichever side is actually to
# move's own legal_moves() list, so it only ever describes the side that's
# about to act, never the side sitting idle.
MOBILITY_PIECE_ORDER = "PNBRQKcC"
NUM_PLANES = 21 + len(MOBILITY_PIECE_ORDER)


def to_planes(position: Position, legal=None) -> np.ndarray:
    """Position -> (NUM_PLANES, 8, 8) float32 NN input tensor (spec 7.6).
    `legal` is the side-to-move's legal_moves() list, if the caller already
    has one (e.g. mid self-play) — pass it in to skip recomputing it here
    for the mobility planes. Left None, to_planes computes its own."""
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

    # Explicit king-safety signal (spec addendum, 2026-08-13): since
    # king_capture_immunity=False means a checkers jump can really end the
    # game, the net gets told directly rather than having to infer threat
    # geometry itself. Plane A is a broadcast boolean ("in check" — same
    # pattern as the side-to-move plane above), Plane B pinpoints the exact
    # source square(s) of the threatening checker(s) so the conv tower can
    # attend to it spatially instead of just knowing the fact. Both stay
    # all-zero in chess_only mode / whenever there's no king on the board.
    king_sq = None
    for sq, ch in enumerate(position.board):
        if ch == "K":
            king_sq = sq
            break
    threats = checkers_attacking(position, king_sq) if king_sq is not None else []
    planes[idx, :, :] = 1.0 if threats else 0.0
    idx += 1
    for s in threats:
        r, f = sq_rf(s)
        planes[idx, r, f] = 1.0
    idx += 1

    # Mobility / legal-destination map: NOT "what's on this square" (that's
    # the occupancy planes above) but "which square(s) can a piece of type X
    # legally reach right now" — the reachability table requested 2026-08-13.
    # Two knights able to land on the same square just both light that one
    # cell; this is a per-square yes/no per type, same convention as every
    # other plane here, not a per-piece-instance list. Covers captures (the
    # destination's occupancy plane already shows what's being taken),
    # en passant and castling (both already ordinary legal_moves() entries —
    # nothing extra to special-case here, the engine solved the hard part
    # already) for free, since it's driven straight off legal_moves() output.
    if legal is None:
        legal = _legal_moves(position)
    mobility_idx = {ch: idx + i for i, ch in enumerate(MOBILITY_PIECE_ORDER)}
    for mv in legal:
        frm, to, _mtype = decode(mv)
        plane = mobility_idx.get(position.board[frm])
        if plane is not None:
            r, f = sq_rf(to)
            planes[plane, r, f] = 1.0
    idx += len(MOBILITY_PIECE_ORDER)

    assert idx == NUM_PLANES
    return planes
