"""Pseudo-legal checkers move generation: steps, jumps, chains. Spec section 3.2."""
from __future__ import annotations

from engine.board import Position, sq_rf, rf_sq, EMPTY
from engine.moves import MoveType, encode, decode

DIAG4 = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
MAN_FORWARD = [(-1, 1), (-1, -1)]  # toward rank 1, spec 3.2


def _checker_steps_from(position: Position, sq: int) -> list[int]:
    piece = position.board[sq]
    is_king = piece == "C"
    dirs = DIAG4 if is_king else MAN_FORWARD
    r0, f0 = sq_rf(sq)
    results = []
    if is_king and position.config.flying_kings:
        for dr, df in dirs:
            r, f = r0 + dr, f0 + df
            while True:
                to = rf_sq(r, f)
                if to is None or position.board[to] != EMPTY:
                    break
                results.append(to)
                r += dr
                f += df
    else:
        for dr, df in dirs:
            to = rf_sq(r0 + dr, f0 + df)
            if to is not None and position.board[to] == EMPTY:
                results.append(to)
    return results


def pseudo_legal_checker_moves(position: Position) -> tuple[list[int], list[int]]:
    """Returns (steps, jumps) as packed moves, ignoring mandatory/maximal capture rules."""
    steps: list[int] = []
    jumps: list[int] = []
    board = position.board
    for sq in range(64):
        if board[sq] not in ("c", "C"):
            continue
        for to in _checker_steps_from(position, sq):
            steps.append(encode(sq, to, MoveType.CHECKER_STEP))
        for _mid, land in position._checker_jumps_from(sq):
            jumps.append(encode(sq, land, MoveType.CHECKER_JUMP))
    return steps, jumps


def _chain_length(position: Position, mv: int) -> int:
    """Total captures achievable by playing mv and then continuing optimally (recursive lookahead)."""
    _frm, to, _mt = decode(mv)
    undo = position.make(mv)
    if position.jump_from == to:
        further = [encode(to, land, MoveType.CHECKER_JUMP) for _m, land in position._checker_jumps_from(to)]
        best = max((_chain_length(position, j) for j in further), default=0)
        total = 1 + best
    else:
        total = 1
    position.unmake(undo)
    return total


def legal_checker_moves(position: Position) -> list[int]:
    """Applies jump-continuation lock, mandatory capture, and maximal capture (spec 3.2)."""
    if position.jump_from is not None:
        sq = position.jump_from
        return [encode(sq, land, MoveType.CHECKER_JUMP) for _m, land in position._checker_jumps_from(sq)]

    steps, jumps = pseudo_legal_checker_moves(position)
    if not jumps:
        return steps

    if not position.config.mandatory_capture:
        return steps + jumps

    if position.config.maximal_capture:
        lengths = [_chain_length(position, mv) for mv in jumps]
        best = max(lengths)
        return [mv for mv, length in zip(jumps, lengths) if length == best]

    return jumps
