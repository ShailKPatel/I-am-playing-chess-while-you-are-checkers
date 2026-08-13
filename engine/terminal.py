"""Win/draw detection. Spec section 5. Order of evaluation matters (5, table)."""
from __future__ import annotations

from engine.board import Position, Side, Result
from engine.legal import legal_moves, in_check


def result(position: Position) -> Result | None:
    """None if the game continues. Evaluated in the order given in spec section 5."""
    if position.mode != "hybrid":
        return _chess_only_result(position)

    if not position.config.king_capture_immunity and "K" not in position.board:
        # Only reachable with the king_capture_immunity experiment flag off —
        # with it on (default), checkers_gen never generates a move that
        # removes K, so this branch never fires.
        return Result.CHECKERS_WIN

    checkers_left = sum(1 for ch in position.board if ch in ("c", "C"))
    if checkers_left == 0:
        return Result.CHESS_WIN

    if position.side_to_move == Side.CHECKERS:
        if not legal_moves(position):
            return Result.CHESS_WIN
    else:
        if not legal_moves(position):
            return Result.CHECKERS_WIN

    if position.repetition_count() >= 3:
        return Result.DRAW

    if position.halfmove_clock >= position.config.fifty_move_plies:
        return Result.DRAW

    return None


def _chess_only_result(position: Position) -> Result | None:
    """Plain chess terminal detection, used only by the perft/differential test mode."""
    if not legal_moves(position):
        if in_check(position):
            return Result.CHECKERS_WIN if position.side_to_move == Side.CHESS else Result.CHESS_WIN
        return Result.DRAW
    if position.repetition_count() >= 3:
        return Result.DRAW
    if position.halfmove_clock >= position.config.fifty_move_plies:
        return Result.DRAW
    return None
