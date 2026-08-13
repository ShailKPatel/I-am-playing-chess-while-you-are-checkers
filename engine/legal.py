"""Legality filter. Spec section 3.5 and the Correctness Traps of 3.6.

The chess side's legality is determined purely by make-the-move / check-attack /
unmake-the-move. No pins, no discovered-check special casing: they all fall out
of evaluating attacks strictly *after* the candidate move (Trap 1, Trap 2).
"""
from __future__ import annotations

from engine.board import Position, Side, EMPTY, rf_sq
from engine.moves import MoveType, decode
from engine.chess_gen import pseudo_legal_chess_moves
from engine.attacks import is_attacked_by_checkers, is_square_attacked_by_chess


def _find_king(board: list[str], king_char: str):
    for sq, ch in enumerate(board):
        if ch == king_char:
            return sq
    return None


def _square_attacked(position: Position, sq: int, own_color: str) -> bool:
    if position.mode == "chess_only":
        return is_square_attacked_by_chess(position, sq, by_white=(own_color == "b"))
    return is_attacked_by_checkers(position, sq)


def _castle_pre_checks(position: Position, color: str, mtype: MoveType) -> bool:
    """Spec 3.3 conditions 3 & 4: not currently in check, doesn't pass through an
    attacked square. Condition 5 (landing square attacked) falls out of the
    generic post-make filter applied to every pseudo-legal move."""
    king_sq = 4 if color == "w" else 60
    if _square_attacked(position, king_sq, color):
        return False
    rank = king_sq // 8
    pass_file = 5 if mtype == MoveType.KING_CASTLE else 3
    pass_sq = rf_sq(rank, pass_file)
    king_char = position.board[king_sq]
    position.board[king_sq] = EMPTY
    position.board[pass_sq] = king_char
    attacked = _square_attacked(position, pass_sq, color)
    position.board[pass_sq] = EMPTY
    position.board[king_sq] = king_char
    return not attacked


def legal_moves(position: Position) -> list[int]:
    """All legal moves for the side to move (spec 3.5): checkers dispatch to
    checkers_gen; chess moves are pseudo-legal-generated then filtered by
    make/attack/unmake, which handles pins, discovered checks, and Traps 1-4
    (spec 3.6) without any special-casing."""
    if position.side_to_move == Side.CHECKERS and position.mode == "hybrid":
        from engine.checkers_gen import legal_checker_moves
        return legal_checker_moves(position)

    color = "w" if position.side_to_move == Side.CHESS else "b"
    king_char = "K" if color == "w" else "k"
    pseudo = pseudo_legal_chess_moves(position, color)
    legal = []
    for mv in pseudo:
        _frm, _to, mtype = decode(mv)
        if mtype in (MoveType.KING_CASTLE, MoveType.QUEEN_CASTLE):
            if not _castle_pre_checks(position, color, mtype):
                continue
        undo = position.make(mv)
        king_sq = _find_king(position.board, king_char)
        safe = king_sq is None or not _square_attacked(position, king_sq, color)
        position.unmake(undo)
        if safe:
            legal.append(mv)
    return legal


def in_check(position: Position) -> bool:
    """Is the chess king (of the side to move, chess-only mode aware) currently attacked?"""
    color = "w" if position.side_to_move == Side.CHESS else "b"
    if position.mode == "hybrid" and position.side_to_move != Side.CHESS:
        return False
    king_char = "K" if color == "w" else "k"
    king_sq = _find_king(position.board, king_char)
    if king_sq is None:
        return False
    return _square_attacked(position, king_sq, color)
