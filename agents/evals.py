"""Evaluation functions. Spec section 8.3. Starting-point weights, not tuned values."""
from __future__ import annotations

from engine.board import Position, Side, sq_rf, rf_sq, LAYOUTS
from engine import chess_gen
from engine.attacks import is_attacked_by_checkers

_INITIAL_CHECKER_COUNT_CACHE: dict[str, int] = {}

KING8 = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]

# 10% of ordinary chess values (spec 8.3: keep search from throwing pieces away for nothing)
_PIECE_VALUE = {"P": 0.1, "N": 0.3, "B": 0.3, "R": 0.5, "Q": 0.9}


def _initial_checker_count(layout: str) -> int:
    if layout not in _INITIAL_CHECKER_COUNT_CACHE:
        board = LAYOUTS[layout]()
        _INITIAL_CHECKER_COUNT_CACHE[layout] = sum(1 for ch in board if ch in ("c", "C"))
    return _INITIAL_CHECKER_COUNT_CACHE[layout]


def _find(board: list[str], char: str) -> int | None:
    for sq, ch in enumerate(board):
        if ch == char:
            return sq
    return None


def _advancement(sq: int) -> int:
    """Ranks travelled toward rank 1 (index 0) from the checkers' home area."""
    r, _f = sq_rf(sq)
    return 7 - r


def chess_side_eval(position: Position) -> float:
    """Chess wants the board empty of checkers (spec 8.3)."""
    board = position.board
    score = 0.0

    surviving_men = surviving_kings = 0
    advancement_penalty = 0
    for sq, ch in enumerate(board):
        if ch == "c":
            surviving_men += 1
            advancement_penalty += _advancement(sq)
        elif ch == "C":
            surviving_kings += 1
        elif ch in _PIECE_VALUE:
            score += _PIECE_VALUE[ch]

    captured = _initial_checker_count(position.config.layout) - surviving_men - surviving_kings
    score += 120 * captured
    score -= 40 * surviving_men
    score -= 90 * surviving_kings
    score -= 15 * advancement_penalty

    if position.side_to_move == Side.CHESS:
        mobility = len(position.legal_moves())
    else:
        mobility = len(chess_gen.pseudo_legal_chess_moves(position, "w"))
    score += mobility

    king_sq = _find(board, "K")
    if king_sq is not None:
        r, f = sq_rf(king_sq)
        adjacent_attacked = 0
        for dr, df in KING8:
            n = rf_sq(r + dr, f + df)
            if n is not None and is_attacked_by_checkers(position, n):
                adjacent_attacked += 1
        score -= 35 * adjacent_attacked
        if is_attacked_by_checkers(position, king_sq):
            score -= 25

    return score


def checkers_side_eval(position: Position) -> float:
    """Checkers want a mating net around the enemy king (spec 8.3)."""
    board = position.board
    score = 0.0

    king_sq = _find(board, "K")
    if king_sq is not None:
        if is_attacked_by_checkers(position, king_sq):
            score += 50
        r, f = sq_rf(king_sq)
        king_mobility = 0
        adjacent_attacked = 0
        for dr, df in KING8:
            n = rf_sq(r + dr, f + df)
            if n is None:
                continue
            if is_attacked_by_checkers(position, n):
                adjacent_attacked += 1
            if board[n] not in ("P", "N", "B", "R", "Q", "K"):
                king_mobility += 1
        score += 20 * adjacent_attacked
        score -= 15 * king_mobility

    for sq, ch in enumerate(board):
        if ch == "C":
            score += 60
        elif ch == "c":
            score += 20
            score += 3 * _advancement(sq)

    for sq, ch in enumerate(board):
        if ch in ("c", "C") and position._checker_jumps_from(sq):
            score += 10

    return score
