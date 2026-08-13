"""Attack detection. Spec section 3.5 (is_attacked_by_checkers) plus a generic
standard-chess attack function used only by the chess-only perft/differential mode.
"""
from __future__ import annotations

from engine.board import Position, sq_rf, rf_sq, EMPTY

DIAG4 = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT_OFFS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
ORTHO4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def checkers_attacking(position: Position, square: int) -> list[int]:
    """Spec 3.5: source square(s) of every checker that can jump X (`square`)
    and land empty. Direction d = (dr, df) is the direction of travel
    S -> X -> L. Men only travel with dr == -1 (toward rank 1, spec 3.2).
    Kings travel any of the four diagonals; flying kings (config) may source
    the jump from any distance along the diagonal provided the path is clear
    (spec 3.2). Empty list == not attacked; is_attacked_by_checkers is just
    bool(this)."""
    board = position.board
    flying = position.config.flying_kings
    r1, f1 = sq_rf(square)
    sources = []
    for dr, df in DIAG4:
        land = rf_sq(r1 + dr, f1 + df)
        if land is None or board[land] != EMPTY:
            continue
        r, f = r1 - dr, f1 - df
        first = True
        while True:
            s = rf_sq(r, f)
            if s is None:
                break
            piece = board[s]
            if piece == EMPTY:
                if not flying:
                    break
                r -= dr
                f -= df
                first = False
                continue
            if piece == "c" and first and dr == -1:
                sources.append(s)
            elif piece == "C" and (first or flying):
                sources.append(s)
            break
    return sources


def is_attacked_by_checkers(position: Position, square: int) -> bool:
    """Spec 3.5: X is attacked if a checker on S can jump it and land empty."""
    return bool(checkers_attacking(position, square))


def _find_king(board: list[str], king_char: str) -> int | None:
    for sq, ch in enumerate(board):
        if ch == king_char:
            return sq
    return None


def is_square_attacked_by_chess(position: Position, square: int, by_white: bool) -> bool:
    """Standard FIDE attack detection, used only in chess_only mode (perft/differential)."""
    board = position.board
    r1, f1 = sq_rf(square)
    pawn, knight, bishop, rook, queen, king = ("PNBRQK" if by_white else "pnbrqk")

    pawn_dr = -1 if by_white else 1  # attacker's pawn sits *behind* the target relative to its push dir
    for df in (1, -1):
        s = rf_sq(r1 + pawn_dr, f1 + df)
        if s is not None and board[s] == pawn:
            return True

    for dr, df in KNIGHT_OFFS:
        s = rf_sq(r1 + dr, f1 + df)
        if s is not None and board[s] == knight:
            return True

    for dr, df in DIAG4:
        s = rf_sq(r1 + dr, f1 + df)
        if s is not None and board[s] == king:
            return True
    for dr, df in ORTHO4:
        s = rf_sq(r1 + dr, f1 + df)
        if s is not None and board[s] == king:
            return True

    for dr, df in DIAG4:
        r, f = r1 + dr, f1 + df
        while True:
            s = rf_sq(r, f)
            if s is None:
                break
            piece = board[s]
            if piece == EMPTY:
                r += dr
                f += df
                continue
            if piece == bishop or piece == queen:
                return True
            break

    for dr, df in ORTHO4:
        r, f = r1 + dr, f1 + df
        while True:
            s = rf_sq(r, f)
            if s is None:
                break
            piece = board[s]
            if piece == EMPTY:
                r += dr
                f += df
                continue
            if piece == rook or piece == queen:
                return True
            break

    return False
