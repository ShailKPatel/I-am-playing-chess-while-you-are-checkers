"""Pseudo-legal chess move generation. Spec section 3.1.

Generic over colour so it can serve both the real game (white/CHESS side only)
and the chess-only perft/differential test mode (both colours).
"""
from __future__ import annotations

from engine.board import Position, sq_rf, rf_sq, EMPTY, WHITE_PIECES, BLACK_CHESS_PIECES, CHECKER_PIECES
from engine.moves import MoveType, encode

KNIGHT_OFFS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
DIAG4 = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ORTHO4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
KING8 = DIAG4 + ORTHO4


def _own_enemy_sets(color: str, chess_only: bool) -> tuple[set, set]:
    if color == "w":
        own = WHITE_PIECES
        enemy = (BLACK_CHESS_PIECES if chess_only else set()) | CHECKER_PIECES
    else:
        own = BLACK_CHESS_PIECES
        enemy = WHITE_PIECES | (CHECKER_PIECES if not chess_only else set())
    return own, enemy


def pseudo_legal_chess_moves(position: Position, color: str) -> list[int]:
    """All pseudo-legal moves for the chess pieces of `color` ('w' or 'b')."""
    board = position.board
    chess_only = position.mode == "chess_only"
    own, enemy = _own_enemy_sets(color, chess_only)
    moves: list[int] = []
    forward = 1 if color == "w" else -1
    start_rank = 1 if color == "w" else 6
    promo_rank = 7 if color == "w" else 0
    pawn_char = "P" if color == "w" else "p"
    knight_char = "N" if color == "w" else "n"
    bishop_char = "B" if color == "w" else "b"
    rook_char = "R" if color == "w" else "r"
    queen_char = "Q" if color == "w" else "q"
    king_char = "K" if color == "w" else "k"

    for sq in range(64):
        piece = board[sq]
        if piece == EMPTY or piece not in own:
            continue
        r, f = sq_rf(sq)

        if piece == pawn_char:
            one = rf_sq(r + forward, f)
            if one is not None and board[one] == EMPTY:
                if r + forward == promo_rank:
                    for mt in (MoveType.PROMO_N, MoveType.PROMO_B, MoveType.PROMO_R, MoveType.PROMO_Q):
                        moves.append(encode(sq, one, mt))
                else:
                    moves.append(encode(sq, one, MoveType.QUIET))
                    if r == start_rank:
                        two = rf_sq(r + 2 * forward, f)
                        if two is not None and board[two] == EMPTY:
                            moves.append(encode(sq, two, MoveType.DOUBLE_PUSH))
            for df in (-1, 1):
                cap = rf_sq(r + forward, f + df)
                if cap is None:
                    continue
                if board[cap] in enemy:
                    if r + forward == promo_rank:
                        for mt in (MoveType.PROMO_CAP_N, MoveType.PROMO_CAP_B, MoveType.PROMO_CAP_R, MoveType.PROMO_CAP_Q):
                            moves.append(encode(sq, cap, mt))
                    else:
                        moves.append(encode(sq, cap, MoveType.CAPTURE))
                elif position.ep_square is not None and cap == position.ep_square:
                    moves.append(encode(sq, cap, MoveType.EP_CAPTURE))

        elif piece == knight_char:
            for dr, df in KNIGHT_OFFS:
                to = rf_sq(r + dr, f + df)
                if to is None or board[to] in own:
                    continue
                mt = MoveType.CAPTURE if board[to] in enemy else MoveType.QUIET
                moves.append(encode(sq, to, mt))

        elif piece == king_char:
            for dr, df in KING8:
                to = rf_sq(r + dr, f + df)
                if to is None or board[to] in own:
                    continue
                mt = MoveType.CAPTURE if board[to] in enemy else MoveType.QUIET
                moves.append(encode(sq, to, mt))
            moves.extend(_castle_moves(position, color))

        elif piece in (bishop_char, rook_char, queen_char):
            dirs = DIAG4 if piece == bishop_char else ORTHO4 if piece == rook_char else KING8
            for dr, df in dirs:
                rr, ff = r + dr, f + df
                while True:
                    to = rf_sq(rr, ff)
                    if to is None or board[to] in own:
                        break
                    if board[to] in enemy:
                        moves.append(encode(sq, to, MoveType.CAPTURE))
                        break
                    moves.append(encode(sq, to, MoveType.QUIET))
                    rr += dr
                    ff += df

    return moves


def _castle_moves(position: Position, color: str) -> list[int]:
    """Structural pseudo-legality only (rights, empty path, rook present).
    Check-safety conditions (spec 3.3 items 3-5) are applied in legal.py."""
    board = position.board
    moves = []
    if color == "w":
        king_sq, rank = 4, 0
        k_right, q_right = "K", "Q"
        rook_k, rook_q = 7, 0
        king_char, rook_char = "K", "R"
    else:
        king_sq, rank = 60, 7
        k_right, q_right = "k", "q"
        rook_k, rook_q = 63, 56
        king_char, rook_char = "k", "r"

    if board[king_sq] != king_char:
        return moves

    if k_right in position.castling and board[rook_k] == rook_char:
        f1, f2 = rf_sq(rank, 5), rf_sq(rank, 6)
        if board[f1] == EMPTY and board[f2] == EMPTY:
            moves.append(encode(king_sq, rf_sq(rank, 6), MoveType.KING_CASTLE))

    if q_right in position.castling and board[rook_q] == rook_char:
        f1, f2, f3 = rf_sq(rank, 1), rf_sq(rank, 2), rf_sq(rank, 3)
        if board[f1] == EMPTY and board[f2] == EMPTY and board[f3] == EMPTY:
            moves.append(encode(king_sq, rf_sq(rank, 2), MoveType.QUEEN_CASTLE))

    return moves
