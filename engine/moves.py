"""Packed integer move encoding. Spec section 7.3.

bits  0-5   from square (0-63)
bits  6-11  to square (0-63)
bits 12-15  move type
"""
from __future__ import annotations

from enum import IntEnum


class MoveType(IntEnum):
    """The 16 move kinds a packed move can encode (fits the 4-bit type field)."""
    QUIET = 0
    CAPTURE = 1
    DOUBLE_PUSH = 2
    KING_CASTLE = 3
    QUEEN_CASTLE = 4
    EP_CAPTURE = 5
    PROMO_N = 6
    PROMO_B = 7
    PROMO_R = 8
    PROMO_Q = 9
    PROMO_CAP_N = 10
    PROMO_CAP_B = 11
    PROMO_CAP_R = 12
    PROMO_CAP_Q = 13
    CHECKER_STEP = 14
    CHECKER_JUMP = 15


PROMO_PIECE = {
    MoveType.PROMO_N: "N", MoveType.PROMO_B: "B", MoveType.PROMO_R: "R", MoveType.PROMO_Q: "Q",
    MoveType.PROMO_CAP_N: "N", MoveType.PROMO_CAP_B: "B", MoveType.PROMO_CAP_R: "R", MoveType.PROMO_CAP_Q: "Q",
}
PROMO_TYPES = {
    "N": (MoveType.PROMO_N, MoveType.PROMO_CAP_N),
    "B": (MoveType.PROMO_B, MoveType.PROMO_CAP_B),
    "R": (MoveType.PROMO_R, MoveType.PROMO_CAP_R),
    "Q": (MoveType.PROMO_Q, MoveType.PROMO_CAP_Q),
}
IS_PROMO = {
    MoveType.PROMO_N, MoveType.PROMO_B, MoveType.PROMO_R, MoveType.PROMO_Q,
    MoveType.PROMO_CAP_N, MoveType.PROMO_CAP_B, MoveType.PROMO_CAP_R, MoveType.PROMO_CAP_Q,
}
IS_CAPTURE = {
    MoveType.CAPTURE, MoveType.EP_CAPTURE, MoveType.CHECKER_JUMP,
    MoveType.PROMO_CAP_N, MoveType.PROMO_CAP_B, MoveType.PROMO_CAP_R, MoveType.PROMO_CAP_Q,
}


def encode(frm: int, to: int, mtype: MoveType) -> int:
    """Packs a move into a single int (spec 7.3)."""
    return (frm & 0x3F) | ((to & 0x3F) << 6) | ((int(mtype) & 0xF) << 12)


def decode(mv: int) -> tuple[int, int, MoveType]:
    """Unpacks a move int into (from_square, to_square, MoveType)."""
    frm = mv & 0x3F
    to = (mv >> 6) & 0x3F
    mtype = MoveType((mv >> 12) & 0xF)
    return frm, to, mtype


SQUARE_NAMES = [f"{chr(ord('a') + f)}{r + 1}" for r in range(8) for f in range(8)]
NAME_TO_SQUARE = {name: i for i, name in enumerate(SQUARE_NAMES)}


def square_name(sq: int) -> str:
    """0-63 square index -> algebraic name, e.g. 0 -> 'a1'."""
    return SQUARE_NAMES[sq]


def square_from_name(name: str) -> int:
    """Algebraic square name -> 0-63 index, e.g. 'a1' -> 0."""
    return NAME_TO_SQUARE[name]


def move_to_uci(mv: int) -> str:
    """Packed move -> UCI-style string (e.g. 'e2e4', 'e1g1' for castling, 'a7a8q')."""
    frm, to, mtype = decode(mv)
    s = square_name(frm) + square_name(to)
    if mtype in PROMO_PIECE:
        s += PROMO_PIECE[mtype].lower()
    return s
