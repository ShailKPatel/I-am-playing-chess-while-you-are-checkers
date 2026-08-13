"""Position: board state, layouts, make/unmake, Zobrist hashing. Spec section 7.2."""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from engine.config import RuleConfig
from engine.moves import MoveType, decode, PROMO_PIECE, IS_PROMO, IS_CAPTURE

EMPTY = "."

WHITE_PIECES = set("PNBRQK")
BLACK_CHESS_PIECES = set("pnbrqk")
CHECKER_PIECES = set("cC")

ROOK_CASTLE_RIGHT = {0: "Q", 7: "K", 56: "q", 63: "k"}
KING_START = {"w": 4, "b": 60}
ROOK_START = {"w": (0, 7), "b": (56, 63)}


class Side(IntEnum):
    """Which army is to move: the chess army or the checkers army."""
    CHESS = 0
    CHECKERS = 1


class Result(IntEnum):
    """Terminal outcome of a game (spec section 5)."""
    CHESS_WIN = 0
    CHECKERS_WIN = 1
    DRAW = 2


def sq_rf(sq: int) -> tuple[int, int]:
    """0-63 square index -> (rank_idx, file_idx)."""
    return sq // 8, sq % 8


def rf_sq(r: int, f: int) -> Optional[int]:
    """(rank_idx, file_idx) -> 0-63 square index, or None if off the board."""
    if 0 <= r < 8 and 0 <= f < 8:
        return r * 8 + f
    return None


# ---------------------------------------------------------------------------
# Zobrist table (fixed seed => reproducible across processes, spec 7.2)
# ---------------------------------------------------------------------------
_PIECE_CHARS = "PNBRQKpnbrqkcC"
_rng = random.Random(0xC4E55)
ZOBRIST_PIECE = {ch: [_rng.getrandbits(64) for _ in range(64)] for ch in _PIECE_CHARS}
ZOBRIST_SIDE = _rng.getrandbits(64)
ZOBRIST_CASTLING = {ch: _rng.getrandbits(64) for ch in "KQkq"}
ZOBRIST_EP_FILE = [_rng.getrandbits(64) for _ in range(8)]
ZOBRIST_JUMP_FROM = [_rng.getrandbits(64) for _ in range(64)]


def _layout_standard_24v16() -> list[str]:
    board = [EMPTY] * 64
    back = "RNBQKBNR"
    for f in range(8):
        board[rf_sq(0, f)] = back[f]
        board[rf_sq(1, f)] = "P"
    for r in (5, 6, 7):
        for f in range(8):
            board[rf_sq(r, f)] = "c"
    return board


def _layout_light_16v16() -> list[str]:
    board = [EMPTY] * 64
    back = "RNBQKBNR"
    for f in range(8):
        board[rf_sq(0, f)] = back[f]
        board[rf_sq(1, f)] = "P"
    for r in (6, 7):
        for f in range(8):
            board[rf_sq(r, f)] = "c"
    return board


def _layout_kings_12v16() -> list[str]:
    # Spec 15 names this "kings_12v16" but its own prose places checkers on
    # ranks 7-8 both colours (16 squares), all as kings. Prose is authoritative
    # per spec section 15; the numeral in the name is just a label.
    board = [EMPTY] * 64
    back = "RNBQKBNR"
    for f in range(8):
        board[rf_sq(0, f)] = back[f]
        board[rf_sq(1, f)] = "P"
    for r in (6, 7):
        for f in range(8):
            board[rf_sq(r, f)] = "C"
    return board


LAYOUTS = {
    "standard_24v16": _layout_standard_24v16,
    "light_16v16": _layout_light_16v16,
    "kings_12v16": _layout_kings_12v16,
}


def _layout_chess_only() -> list[str]:
    board = [EMPTY] * 64
    back = "RNBQKBNR"
    for f in range(8):
        board[rf_sq(0, f)] = back[f]
        board[rf_sq(1, f)] = "P"
        board[rf_sq(6, f)] = "p"
        board[rf_sq(7, f)] = back[f].lower()
    return board


STANDARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def parse_fen(fen: str) -> tuple[list[str], str, set, Optional[int], int, int]:
    """Parse a standard chess FEN into engine-internal pieces. Used only by chess_only mode."""
    parts = fen.strip().split()
    board_part, side_part, castle_part, ep_part = parts[0], parts[1], parts[2], parts[3]
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    board = [EMPTY] * 64
    ranks = board_part.split("/")
    for i, rank_str in enumerate(ranks):
        r = 7 - i
        f = 0
        for ch in rank_str:
            if ch.isdigit():
                f += int(ch)
            else:
                board[rf_sq(r, f)] = ch
                f += 1
    castling = set(c for c in castle_part if c in "KQkq")
    ep_square = None
    if ep_part != "-":
        file_idx = ord(ep_part[0]) - ord("a")
        rank_idx = int(ep_part[1]) - 1
        ep_square = rf_sq(rank_idx, file_idx)
    side = "w" if side_part == "w" else "b"
    return board, side, castling, ep_square, halfmove, fullmove


@dataclass
class Undo:
    """Everything Position.make() changed, so unmake() can restore it exactly."""
    mv: int
    moved_piece: str
    frm: int
    to: int
    mtype: MoveType
    captured_piece: str
    captured_square: Optional[int]
    prev_castling: frozenset
    prev_ep_square: Optional[int]
    prev_halfmove_clock: int
    prev_fullmove_number: int
    prev_jump_from: Optional[int]
    prev_side_to_move: Side
    prev_key: int
    rook_frm: Optional[int] = None
    rook_to: Optional[int] = None
    promoted: bool = False


class Position:
    """Board state, side to move, and all rule-relevant state (spec 7.2)."""

    def __init__(self, config: RuleConfig, layout: Optional[str] = None):
        self.config = config
        self.mode = "hybrid"
        layout_name = layout or config.layout
        if layout_name == "chess_only_test":
            self.mode = "chess_only"
            self.board = _layout_chess_only()
            self.chess_color = "w"
            self.castling = {"K", "Q", "k", "q"}
        else:
            self.board = LAYOUTS[layout_name]()
            self.chess_color = "w"
            self.castling = {"K", "Q"}
        self.side_to_move = self._initial_side()
        self.ep_square: Optional[int] = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.jump_from: Optional[int] = None
        self.history: list[int] = []
        self._key = self._compute_key()
        self.history.append(self._key)

    def _initial_side(self) -> Side:
        fm = self.config.first_mover
        if fm == "chess":
            return Side.CHESS
        if fm == "checkers":
            return Side.CHECKERS
        return Side.CHESS if random.random() < 0.5 else Side.CHECKERS

    @classmethod
    def from_fen(cls, fen: str, config: RuleConfig) -> "Position":
        """Standard chess FEN -> chess_only-mode Position (perft/differential test harness only, spec 3.4/9.1)."""
        pos = cls.__new__(cls)
        pos.config = config
        pos.mode = "chess_only"
        board, side, castling, ep_square, halfmove, fullmove = parse_fen(fen)
        pos.board = board
        pos.chess_color = side
        pos.castling = castling
        pos.side_to_move = Side.CHESS if side == "w" else Side.CHECKERS
        pos.ep_square = ep_square
        pos.halfmove_clock = halfmove
        pos.fullmove_number = fullmove
        pos.jump_from = None
        pos.history = []
        pos._key = pos._compute_key()
        pos.history.append(pos._key)
        return pos

    # -- Zobrist -----------------------------------------------------------
    def _compute_key(self) -> int:
        k = 0
        for sq, ch in enumerate(self.board):
            if ch != EMPTY:
                k ^= ZOBRIST_PIECE[ch][sq]
        if self.side_to_move == Side.CHECKERS:
            k ^= ZOBRIST_SIDE
        for ch in self.castling:
            k ^= ZOBRIST_CASTLING[ch]
        if self.ep_square is not None:
            k ^= ZOBRIST_EP_FILE[self.ep_square % 8]
        if self.jump_from is not None:
            k ^= ZOBRIST_JUMP_FROM[self.jump_from]
        return k

    def key(self) -> int:
        """Zobrist hash of the full position: board + side + castling + ep + jump_from."""
        return self._key

    # -- Public interface (spec 7.2) ----------------------------------------
    # `side_to_move` is a plain attribute (a Side enum value) rather than a
    # method: it's read on every hot-path call site (movegen, eval, search),
    # and spec section 14 asks for plain attribute access over indirection there.
    def legal_moves(self) -> list[int]:
        """All legal moves for the side to move, as packed ints (spec 7.3)."""
        from engine.legal import legal_moves
        return legal_moves(self)

    def result(self):
        """Terminal result of the game, or None if it continues (spec 5)."""
        from engine.terminal import result
        return result(self)

    def copy(self) -> "Position":
        """Deep-enough copy: board, config reference, and all state fields."""
        pos = Position.__new__(Position)
        pos.config = self.config
        pos.mode = self.mode
        pos.board = list(self.board)
        pos.chess_color = self.chess_color
        pos.castling = set(self.castling)
        pos.side_to_move = self.side_to_move
        pos.ep_square = self.ep_square
        pos.halfmove_clock = self.halfmove_clock
        pos.fullmove_number = self.fullmove_number
        pos.jump_from = self.jump_from
        pos.history = list(self.history)
        pos._key = self._key
        return pos

    def to_planes(self):
        """NN input tensor, shape (19, 8, 8) float32 (spec 7.6). Unused today; built for a future training phase."""
        from engine.planes import to_planes
        return to_planes(self)

    # -- make / unmake -------------------------------------------------
    def make(self, mv: int) -> Undo:
        """Applies a packed move in place and returns an Undo to reverse it exactly (spec 7.2)."""
        frm, to, mtype = decode(mv)
        moved = self.board[frm]
        captured = EMPTY
        captured_square = None
        undo = Undo(
            mv=mv, moved_piece=moved, frm=frm, to=to, mtype=mtype,
            captured_piece=EMPTY, captured_square=None,
            prev_castling=frozenset(self.castling), prev_ep_square=self.ep_square,
            prev_halfmove_clock=self.halfmove_clock, prev_fullmove_number=self.fullmove_number,
            prev_jump_from=self.jump_from, prev_side_to_move=self.side_to_move,
            prev_key=self._key,
        )

        irreversible = False
        new_ep_square = None

        if mtype == MoveType.CHECKER_STEP:
            self.board[to] = moved
            self.board[frm] = EMPTY
            irreversible = True
            self._maybe_promote_checker(to)

        elif mtype == MoveType.CHECKER_JUMP:
            cap_sq = self._find_captured_between(frm, to)
            captured = self.board[cap_sq]
            captured_square = cap_sq
            self.board[cap_sq] = EMPTY
            self.board[to] = moved
            self.board[frm] = EMPTY
            irreversible = True
            promoted = self._maybe_promote_checker(to)
            undo.promoted = promoted
            if promoted and self.config.promotion_ends_jump_chain:
                self.jump_from = None
                self.side_to_move = Side.CHESS
            else:
                further = self._checker_jumps_from(to)
                if further:
                    self.jump_from = to
                else:
                    self.jump_from = None
                    self.side_to_move = Side.CHESS

        elif mtype in (MoveType.KING_CASTLE, MoveType.QUEEN_CASTLE):
            self.board[to] = moved
            self.board[frm] = EMPTY
            r = frm // 8
            if mtype == MoveType.KING_CASTLE:
                rook_frm, rook_to = rf_sq(r, 7), rf_sq(r, 5)
            else:
                rook_frm, rook_to = rf_sq(r, 0), rf_sq(r, 3)
            self.board[rook_to] = self.board[rook_frm]
            self.board[rook_frm] = EMPTY
            undo.rook_frm, undo.rook_to = rook_frm, rook_to
            self._clear_castling_rights(moved)

        elif mtype == MoveType.EP_CAPTURE:
            direction = -8 if moved.isupper() else 8
            cap_sq = to + direction
            captured = self.board[cap_sq]
            captured_square = cap_sq
            self.board[cap_sq] = EMPTY
            self.board[to] = moved
            self.board[frm] = EMPTY
            irreversible = True

        elif mtype == MoveType.DOUBLE_PUSH:
            self.board[to] = moved
            self.board[frm] = EMPTY
            new_ep_square = (frm + to) // 2
            irreversible = True

        elif mtype in IS_PROMO:
            if mtype in IS_CAPTURE:
                captured = self.board[to]
                captured_square = to
            promo_char = PROMO_PIECE[mtype]
            self.board[to] = promo_char if moved.isupper() else promo_char.lower()
            self.board[frm] = EMPTY
            irreversible = True

        elif mtype == MoveType.CAPTURE:
            captured = self.board[to]
            captured_square = to
            self.board[to] = moved
            self.board[frm] = EMPTY
            irreversible = True

        else:  # QUIET
            self.board[to] = moved
            self.board[frm] = EMPTY
            if moved.upper() == "P":
                irreversible = True

        undo.captured_piece = captured
        undo.captured_square = captured_square

        if mtype != MoveType.CHECKER_JUMP:
            if moved.upper() in ("K", "R") or ROOK_CASTLE_RIGHT.get(frm) or ROOK_CASTLE_RIGHT.get(to):
                self._clear_castling_rights(moved, frm=frm, to=to)
            self.ep_square = new_ep_square
            self.side_to_move = Side.CHECKERS if self.side_to_move == Side.CHESS else Side.CHESS

        if captured != EMPTY:
            irreversible = True
        if irreversible:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.side_to_move == Side.CHESS and undo.prev_side_to_move == Side.CHECKERS:
            self.fullmove_number += 1

        self._key = self._compute_key()
        self.history.append(self._key)
        return undo

    def unmake(self, undo: Undo) -> None:
        """Reverses exactly one make() call, restoring the position (spec 7.2)."""
        self.history.pop()
        mtype = undo.mtype
        frm, to = undo.frm, undo.to

        if mtype == MoveType.CHECKER_STEP:
            self.board[frm] = undo.moved_piece
            self.board[to] = EMPTY
        elif mtype == MoveType.CHECKER_JUMP:
            self.board[frm] = undo.moved_piece
            self.board[to] = EMPTY
            self.board[undo.captured_square] = undo.captured_piece
        elif mtype in (MoveType.KING_CASTLE, MoveType.QUEEN_CASTLE):
            self.board[frm] = undo.moved_piece
            self.board[to] = EMPTY
            self.board[undo.rook_frm] = self.board[undo.rook_to]
            self.board[undo.rook_to] = EMPTY
        elif mtype == MoveType.EP_CAPTURE:
            self.board[frm] = undo.moved_piece
            self.board[to] = EMPTY
            self.board[undo.captured_square] = undo.captured_piece
        elif mtype in IS_PROMO:
            self.board[frm] = undo.moved_piece
            self.board[to] = undo.captured_piece if mtype in IS_CAPTURE else EMPTY
        elif mtype == MoveType.CAPTURE:
            self.board[frm] = undo.moved_piece
            self.board[to] = undo.captured_piece
        else:  # QUIET, DOUBLE_PUSH
            self.board[frm] = undo.moved_piece
            self.board[to] = EMPTY

        self.castling = set(undo.prev_castling)
        self.ep_square = undo.prev_ep_square
        self.halfmove_clock = undo.prev_halfmove_clock
        self.fullmove_number = undo.prev_fullmove_number
        self.jump_from = undo.prev_jump_from
        self.side_to_move = undo.prev_side_to_move
        self._key = undo.prev_key

    # -- helpers -------------------------------------------------------
    def _clear_castling_rights(self, moved_piece: str, frm: Optional[int] = None, to: Optional[int] = None) -> None:
        if moved_piece == "K":
            self.castling -= {"K", "Q"}
        elif moved_piece == "k":
            self.castling -= {"k", "q"}
        for sq in (frm, to):
            right = ROOK_CASTLE_RIGHT.get(sq)
            if right:
                self.castling.discard(right)

    def _maybe_promote_checker(self, sq: int) -> bool:
        """Man reaching rank 1 promotes to king. Spec 3.2 promotion rule."""
        if self.board[sq] == "c" and sq // 8 == 0:
            self.board[sq] = "C"
            return True
        return False

    def _find_captured_between(self, frm: int, to: int) -> int:
        r0, f0 = sq_rf(frm)
        r1, f1 = sq_rf(to)
        dr = 1 if r1 > r0 else -1
        df = 1 if f1 > f0 else -1
        r, f = r0 + dr, f0 + df
        found = None
        while (r, f) != (r1, f1):
            sq = rf_sq(r, f)
            if self.board[sq] != EMPTY:
                found = sq
            r += dr
            f += df
        return found

    def _checker_jumps_from(self, sq: int) -> list[tuple[int, int]]:
        """Pseudo-legal jump destinations for the checker piece at sq (any distance if flying)."""
        piece = self.board[sq]
        if piece not in CHECKER_PIECES:
            return []
        is_king = piece == "C"
        r0, f0 = sq_rf(sq)
        results = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        if not is_king:
            directions = [(-1, 1), (-1, -1)]  # forward = decreasing rank, spec 3.2
        for dr, df in directions:
            if is_king and self.config.flying_kings:
                r, f = r0 + dr, f0 + df
                path_clear = []
                while True:
                    mid = rf_sq(r, f)
                    if mid is None:
                        break
                    if self.board[mid] == EMPTY:
                        path_clear.append(mid)
                        r += dr
                        f += df
                        continue
                    # first occupied square along the diagonal
                    target_piece = self.board[mid]
                    # The king is never a literal capture target (spec 3.6 Trap 4):
                    # an attacked king square is check, resolved on the chess side's
                    # own turn, never removed by a checkers jump — including a hop
                    # that only becomes possible mid-chain, after an earlier hop in
                    # the same turn clears a blocking piece.
                    if target_piece not in CHECKER_PIECES and target_piece != "K":
                        lr, lf = r + dr, f + df
                        while True:
                            land = rf_sq(lr, lf)
                            if land is None or self.board[land] != EMPTY:
                                break
                            results.append((mid, land))
                            lr += dr
                            lf += df
                    break
            else:
                mid = rf_sq(r0 + dr, f0 + df)
                if mid is None or self.board[mid] in (EMPTY, "K") or self.board[mid] in CHECKER_PIECES:
                    continue
                land = rf_sq(r0 + 2 * dr, f0 + 2 * df)
                if land is not None and self.board[land] == EMPTY:
                    results.append((mid, land))
        return results

    def repetition_count(self) -> int:
        """How many times the current position's key has occurred in this game's history."""
        return self.history.count(self._key)

    def to_fen(self) -> str:
        """Standard chess FEN export. Only meaningful in chess_only mode."""
        rows = []
        for r in range(7, -1, -1):
            row = ""
            empties = 0
            for f in range(8):
                ch = self.board[rf_sq(r, f)]
                if ch == EMPTY:
                    empties += 1
                else:
                    if empties:
                        row += str(empties)
                        empties = 0
                    row += ch
            if empties:
                row += str(empties)
            rows.append(row)
        board_part = "/".join(rows)
        side_part = "w" if self.side_to_move == Side.CHESS else "b"
        castle_part = "".join(c for c in "KQkq" if c in self.castling) or "-"
        ep_part = "-"
        if self.ep_square is not None:
            r, f = sq_rf(self.ep_square)
            ep_part = f"{chr(ord('a') + f)}{r + 1}"
        return f"{board_part} {side_part} {castle_part} {ep_part} {self.halfmove_clock} {self.fullmove_number}"
