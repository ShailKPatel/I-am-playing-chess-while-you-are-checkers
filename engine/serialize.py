"""State string read/write and human-readable move notation. Spec section 7.5."""
from __future__ import annotations

from engine.board import Position, Side, EMPTY, rf_sq
from engine.config import RuleConfig
from engine.moves import decode, MoveType, PROMO_PIECE, square_name, square_from_name


def serialize(position: Position) -> str:
    """Position -> the FEN-like state string of spec 7.5. Round-trips exactly with deserialize()."""
    rows = []
    for r in range(7, -1, -1):
        row = ""
        empty_count = 0
        for f in range(8):
            ch = position.board[r * 8 + f]
            if ch == EMPTY:
                empty_count += 1
            else:
                if empty_count:
                    row += str(empty_count)
                    empty_count = 0
                row += ch
        if empty_count:
            row += str(empty_count)
        rows.append(row)
    board_str = "|".join(rows)
    side = "w" if position.side_to_move == Side.CHESS else "b"
    castling = "".join(c for c in "KQ" if c in position.castling) or "-"
    ep = square_name(position.ep_square) if position.ep_square is not None else "-"
    jf = square_name(position.jump_from) if position.jump_from is not None else "-"
    return "/".join([
        board_str, side, castling, ep,
        str(position.halfmove_clock), str(position.fullmove_number), jf,
        position.config.config_hash(),
    ])


def deserialize(state: str, config: RuleConfig) -> Position:
    """State string -> Position. Raises ValueError on malformed input or a
    config-hash mismatch (spec 7.5: never silently proceed on a mismatch)."""
    parts = state.split("/")
    if len(parts) != 8:
        raise ValueError(f"malformed state string: expected 8 fields, got {len(parts)}")
    board_str, side, castling, ep, halfmove, fullmove, jf, cfg_hash = parts
    if cfg_hash != config.config_hash():
        raise ValueError("state was serialized under a different RuleConfig")

    board = [EMPTY] * 64
    rows = board_str.split("|")
    if len(rows) != 8:
        raise ValueError("malformed state string: expected 8 ranks")
    for i, row in enumerate(rows):
        r = 7 - i
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                if f > 7:
                    raise ValueError("malformed state string: rank overflow")
                board[rf_sq(r, f)] = ch
                f += 1

    pos = Position.__new__(Position)
    pos.config = config
    pos.mode = "hybrid"
    pos.board = board
    pos.chess_color = "w"
    pos.castling = set(c for c in castling if c in "KQ")
    pos.side_to_move = Side.CHESS if side == "w" else Side.CHECKERS
    pos.ep_square = square_from_name(ep) if ep != "-" else None
    pos.halfmove_clock = int(halfmove)
    pos.fullmove_number = int(fullmove)
    pos.jump_from = square_from_name(jf) if jf != "-" else None
    pos.history = []
    pos._key = pos._compute_key()
    pos.history.append(pos._key)
    return pos


def move_to_str(position: Position, mv: int) -> str:
    """SAN-ish notation: piece letter (chess only) + from + capture marker + to + promotion."""
    frm, to, mtype = decode(mv)
    piece = position.board[frm]
    is_capture = position.board[to] != EMPTY or mtype in (MoveType.EP_CAPTURE, MoveType.CHECKER_JUMP)
    if mtype == MoveType.KING_CASTLE:
        return "O-O"
    if mtype == MoveType.QUEEN_CASTLE:
        return "O-O-O"
    prefix = piece.upper() if piece.upper() not in ("P", "C") else ""
    sep = "x" if is_capture else "-"
    s = f"{prefix}{square_name(frm)}{sep}{square_name(to)}"
    if mtype in PROMO_PIECE:
        s += f"={PROMO_PIECE[mtype]}"
    return s
