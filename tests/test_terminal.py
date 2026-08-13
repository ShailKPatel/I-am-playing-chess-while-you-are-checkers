"""Terminal condition tests. Spec section 9.5."""
from engine.board import Position, Side, Result
from engine.config import RuleConfig
from engine.moves import square_from_name as sq, encode, MoveType
from engine.terminal import result


def blank(cfg, side):
    pos = Position(cfg)
    pos.board = ["."] * 64
    pos.castling = set()
    pos.ep_square = None
    pos.jump_from = None
    pos.side_to_move = side
    pos.halfmove_clock = 0
    pos.fullmove_number = 1
    pos._key = pos._compute_key()
    pos.history = [pos._key]
    return pos


def test_checkmate_is_checkers_win():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("d4")] = "K"
    pos.board[sq("e5")] = "c"
    pos.board[sq("b2")] = "C"
    pos.board[sq("f6")] = "c"
    for s in ("d3", "e3", "c4", "e4", "c5", "d5"):
        pos.board[sq(s)] = "P"
    assert result(pos) == Result.CHECKERS_WIN


def test_stalemate_not_in_check_is_checkers_win_not_draw():
    """Spec 5.1: no legal move for chess covers both checkmate and stalemate as a checkers win."""
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("a1")] = "K"
    pos.board[sq("a2")] = "P"
    pos.board[sq("b1")] = "P"
    pos.board[sq("b2")] = "c"
    pos.board[sq("a3")] = "C"
    from engine.legal import in_check, legal_moves
    assert not in_check(pos)
    assert legal_moves(pos) == []
    assert result(pos) == Result.CHECKERS_WIN


def test_last_checker_captured_is_chess_win():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("e1")] = "K"
    pos.board[sq("e8")] = "R"  # arbitrary, no checkers left on board
    assert result(pos) == Result.CHESS_WIN


def test_checkers_side_blocked_is_chess_win():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHECKERS)
    pos.board[sq("a3")] = "c"
    pos.board[sq("b2")] = "K"
    pos.board[sq("c1")] = "P"
    assert result(pos) == Result.CHESS_WIN


def test_lone_chess_king_does_not_end_game():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("e1")] = "K"
    pos.board[sq("e8")] = "c"
    assert result(pos) is None


def test_threefold_repetition_draw():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("a1")] = "K"
    pos.board[sq("h8")] = "C"
    seq = [("a1", "b1"), ("h8", "g7"), ("b1", "a1"), ("g7", "h8")]
    r = None
    for _ in range(3):
        for frm, to in seq:
            mt = MoveType.QUIET if pos.side_to_move == Side.CHESS else MoveType.CHECKER_STEP
            pos.make(encode(sq(frm), sq(to), mt))
            r = result(pos)
            if r is not None:
                break
        if r is not None:
            break
    assert r == Result.DRAW
    assert pos.repetition_count() == 3


def test_fifty_move_rule_fires():
    cfg = RuleConfig(fifty_move_plies=10)
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("a1")] = "K"
    pos.board[sq("h8")] = "c"
    pos.halfmove_clock = 10
    assert result(pos) == Result.DRAW


def test_fifty_move_counter_resets_on_capture():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("d4")] = "K"
    pos.board[sq("a8")] = "R"
    pos.board[sq("h1")] = "c"
    pos.halfmove_clock = 40
    pos.make(encode(sq("a8"), sq("h1"), MoveType.CAPTURE))
    assert pos.halfmove_clock == 0


def test_fifty_move_counter_resets_on_checker_promotion():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHECKERS)
    pos.board[sq("e1")] = "K"
    pos.board[sq("b2")] = "c"
    pos.halfmove_clock = 40
    pos.make(encode(sq("b2"), sq("a1"), MoveType.CHECKER_STEP))
    assert pos.board[sq("a1")] == "C"
    assert pos.halfmove_clock == 0


def test_fifty_move_counter_resets_on_pawn_move():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("e1")] = "K"
    pos.board[sq("d2")] = "P"
    pos.board[sq("h8")] = "c"
    pos.halfmove_clock = 40
    pos.make(encode(sq("d2"), sq("d3"), MoveType.QUIET))
    assert pos.halfmove_clock == 0


def test_fifty_move_counter_increments_on_quiet_non_pawn_move():
    cfg = RuleConfig()
    pos = blank(cfg, Side.CHESS)
    pos.board[sq("e1")] = "K"
    pos.board[sq("b1")] = "N"
    pos.board[sq("h8")] = "c"
    pos.halfmove_clock = 5
    pos.make(encode(sq("b1"), sq("c3"), MoveType.QUIET))
    assert pos.halfmove_clock == 6
