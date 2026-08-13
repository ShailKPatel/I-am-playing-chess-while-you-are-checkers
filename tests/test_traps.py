"""One test per Correctness Trap, spec sections 3.6 and 9.4."""
from engine.board import Position, Side, Result
from engine.config import RuleConfig
from engine.moves import square_from_name as sq, decode, move_to_uci
from engine.legal import legal_moves as lm, in_check
from engine.terminal import result


def blank(cfg, side=Side.CHESS):
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


def test_trap1_king_walks_into_jump_by_vacating_landing_square():
    """Checker on e6, empty intermediate square d5, king on landing square c4.
    King's move onto d5 must be illegal (it would empty c4, enabling e6's jump)."""
    cfg = RuleConfig(king_capture_immunity=True)
    pos = blank(cfg)
    pos.board[sq("e6")] = "c"
    pos.board[sq("c4")] = "K"
    dests = {decode(m)[1] for m in lm(pos)}
    assert sq("d5") not in dests
    assert sq("b3") in dests  # an unrelated escape stays legal


def test_trap2_discovered_check_by_vacating_landing_square():
    """King at d5 sits where a checker jump would land it if c4 (landing) empties.
    Any move of the blocking bishop off c4 is illegal, in every direction."""
    cfg = RuleConfig(king_capture_immunity=True)
    pos = blank(cfg)
    pos.board[sq("d5")] = "K"
    pos.board[sq("e6")] = "c"
    pos.board[sq("c4")] = "B"
    ucis = {move_to_uci(m) for m in lm(pos)}
    assert not any(u.startswith("c4") for u in ucis)


def test_trap3_resolving_check_by_capture_or_block():
    """Capturing the threatening checker, or occupying its landing square, resolves check."""
    cfg = RuleConfig(king_capture_immunity=True)
    pos = blank(cfg)
    pos.board[sq("d4")] = "K"
    pos.board[sq("e5")] = "c"  # threatens d4, landing c3 empty
    pos.board[sq("g6")] = "N"  # can capture e5
    pos.board[sq("a1")] = "B"  # can block landing c3
    pos.board[sq("h1")] = "R"  # unrelated piece: none of its moves resolve check
    assert in_check(pos)
    ucis = {move_to_uci(m) for m in lm(pos)}
    assert "g6e5" in ucis
    assert "a1c3" in ucis
    assert not any(u.startswith("h1") for u in ucis)


def test_trap4_no_king_capture_checkmate_detected_instead():
    """Fully boxed king in check: legal_moves() is empty and result() is CHECKERS_WIN.
    The king piece itself is never removed from the board — no CHECKER_JUMP ever
    targets it (spec 3.6 Trap 4: check is evaded or the game ends, the king is
    never literally captured)."""
    cfg = RuleConfig(king_capture_immunity=True)
    pos = blank(cfg)
    pos.board[sq("d4")] = "K"
    pos.board[sq("e5")] = "c"  # checks d4 (landing c3 empty)
    pos.board[sq("b2")] = "C"  # covers escape to c3 once d4 empties
    pos.board[sq("f6")] = "c"  # covers escape/capture to e5 once d4 empties
    for s in ("d3", "e3", "c4", "e4", "c5", "d5"):
        pos.board[sq(s)] = "P"

    assert in_check(pos)
    assert lm(pos) == []
    assert result(pos) == Result.CHECKERS_WIN


def test_king_never_captured_mid_jump_chain():
    """Regression: a checkers multi-jump chain must never remove the king, even
    when an earlier hop in the *same* chain relocates the jumping piece to a
    square newly adjacent to the king with a free landing square beyond it.
    Chess gets no turn in between hops of one chain (spec 7.4), so nothing but
    the move generator itself can stop this — the chess-side legality filter
    (which only runs on chess's own moves) can't see it coming."""
    cfg = RuleConfig(king_capture_immunity=True)
    pos = blank(cfg, Side.CHECKERS)
    pos.board[sq("d4")] = "K"
    pos.board[sq("f6")] = "P"
    pos.board[sq("g7")] = "c"

    hop1 = next(m for m in lm(pos) if move_to_uci(m) == "g7e5")
    pos.make(hop1)
    assert pos.jump_from is None, "chain must not continue into a king capture"
    assert pos.side_to_move == Side.CHESS
    assert pos.board[sq("d4")] == "K"
    assert in_check(pos), "the halted chain should deliver check, not a capture"
    assert pos.board[sq("d4")] == "K"  # king was never captured
