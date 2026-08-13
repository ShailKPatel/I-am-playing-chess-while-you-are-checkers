"""Hand-constructed checkers rule tests. Spec section 9.3."""

from engine.board import Position, Side
from engine.config import RuleConfig
from engine.moves import decode, encode, square_from_name as sq, MoveType


def blank(cfg, side=Side.CHECKERS):
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


def dests_from(pos, frm):
    return {(f, t, mt) for (f, t, mt) in (decode(m) for m in pos.legal_moves()) if f == frm}


def test_man_forward_step_backward_rejected():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("d6")] = "c"
    d = dests_from(pos, sq("d6"))
    tos = {t for (_f, t, _mt) in d}
    assert sq("c5") in tos and sq("e5") in tos
    assert sq("c7") not in tos and sq("e7") not in tos


def test_single_jump_and_blocked_landing():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("d6")] = "c"
    pos.board[sq("c5")] = "P"
    d = dests_from(pos, sq("d6"))
    jumps = {(f, t, mt) for f, t, mt in d if mt == MoveType.CHECKER_JUMP}
    assert (sq("d6"), sq("b4"), MoveType.CHECKER_JUMP) in jumps

    pos2 = blank(cfg)
    pos2.board[sq("d6")] = "c"
    pos2.board[sq("c5")] = "P"
    pos2.board[sq("b4")] = "N"
    d2 = dests_from(pos2, sq("d6"))
    jumps2 = {(f, t, mt) for f, t, mt in d2 if mt == MoveType.CHECKER_JUMP}
    assert jumps2 == set()


def test_jump_offboard_landing_rejected():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("b2")] = "c"
    pos.board[sq("a1")] = "P"
    d = dests_from(pos, sq("b2"))
    jumps = {t for f, t, mt in d if mt == MoveType.CHECKER_JUMP}
    assert jumps == set()


def test_two_hop_chain_keeps_side_to_move():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("d8")] = "c"
    pos.board[sq("c7")] = "P"
    pos.board[sq("c5")] = "N"
    mv1 = encode(sq("d8"), sq("b6"), MoveType.CHECKER_JUMP)
    assert mv1 in pos.legal_moves()
    pos.make(mv1)
    assert pos.side_to_move == Side.CHECKERS
    assert pos.jump_from == sq("b6")
    d2 = dests_from(pos, sq("b6"))
    assert (sq("b6"), sq("d4"), MoveType.CHECKER_JUMP) in d2
    pos.make(encode(sq("b6"), sq("d4"), MoveType.CHECKER_JUMP))
    assert pos.side_to_move == Side.CHESS
    assert pos.jump_from is None


def test_three_hop_chain():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("h8")] = "c"
    pos.board[sq("g7")] = "P"
    pos.board[sq("g5")] = "N"
    pos.board[sq("g3")] = "B"
    mv1 = encode(sq("h8"), sq("f6"), MoveType.CHECKER_JUMP)
    pos.make(mv1)
    assert pos.jump_from == sq("f6")
    mv2 = encode(sq("f6"), sq("h4"), MoveType.CHECKER_JUMP)
    assert mv2 in pos.legal_moves()
    pos.make(mv2)
    assert pos.jump_from == sq("h4")
    d3 = dests_from(pos, sq("h4"))
    assert (sq("h4"), sq("f2"), MoveType.CHECKER_JUMP) in d3
    pos.make(encode(sq("h4"), sq("f2"), MoveType.CHECKER_JUMP))
    assert pos.side_to_move == Side.CHESS
    assert pos.jump_from is None
    assert pos.board[sq("g7")] == "." and pos.board[sq("g5")] == "." and pos.board[sq("e5")] == "."


def test_mandatory_capture_on_hides_quiet():
    cfg = RuleConfig(mandatory_capture=True)
    pos = blank(cfg)
    pos.board[sq("d6")] = "c"
    pos.board[sq("c5")] = "P"
    pos.board[sq("h6")] = "c"  # has a quiet step available but must not be offered
    moves = [decode(m) for m in pos.legal_moves()]
    assert all(mt == MoveType.CHECKER_JUMP for _f, _t, mt in moves)


def test_mandatory_capture_off_shows_quiet():
    cfg = RuleConfig(mandatory_capture=False)
    pos = blank(cfg)
    pos.board[sq("d6")] = "c"
    pos.board[sq("c5")] = "P"
    moves = [decode(m) for m in pos.legal_moves()]
    types = {mt for _f, _t, mt in moves}
    assert MoveType.CHECKER_JUMP in types
    assert MoveType.CHECKER_STEP in types


def test_maximal_capture_only_longest_offered():
    cfg = RuleConfig(mandatory_capture=True, maximal_capture=True)
    pos = blank(cfg)
    # piece A: single jump (1 capture)
    pos.board[sq("a6")] = "c"
    pos.board[sq("b5")] = "P"
    # piece B: two-hop chain (2 captures)
    pos.board[sq("d8")] = "c"
    pos.board[sq("c7")] = "P"
    pos.board[sq("c5")] = "N"
    moves = [decode(m) for m in pos.legal_moves()]
    froms = {f for f, _t, _mt in moves}
    assert sq("a6") not in froms
    assert sq("d8") in froms


def test_promotion_on_rank1():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("b2")] = "c"
    mv = encode(sq("b2"), sq("a1"), MoveType.CHECKER_STEP)
    pos.make(mv)
    assert pos.board[sq("a1")] == "C"


def test_promotion_mid_chain_ends_chain():
    # e5 -(capture d4)-> c3 -(capture d2)-> e1 (rank 1, promotes). A backward
    # jump e1-over-f2-landing-g3 would be available to a king, proving the
    # chain-ending is caused by promotion, not by lack of a further jump.
    cfg = RuleConfig(promotion_ends_jump_chain=True)
    pos = blank(cfg)
    pos.board[sq("e5")] = "c"
    pos.board[sq("d4")] = "P"
    pos.board[sq("d2")] = "B"
    pos.board[sq("f2")] = "N"
    mv1 = encode(sq("e5"), sq("c3"), MoveType.CHECKER_JUMP)
    pos.make(mv1)
    assert pos.jump_from == sq("c3")
    mv2 = encode(sq("c3"), sq("e1"), MoveType.CHECKER_JUMP)
    assert mv2 in pos.legal_moves()
    pos.make(mv2)
    assert pos.board[sq("e1")] == "C"
    assert pos.jump_from is None
    assert pos.side_to_move == Side.CHESS
    assert pos.board[sq("f2")] == "N"  # never captured: chain ended at promotion


def test_king_moves_and_jumps_backward_man_does_not():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("d4")] = "C"
    d = dests_from(pos, sq("d4"))
    tos = {t for _f, t, mt in d if mt == MoveType.CHECKER_STEP}
    assert sq("c5") in tos and sq("e5") in tos and sq("c3") in tos and sq("e3") in tos

    pos2 = blank(cfg)
    pos2.board[sq("d4")] = "c"
    d2 = dests_from(pos2, sq("d4"))
    tos2 = {t for _f, t, mt in d2 if mt == MoveType.CHECKER_STEP}
    assert sq("c5") not in tos2 and sq("e5") not in tos2
    assert sq("c3") in tos2 and sq("e3") in tos2


def test_king_backward_jump():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("d4")] = "C"
    pos.board[sq("e5")] = "P"
    d = dests_from(pos, sq("d4"))
    jumps = {t for _f, t, mt in d if mt == MoveType.CHECKER_JUMP}
    assert sq("f6") in jumps


def test_flying_king_movement_and_capture():
    cfg_fixed = RuleConfig(flying_kings=False)
    pos = blank(cfg_fixed)
    pos.board[sq("a1")] = "C"
    d = dests_from(pos, sq("a1"))
    tos = {t for _f, t, mt in d if mt == MoveType.CHECKER_STEP}
    assert tos == {sq("b2")}

    cfg_fly = RuleConfig(flying_kings=True)
    pos2 = blank(cfg_fly)
    pos2.board[sq("a1")] = "C"
    d2 = dests_from(pos2, sq("a1"))
    tos2 = {t for _f, t, mt in d2 if mt == MoveType.CHECKER_STEP}
    assert tos2 == {sq("b2"), sq("c3"), sq("d4"), sq("e5"), sq("f6"), sq("g7"), sq("h8")}

    pos3 = blank(cfg_fly)
    pos3.board[sq("a1")] = "C"
    pos3.board[sq("d4")] = "P"
    d3 = dests_from(pos3, sq("a1"))
    jumps3 = {t for _f, t, mt in d3 if mt == MoveType.CHECKER_JUMP}
    assert jumps3 == {sq("e5"), sq("f6"), sq("g7"), sq("h8")}

    pos4 = blank(cfg_fixed)
    pos4.board[sq("a1")] = "C"
    pos4.board[sq("d4")] = "P"
    d4 = dests_from(pos4, sq("a1"))
    jumps4 = {t for _f, t, mt in d4 if mt == MoveType.CHECKER_JUMP}
    assert jumps4 == set()  # not adjacent, fixed king cannot reach it


def test_cannot_jump_friendly_checker():
    cfg = RuleConfig()
    pos = blank(cfg)
    pos.board[sq("d6")] = "c"
    pos.board[sq("c5")] = "c"
    d = dests_from(pos, sq("d6"))
    jumps = {t for _f, t, mt in d if mt == MoveType.CHECKER_JUMP}
    assert jumps == set()
