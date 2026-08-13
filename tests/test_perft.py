"""Perft: proves the chess move generator correct on castling, en passant,
promotion, pins and check evasion. Spec section 9.1."""
import pytest

from engine.board import Position, STANDARD_FEN
from engine.config import RuleConfig

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"

STANDARD_COUNTS = [20, 400, 8902, 197281, 4865609]
KIWIPETE_COUNTS = [48, 2039, 97862, 4085603]


def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    n = 0
    for mv in pos.legal_moves():
        undo = pos.make(mv)
        n += perft(pos, depth - 1)
        pos.unmake(undo)
    return n


@pytest.fixture
def config():
    return RuleConfig(first_mover="chess")


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_perft_standard(config, depth):
    pos = Position.from_fen(STANDARD_FEN, config)
    assert perft(pos, depth) == STANDARD_COUNTS[depth - 1]


@pytest.mark.slow
def test_perft_standard_depth5(config):
    pos = Position.from_fen(STANDARD_FEN, config)
    assert perft(pos, 5) == STANDARD_COUNTS[4]


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_perft_kiwipete(config, depth):
    pos = Position.from_fen(KIWIPETE, config)
    assert perft(pos, depth) == KIWIPETE_COUNTS[depth - 1]


@pytest.mark.slow
def test_perft_kiwipete_depth4(config):
    pos = Position.from_fen(KIWIPETE, config)
    assert perft(pos, 4) == KIWIPETE_COUNTS[3]
