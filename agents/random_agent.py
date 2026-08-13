"""Spec 8.2: uniform random over legal moves."""
from __future__ import annotations

import random

from engine.board import Position


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select(self, pos: Position) -> int:
        moves = pos.legal_moves()
        if not moves:
            raise ValueError("no legal moves available")
        return self.rng.choice(moves)
