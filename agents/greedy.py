"""Spec 8.2: one-ply greedy, maximize the side's evaluation function, random tiebreak."""
from __future__ import annotations

import random

from engine.board import Position, Side
from agents.evals import chess_side_eval, checkers_side_eval


class GreedyAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select(self, pos: Position) -> int:
        moves = pos.legal_moves()
        if not moves:
            raise ValueError("no legal moves available")
        eval_fn = chess_side_eval if pos.side_to_move == Side.CHESS else checkers_side_eval

        best_score = float("-inf")
        best_moves = []
        for mv in moves:
            undo = pos.make(mv)
            score = eval_fn(pos)
            pos.unmake(undo)
            if score > best_score:
                best_score = score
                best_moves = [mv]
            elif score == best_score:
                best_moves.append(mv)
        return self.rng.choice(best_moves)
