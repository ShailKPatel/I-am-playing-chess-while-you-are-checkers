"""Spec 8.2: iterative deepening alpha-beta with a Zobrist-keyed transposition
table and captures-first move ordering. Respects a depth limit and a wall-clock
time limit.

The two sides chase different, non-opposite victory conditions (spec section 1),
so this is not a standard negamax search. It is a minimax search that maximizes
the *root side's* eval_fn at nodes where the root side is to move, and minimizes
it (treats the opponent as adversarial toward that same score) otherwise.
"""
from __future__ import annotations

import time
from typing import Callable

from engine.board import Position, Side
from engine.moves import decode
from engine.moves import IS_CAPTURE

EvalFn = Callable[[Position], float]

_EXACT, _LOWER, _UPPER = 0, 1, 2


class AlphaBetaAgent:
    def __init__(self, depth: int, eval_fn: EvalFn, time_limit_seconds: float | None = 5.0):
        self.max_depth = depth
        self.eval_fn = eval_fn
        self.time_limit_seconds = time_limit_seconds
        self.tt: dict[int, tuple[int, float, int, int]] = {}

    def select(self, pos: Position) -> int:
        moves = pos.legal_moves()
        if not moves:
            raise ValueError("no legal moves available")
        if len(moves) == 1:
            return moves[0]

        root_side = pos.side_to_move
        deadline = None if self.time_limit_seconds is None else time.monotonic() + self.time_limit_seconds
        self.tt.clear()

        best_move = moves[0]
        depth = 1
        try:
            while depth <= self.max_depth:
                score, mv = self._search_root(pos, depth, root_side, deadline)
                if mv is not None:
                    best_move = mv
                depth += 1
        except _TimeUp:
            pass
        return best_move

    def _search_root(self, pos: Position, depth: int, root_side: Side, deadline):
        moves = self._ordered_moves(pos)
        alpha, beta = float("-inf"), float("inf")
        best_score = float("-inf")
        best_move = None
        for mv in moves:
            undo = pos.make(mv)
            score = self._search(pos, depth - 1, alpha, beta, root_side, deadline)
            pos.unmake(undo)
            if score > best_score:
                best_score = score
                best_move = mv
            alpha = max(alpha, score)
        return best_score, best_move

    def _ordered_moves(self, pos: Position) -> list[int]:
        moves = pos.legal_moves()
        def key(mv):
            _f, _t, mt = decode(mv)
            return 0 if mt in IS_CAPTURE else 1
        return sorted(moves, key=key)

    def _search(self, pos: Position, depth: int, alpha: float, beta: float, root_side: Side, deadline) -> float:
        if deadline is not None and time.monotonic() > deadline:
            raise _TimeUp()

        if pos.result() is not None or depth == 0:
            return self.eval_fn(pos)

        key = pos.key()
        tt_entry = self.tt.get(key)
        if tt_entry is not None and tt_entry[0] >= depth:
            _d, score, flag, _mv = tt_entry
            if flag == _EXACT:
                return score
            if flag == _LOWER:
                alpha = max(alpha, score)
            elif flag == _UPPER:
                beta = min(beta, score)
            if alpha >= beta:
                return score

        maximizing = pos.side_to_move == root_side
        moves = self._ordered_moves(pos)
        orig_alpha, orig_beta = alpha, beta

        if maximizing:
            value = float("-inf")
            for mv in moves:
                undo = pos.make(mv)
                value = max(value, self._search(pos, depth - 1, alpha, beta, root_side, deadline))
                pos.unmake(undo)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for mv in moves:
                undo = pos.make(mv)
                value = min(value, self._search(pos, depth - 1, alpha, beta, root_side, deadline))
                pos.unmake(undo)
                beta = min(beta, value)
                if alpha >= beta:
                    break

        if value <= orig_alpha:
            flag = _UPPER
        elif value >= orig_beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self.tt[key] = (depth, value, flag, 0)
        return value


class _TimeUp(Exception):
    pass
