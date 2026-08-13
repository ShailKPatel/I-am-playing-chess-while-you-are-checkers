"""Gym-style environment for the checkers side. Spec section 7.7.

Unused by anything today (neural-net training is out of scope, spec section 0);
built now so that phase has a stable interface to target.
"""
from __future__ import annotations

import numpy as np

from engine.board import Position, Side, Result
from engine.config import RuleConfig

ACTION_SPACE_SIZE = 1 << 16  # full packed-move range (spec 7.3)


class CheckersSideEnv:
    action_space_size = ACTION_SPACE_SIZE

    def __init__(self, config: RuleConfig | None = None):
        self.config = config or RuleConfig()
        self.pos: Position | None = None

    def reset(self, seed: int | None = None):
        import random
        if seed is not None:
            random.seed(seed)
        self.pos = Position(self.config)
        return self.pos.to_planes()

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=bool)
        if self.pos.side_to_move != Side.CHECKERS:
            return mask
        for mv in self.pos.legal_moves():
            mask[mv] = True
        return mask

    def step(self, action: int):
        if self.pos.side_to_move != Side.CHECKERS:
            raise ValueError("not the checkers side's turn")
        if action not in self.pos.legal_moves():
            raise ValueError(f"illegal action: {action}")
        self.pos.make(action)
        result = self.pos.result()
        done = result is not None
        reward = 0.0
        if result == Result.CHECKERS_WIN:
            reward = 1.0
        elif result == Result.CHESS_WIN:
            reward = -1.0
        return self.pos.to_planes(), reward, done, {}
