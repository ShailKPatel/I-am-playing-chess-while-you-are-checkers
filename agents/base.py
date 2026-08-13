"""Agent protocol. Spec section 8.1. Every agent, human-facing or trained, implements this."""
from __future__ import annotations

from typing import Protocol

from engine.board import Position


class Agent(Protocol):
    def select(self, pos: Position) -> int: ...
