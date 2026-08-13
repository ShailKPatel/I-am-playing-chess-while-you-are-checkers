"""RuleConfig: the single source of truth for every rule variant. See spec section 7.1."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RuleConfig:
    """Every rule variant this game supports (spec 7.1). Immutable, embedded
    in every serialized state and every self-play/match record, so a model or
    client can never silently disagree with the server about the ruleset."""
    mandatory_capture: bool = True
    maximal_capture: bool = False
    flying_kings: bool = False
    layout: str = "standard_24v16"
    first_mover: str = "random"  # chess | checkers | random
    fifty_move_plies: int = 100
    promotion_ends_jump_chain: bool = True
    king_capture_immunity: bool = False
    """Was a temporary experiment flag, now the live default (2026-08-13):
    checkers went 0-1015 while True. True = legacy behaviour, chess's king
    protected two ways checkers gets no equivalent for — legal.py filters out
    any chess move leaving the king attacked (info checkers never gets about
    its own pieces), and checkers_gen.py refuses to ever generate a jump that
    captures a K square at all. False (current default) = symmetric: chess
    can walk into an attack same as checkers can lose a piece, and a checkers
    jump onto/through K is a real, game-ending capture."""

    def config_hash(self) -> str:
        """Short stable hash identifying this exact rule set (spec 7.5)."""
        payload = repr(sorted(asdict(self).items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:10]

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "RuleConfig":
        return RuleConfig(**d)
