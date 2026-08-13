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

    def config_hash(self) -> str:
        """Short stable hash identifying this exact rule set (spec 7.5)."""
        payload = repr(sorted(asdict(self).items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:10]

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "RuleConfig":
        return RuleConfig(**d)
