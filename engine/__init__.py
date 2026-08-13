"""Public surface re-exported flat, so `from engine import Position` works
identically against this real multi-file package and against the single-file
bundle generated for training (bot_training/build_bundle.py concatenates
engine/*.py into one engine.py with the same flat names)."""
from engine.board import Position, Side, Result
from engine.config import RuleConfig
from engine.moves import MoveType, encode, decode, square_name, square_from_name, move_to_uci

__all__ = [
    "Position", "Side", "Result", "RuleConfig",
    "MoveType", "encode", "decode", "square_name", "square_from_name", "move_to_uci",
]
