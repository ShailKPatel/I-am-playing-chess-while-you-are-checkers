"""Regenerates bot_training/train.py: engine/*.py + scripts/train_logic.py,
flattened into ONE dependency-free (numpy + torch only) file. Run this
locally whenever engine/ or scripts/train_logic.py changes:

    python scripts/build_bot_training.py

This script and scripts/train_logic.py are dev-only sources — neither is
itself uploaded to Colab. bot_training/ ends up holding exactly the 3 files
that matter: train.py, chess_net.pt, checkers_net.pt (the last two appear
after the first training run).

Uses ast (not text/regex): internal `from engine.x import y` / `from engine
import y` lines are dropped correctly (their names already land in the same
flat namespace after concatenation) and every external import (numpy, torch,
random, ...) is deduplicated and hoisted to the top exactly once, wherever in
the source tree it appeared — including nested inside function bodies, e.g.
engine/board.py's lazy `from engine.legal import legal_moves`.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "engine"
TRAIN_LOGIC = Path(__file__).resolve().parent / "train_logic.py"
OUT_FILE = REPO_ROOT / "bot_training" / "train.py"

# Dependency order: each file may only reference names defined in an earlier
# file (forward references inside function *bodies* are fine either way,
# since Python resolves globals at call time, not definition time).
ENGINE_SOURCE_FILES = [
    "config.py",
    "moves.py",
    "board.py",
    "attacks.py",
    "chess_gen.py",
    "checkers_gen.py",
    "legal.py",
    "terminal.py",
    "serialize.py",
    "planes.py",
]


class ImportStripper(ast.NodeTransformer):
    """Drops internal `engine` imports (already in-namespace after concatenation,
    whether `from engine.x import y` or `from engine import y`) and every
    `__future__` import (hoisted once, manually); collects every external
    import — wherever it appears, top-level or nested — for deduplicated
    re-emission at the top of the bundle."""

    def __init__(self):
        self.plain_imports: set[tuple[str, str | None]] = set()
        self.from_imports: dict[str, set[tuple[str, str | None]]] = {}

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name == "engine" or alias.name.startswith("engine."):
                continue
            self.plain_imports.add((alias.name, alias.asname))
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == "__future__":
            return None
        if node.module is not None and (node.module == "engine" or node.module.startswith("engine.")):
            return None
        self.from_imports.setdefault(node.module, set())
        for alias in node.names:
            self.from_imports[node.module].add((alias.name, alias.asname))
        return None


def _stripped_body(path: Path, stripper: ImportStripper, label: str) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    tree = stripper.visit(tree)
    ast.fix_missing_locations(tree)
    return f"# ==== {label} ====\n" + ast.unparse(tree)


def build() -> str:
    stripper = ImportStripper()
    bodies = [_stripped_body(ENGINE_DIR / f, stripper, f"engine/{f}") for f in ENGINE_SOURCE_FILES]
    bodies.append(_stripped_body(TRAIN_LOGIC, stripper, "training loop"))

    header_lines = [
        '"""Chesskers self-play training — single file, upload with the two',
        ".pt checkpoints (created on first run) and nothing else.",
        "",
        "    !pip install torch numpy",
        "    !python train.py --games 100",
        "",
        "chess_net always plays the chess army and only learns from chess-side",
        "trajectories; checkers_net is the same for checkers. They never swap",
        "sides, never share weights or gradients — one optimizer each.",
        "",
        "Actor-critic self-play (policy gradient + a learned value head), no",
        "search. Reward is terminal-only: +1 / -1 / 0, no shaping. A shorter",
        "winning game is preferred through the discount factor (--gamma) on",
        "that side's own move sequence, not a second reward term.",
        "",
        "Checkpoints (chess_net.pt / checkers_net.pt) are written atomically —",
        "a temp file, then an atomic rename — after every --checkpoint-every",
        "games (default: every game). If the runtime dies mid-game or",
        "mid-write, the file on disk is still exactly the last FULLY",
        "COMPLETED game's weights, never a half-written file, and the next",
        "run auto-resumes from it (each checkpoint also carries a persistent",
        "games-trained counter, so game counts keep incrementing correctly",
        "across separate Colab sessions instead of restarting at 1). Only",
        "model weights are checkpointed, not optimizer momentum, so a resume",
        "is not bit-identical to an uninterrupted run, but is always valid.",
        "",
        "Progress prints as one self-overwriting status line per game (not a",
        "line per game — this can be thousands of games) plus a permanent",
        "log line every --log-every games, so a crash leaves a short, readable",
        "trail instead of a huge scrollback.",
        "",
        "RuleConfig here is plain defaults, matching the deployed server's",
        "default ACTIVE_CONFIG (engine/config.py: the whole point of the",
        "config hash is to make a rules mismatch loud, not silent — check the",
        "printed hash against the server's GET /api/config if it's been",
        'changed from defaults).',
        "",
        "Auto-generated by scripts/build_bot_training.py from engine/*.py +",
        'scripts/train_logic.py. Do not hand-edit — re-run that script after',
        'changing either source."""',
        "from __future__ import annotations",
        "",
    ]
    for name, asname in sorted(stripper.plain_imports):
        header_lines.append(f"import {name}" + (f" as {asname}" if asname else ""))
    for module in sorted(stripper.from_imports):
        names = sorted(stripper.from_imports[module])
        rendered = ", ".join(n + (f" as {a}" if a else "") for n, a in names)
        header_lines.append(f"from {module} import {rendered}")

    return "\n".join(header_lines) + "\n\n" + "\n\n\n".join(bodies) + "\n"


if __name__ == "__main__":
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(build())
    print(f"wrote {OUT_FILE}")
