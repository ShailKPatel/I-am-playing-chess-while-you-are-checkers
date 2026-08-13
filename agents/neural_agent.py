"""Loads the two checkpoints produced by bot_training/train.py and uses them
to pick moves in the deployed game (CLI/API bot mode) — the same chess_net.pt
/ checkers_net.pt trained on Colab, no separate copy.

torch is imported lazily (inside __init__/select, not at module load time):
the web image stays torch-free unless agent="neural" is actually requested
(Dockerfile deliberately keeps torch out of the deployed image).

The network architecture here must exactly match scripts/train_logic.py's
PolicyValueNet (the source bot_training/train.py is generated from) —
duplicated rather than imported, because bot_training/ is a standalone,
upload-to-Colab bundle, not a dependency of this app.
"""
from __future__ import annotations

import os

from engine.board import Position, Side
from engine.moves import decode, MoveType

DEFAULT_CHESS_PATH = "bot_training/chess_net.pt"
DEFAULT_CHECKERS_PATH = "bot_training/checkers_net.pt"


class NeuralAgent:
    def __init__(self, side: Side, path: str | None = None, device: str = "cpu"):
        import torch
        import torch.nn as nn

        class ResidualBlock(nn.Module):
            def __init__(self, channels: int, groups: int = 8):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.norm1 = nn.GroupNorm(groups, channels)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                self.norm2 = nn.GroupNorm(groups, channels)

            def forward(self, x):
                residual = x
                out = nn.functional.relu(self.norm1(self.conv1(x)))
                out = self.norm2(self.conv2(out))
                return nn.functional.relu(out + residual)

        class PolicyValueNet(nn.Module):
            def __init__(self, in_channels: int = 19, channels: int = 128, num_blocks: int = 10):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.ReLU(inplace=True),
                )
                self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
                flat = channels * 8 * 8
                self.from_head = nn.Linear(flat, 64)
                self.to_head = nn.Linear(flat, 64)
                self.type_head = nn.Linear(flat, len(MoveType))
                self.value_head = nn.Sequential(
                    nn.Linear(flat, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Tanh(),
                )

            def forward(self, planes):
                x = self.stem(planes)
                x = self.tower(x)
                x = x.flatten(1)
                value = self.value_head(x).squeeze(-1)
                return self.from_head(x), self.to_head(x), self.type_head(x), value

        self.device = device
        # Sizes must match train_logic.py's train() exactly per side — chess
        # and checkers are NOT the same size (checkers' task is structurally
        # harder, gets more capacity; see train_logic.py).
        channels, num_blocks = (128, 10) if side == Side.CHESS else (160, 13)
        self.net = PolicyValueNet(channels=channels, num_blocks=num_blocks).to(device)

        weight_path = path or (DEFAULT_CHESS_PATH if side == Side.CHESS else DEFAULT_CHECKERS_PATH)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"no trained weights at {weight_path!r} — train with bot_training/train.py and place "
                f"chess_net.pt / checkers_net.pt there (or pass an explicit path)"
            )
        payload = torch.load(weight_path, map_location=device)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def select(self, pos: Position) -> int:
        import torch

        legal = pos.legal_moves()
        if not legal:
            raise ValueError("no legal moves available")

        x = torch.from_numpy(pos.to_planes()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            from_logits, to_logits, type_logits, _value = self.net(x)

        frms, tos, mtypes = zip(*(decode(mv) for mv in legal))
        frms_t = torch.tensor(frms, dtype=torch.long)
        tos_t = torch.tensor(tos, dtype=torch.long)
        mtypes_t = torch.tensor([int(t) for t in mtypes], dtype=torch.long)
        scores = from_logits[0][frms_t] + to_logits[0][tos_t] + type_logits[0][mtypes_t]
        idx = int(torch.argmax(scores))  # greedy at inference time, no sampling
        return legal[idx]
