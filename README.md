---
title: I am playing chess while you are playing checkers
emoji: ♟️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# I am playing chess while you are playing checkers

One board, two armies, two swapped victory conditions. White commands a
standard chess army but wins the way a **checkers** game is won: capture
every enemy piece, or leave the opponent with no legal move. Black commands
24 checker men but wins the way a **chess** game is won: checkmate the
enemy king. See [static/rules.html](static/rules.html) (or `/rules.html` on
a running deployment) for the full ruleset, including why the checkers army
starts on both light and dark squares.

## Layout

```
chesskers/
  engine/         rules engine — board, movegen, legality, terminal conditions, serialization
  envs/           Gym-style wrappers (unused today; interface for a future RL phase)
  agents/         RandomAgent, GreedyAgent, AlphaBetaAgent, NeuralAgent, evaluation functions
  cli/            chesskers perft / play / selfplay / match
  api/            stateless FastAPI service
  static/         vanilla-JS frontend (no build step)
  tests/          perft, differential-vs-python-chess, rule and invariant tests
  scripts/        dev-only: regenerates bot_training/train.py from engine/ + scripts/train_logic.py
  bot_training/   the 3-file Colab upload for self-play training — see below
```

`engine/` has no dependency on any web or serving library — `import engine`
works with only `numpy` installed. `api/`, `cli/`, `agents/`, and `envs/` are
all peers that depend on `engine/`, not layers on top of each other or each
other.

## Install

```bash
pip install -e ".[dev]"   # engine + web + tests, for local development
```

## Run the tests

```bash
pytest                 # perft depths 1-4, differential, rules, traps, terminal, invariants
pytest -m slow          # + perft depth 5 and Kiwipete depth 4, + a 100k-iteration invariant sweep
```

## CLI

```bash
chesskers perft --depth 5
chesskers play --side chess --vs alphabeta:4
chesskers selfplay --agents random,random --n 1000 --out results.jsonl
chesskers match --a alphabeta:3 --b random --games 100
```

## Run the web app

```bash
pip install -e ".[web]"
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/`.

## Training (Colab)

`bot_training/` is a self-contained, 3-file upload for self-play training —
nothing else from this repo is needed there:

- `train.py` — auto-generated (`python scripts/build_bot_training.py`) from
  `engine/*.py` + `scripts/train_logic.py`, flattened into one
  dependency-free (numpy + torch only) file. Two independent networks —
  `chess_net` only ever plays and learns as the chess army, `checkers_net`
  only ever plays and learns as the checkers army, never crossed, never
  shared. Same architecture for both: 10 residual blocks x 128 filters
  (~5.2M params), the same family AlphaZero/Leela Chess Zero use, sized to
  what a free Colab session can actually train on rather than their full
  scale.
- `chess_net.pt`, `checkers_net.pt` — created automatically on first run,
  loaded and continued from on every run after. Checkpoints are written
  atomically (temp file + atomic rename) after every game by default, and
  each one carries a persistent games-trained counter, so resuming in a new
  Colab session on different files correctly continues the count instead of
  restarting at zero.

```bash
cd bot_training
pip install torch numpy
python train.py --games 100
```

| outcome | chess points | checkers points |
|---|---|---|
| win | 150 (flat) | 150 (flat) |
| loss | material captured (0 to 24) | material captured (0 to ~39, ~103 pathological) |
| draw | material captured − 12 | material captured − 20 |

Reward is win = a flat 150-point bonus (never combined with material). A
loss earns partial credit for material actually captured that game, always
>= 0 — chess gets 1 point per checker piece captured (max 24), checkers gets
standard chess piece values for chess pieces captured (pawn 1, knight 3,
bishop 3, rook 5, queen 9, capped so even every pawn promoting to a queen
tops out at 103 — comfortably under the win bonus). Losing is not
inherently negative under this scheme: losing while capturing a lot is
meant to score meaningfully better than losing badly, while always staying
far below the win bonus so partial credit can never substitute for actually
winning.

A draw gets the same material credit minus a draw-specific penalty (-12 for
chess, -20 for checkers — 50% of each side's own realistic maximum credit,
scaled per side since their ceilings are on very different scales, not a
shared flat number) that a loss does not get. This can make a hard-fought
loss score above a cheap draw — intentional: self-play was converging on a
passive, mutually "safe" repetition draw (not through any coordination
between the two independently-optimized networks, but because both could
separately find drawing locally safer than risking a real result once the
reward function didn't distinguish a cheap draw from a hard-fought one), and
the fix targets that specific outcome rather than discouraging losses in
general — punishing loss instead would only make avoiding any loss risk
(i.e. avoiding decisive play) more attractive, the opposite of the goal.

Speed preference runs through the discount factor (`--gamma`), not a
separate reward term — it only ever discounts a real outcome, so there's no
way to farm reward by ending a game quickly in any state.

By default every run appends one row per game to `training_log.csv`
(`game, timestamp, first_mover, winner, plies, seconds, chess_points,
checkers_points, reason`) — pass `--no-log` to skip it. `RuleConfig` here is
plain defaults, matching the deployed server's default `ACTIVE_CONFIG`
(`engine/config.py`'s config hash makes a mismatch loud, not silent — check
the printed hash against the server's `GET /api/config` if that's been
changed).

Once trained, drop `chess_net.pt` / `checkers_net.pt` into `bot_training/`
locally and the deployed game's bot mode uses them directly
(`agents/neural_agent.py`, `agent="neural"` in the API/CLI) — no separate
copy or extra wiring.

## Docker

```bash
docker build -t chesskers .
docker run -p 7860:7860 chesskers
```

Built for Hugging Face Spaces' Docker SDK: single container, stateless
(client holds the game state string), nothing written outside `/tmp`.
