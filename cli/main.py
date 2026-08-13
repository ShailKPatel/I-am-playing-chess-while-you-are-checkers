"""Chesskers CLI. Spec section 10.

perft, play, selfplay and match are how the game gets balance-tested and how
future models get evaluated, not just conveniences.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

from engine.board import Position, Side, Result, STANDARD_FEN
from engine.config import RuleConfig
from engine.moves import move_to_uci
from agents.random_agent import RandomAgent
from agents.greedy import GreedyAgent
from agents.alphabeta import AlphaBetaAgent
from agents.evals import chess_side_eval, checkers_side_eval


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def load_config(path: str | None) -> RuleConfig:
    if not path:
        return RuleConfig()
    with open(path) as f:
        return RuleConfig.from_json(json.load(f))


def make_agent(spec: str, side: Side, seed: int):
    """Parses agent specs like 'random', 'greedy', 'alphabeta:4'."""
    parts = spec.split(":")
    name = parts[0].lower()
    eval_fn = chess_side_eval if side == Side.CHESS else checkers_side_eval
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyAgent(seed=seed)
    if name == "alphabeta":
        depth = int(parts[1]) if len(parts) > 1 else 3
        return AlphaBetaAgent(depth=depth, eval_fn=eval_fn, time_limit_seconds=1.5)
    if name == "neural":
        from agents.neural_agent import NeuralAgent
        path = parts[1] if len(parts) > 1 else None
        return NeuralAgent(side=side, path=path)
    raise ValueError(f"unknown agent spec: {spec}")


def render_board(pos: Position) -> str:
    lines = []
    for r in range(7, -1, -1):
        row = [pos.board[r * 8 + f] for f in range(8)]
        lines.append(f"{r + 1}  " + " ".join(ch if ch != "." else "." for ch in row))
    lines.append("   " + " ".join("abcdefgh"))
    return "\n".join(lines)


def result_name(result: Result) -> str:
    return {Result.CHESS_WIN: "chess", Result.CHECKERS_WIN: "checkers", Result.DRAW: "draw"}[result]


# ---------------------------------------------------------------------------
# perft
# ---------------------------------------------------------------------------
def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    n = 0
    for mv in pos.legal_moves():
        undo = pos.make(mv)
        n += perft(pos, depth - 1)
        pos.unmake(undo)
    return n


def cmd_perft(args):
    cfg = load_config(args.config)
    cfg = RuleConfig(**{**cfg.to_json(), "first_mover": "chess"})
    fen = args.fen or STANDARD_FEN
    pos = Position.from_fen(fen, cfg)
    t0 = time.monotonic()
    for d in range(1, args.depth + 1):
        n = perft(pos, d)
        dt = time.monotonic() - t0
        print(f"depth {d}: {n} nodes ({dt:.2f}s total)")


# ---------------------------------------------------------------------------
# play
# ---------------------------------------------------------------------------
def cmd_play(args):
    cfg = load_config(args.config)
    if args.seed is not None:
        random.seed(args.seed)
    if args.state:
        from engine.serialize import deserialize
        pos = deserialize(args.state, cfg)
    else:
        pos = Position(cfg)
    human_side = Side.CHESS if args.side == "chess" else Side.CHECKERS
    bot_side = Side.CHECKERS if human_side == Side.CHESS else Side.CHESS
    bot = make_agent(args.vs, bot_side, args.seed or 0)

    while True:
        result = pos.result()
        print()
        print(render_board(pos))
        if result is not None:
            print(f"\nGame over: {result_name(result)} wins" if result != Result.DRAW else "\nGame over: draw")
            return
        turn = "chess" if pos.side_to_move == Side.CHESS else "checkers"
        print(f"\n{turn} to move" + (" (jump chain in progress)" if pos.jump_from is not None else ""))

        if pos.side_to_move == human_side:
            legal = pos.legal_moves()
            uci_map = {move_to_uci(mv): mv for mv in legal}
            print("legal moves: " + ", ".join(sorted(uci_map)))
            move_str = input("your move (e.g. e2e4): ").strip()
            if move_str not in uci_map:
                print("illegal or unrecognized move")
                continue
            pos.make(uci_map[move_str])
        else:
            mv = bot.select(pos)
            print(f"bot plays {move_to_uci(mv)}")
            pos.make(mv)


# ---------------------------------------------------------------------------
# selfplay
# ---------------------------------------------------------------------------
def play_one_game(cfg: RuleConfig, agent_chess, agent_checkers, seed: int, max_plies: int = 2000):
    random.seed(seed)
    pos = Position(cfg)
    plies = 0
    while pos.result() is None and plies < max_plies:
        agent = agent_chess if pos.side_to_move == Side.CHESS else agent_checkers
        mv = agent.select(pos)
        pos.make(mv)
        plies += 1
    result = pos.result()
    return result, plies


def cmd_selfplay(args):
    cfg = load_config(args.config)
    base_seed = args.seed if args.seed is not None else random.randrange(2**31)
    names = args.agents.split(",")
    if len(names) != 2:
        raise ValueError("--agents expects exactly two comma-separated specs: chess,checkers")

    counts = {"chess": 0, "checkers": 0, "draw": 0}
    with open(args.out, "w") as f:
        for i in range(args.n):
            seed = base_seed + i
            agent_chess = make_agent(names[0], Side.CHESS, seed)
            agent_checkers = make_agent(names[1], Side.CHECKERS, seed)
            result, plies = play_one_game(cfg, agent_chess, agent_checkers, seed)
            outcome = result_name(result) if result is not None else "unterminated"
            counts[outcome] = counts.get(outcome, 0) + 1
            f.write(json.dumps({
                "game": i, "result": outcome, "plies": plies,
                "config": cfg.to_json(), "seed": seed,
            }) + "\n")

    print(f"{args.n} games written to {args.out}")
    print(f"{'side':<10}{'wins':<8}{'rate':<8}")
    for side in ("chess", "checkers", "draw"):
        print(f"{side:<10}{counts.get(side, 0):<8}{counts.get(side, 0) / args.n:<8.3f}")


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------
def _elo_diff(score: float, games: int) -> tuple[float, float]:
    """Elo difference implied by a score rate, with a rough 95% CI half-width."""
    eps = 1e-6
    p = min(max(score, eps), 1 - eps)
    diff = -400 * math.log10(1 / p - 1)
    se = math.sqrt(p * (1 - p) / games)
    p_hi = min(max(score + 1.96 * se, eps), 1 - eps)
    p_lo = min(max(score - 1.96 * se, eps), 1 - eps)
    diff_hi = -400 * math.log10(1 / p_hi - 1)
    diff_lo = -400 * math.log10(1 / p_lo - 1)
    return diff, (diff_hi - diff_lo) / 2


def cmd_match(args):
    cfg = load_config(args.config)
    base_seed = args.seed if args.seed is not None else random.randrange(2**31)

    a_wins = b_wins = draws = 0
    for i in range(args.games):
        seed = base_seed + i
        a_is_chess = (i < args.games // 2) if args.swap_sides else True
        chess_spec, checkers_spec = (args.a, args.b) if a_is_chess else (args.b, args.a)
        agent_chess = make_agent(chess_spec, Side.CHESS, seed)
        agent_checkers = make_agent(checkers_spec, Side.CHECKERS, seed)
        result, _plies = play_one_game(cfg, agent_chess, agent_checkers, seed)

        if result == Result.DRAW or result is None:
            draws += 1
            continue
        chess_won = result == Result.CHESS_WIN
        a_won = chess_won if a_is_chess else not chess_won
        if a_won:
            a_wins += 1
        else:
            b_wins += 1

    decisive = a_wins + b_wins
    print(f"A ({args.a}) wins: {a_wins}   B ({args.b}) wins: {b_wins}   draws: {draws}")
    if decisive > 0:
        score = (a_wins + 0.5 * draws) / args.games
        diff, ci = _elo_diff(score, args.games)
        print(f"Elo difference (A - B): {diff:+.1f} +/- {ci:.1f} (95% CI)")
    else:
        print("Elo difference: undefined (no decisive games)")


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chesskers")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("perft")
    pp.add_argument("--depth", type=int, default=5)
    pp.add_argument("--fen", default=None)
    pp.add_argument("--config", default=None)
    pp.add_argument("--seed", type=int, default=None)
    pp.set_defaults(func=cmd_perft)

    pl = sub.add_parser("play")
    pl.add_argument("--side", choices=["chess", "checkers"], default="chess")
    pl.add_argument("--vs", default="random")
    pl.add_argument("--config", default=None)
    pl.add_argument("--seed", type=int, default=None)
    pl.add_argument("--state", default=None, help="resume from a state string (see api /api/new response)")
    pl.set_defaults(func=cmd_play)

    sp = sub.add_parser("selfplay")
    sp.add_argument("--agents", required=True, help="chess_agent,checkers_agent e.g. random,random")
    sp.add_argument("--n", type=int, default=1000)
    sp.add_argument("--out", default="results.jsonl")
    sp.add_argument("--config", default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.set_defaults(func=cmd_selfplay)

    mt = sub.add_parser("match")
    mt.add_argument("--a", required=True)
    mt.add_argument("--b", required=True)
    mt.add_argument("--games", type=int, default=100)
    mt.add_argument("--swap-sides", action="store_true")
    mt.add_argument("--config", default=None)
    mt.add_argument("--seed", type=int, default=None)
    mt.set_defaults(func=cmd_match)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
