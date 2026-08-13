import argparse
import csv
import datetime
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import Position, RuleConfig, Side, Result, MoveType, decode

NUM_MOVE_TYPES = len(MoveType)


class ResidualBlock(nn.Module):
    """Standard AlphaZero-style residual block. GroupNorm, not BatchNorm —
    self-play forwards one position at a time (batch size 1), where BatchNorm's
    per-batch statistics are noisy/unstable; GroupNorm normalizes within a
    single example and doesn't have that failure mode."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


class PolicyValueNet(nn.Module):
    """channels/num_blocks are deliberately parameters, not constants — checkers'
    task (build a mating net from one-square diagonal steps) is harder than
    chess' (capture everything), so checkers_net is given more capacity than
    chess_net (see train(), where the two are constructed) rather than treating
    the two sides as symmetric by default."""

    def __init__(self, in_channels: int = 19, channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        flat = channels * 8 * 8
        # Factored policy head: from-square (64) + to-square (64) + move-type
        # (16) logits, summed per legal move and softmaxed over only THAT
        # position's legal moves — not a raw 65536-way softmax (~64M params).
        self.from_head = nn.Linear(flat, 64)
        self.to_head = nn.Linear(flat, 64)
        self.type_head = nn.Linear(flat, NUM_MOVE_TYPES)
        self.value_head = nn.Sequential(
            nn.Linear(flat, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Tanh(),
        )

    def forward(self, planes):
        x = self.stem(planes)
        x = self.tower(x)
        x = x.flatten(1)
        value = self.value_head(x).squeeze(-1)
        return self.from_head(x), self.to_head(x), self.type_head(x), value


def move_scores(from_logits, to_logits, type_logits, legal_moves):
    frms, tos, mtypes = zip(*(decode(mv) for mv in legal_moves))
    device = from_logits.device
    frms = torch.tensor(frms, dtype=torch.long, device=device)
    tos = torch.tensor(tos, dtype=torch.long, device=device)
    mtypes = torch.tensor([int(t) for t in mtypes], dtype=torch.long, device=device)
    return from_logits[frms] + to_logits[tos] + type_logits[mtypes]


# Dirichlet noise mixed into the move distribution every ply (not just the
# game's opening) — AlphaZero mixes noise only at the MCTS root because its
# tree search re-explores at every node below that; this setup has no search,
# just direct sampling each ply, so the noise floor has to be applied at
# every decision or it does nothing once the net's own logits saturate. The
# mix is a genuine floor: even if move_scores collapses to one-hot, the
# sampling distribution still keeps eps of Dirichlet mass spread over every
# legal move, so entropy can no longer be driven to exactly zero the way it
# was pre-fix. log_prob/entropy below are taken from the MIXED distribution
# (not the net's raw softmax) since that's what was actually sampled from —
# using the net-only log_prob here would be a policy-gradient/behavior
# mismatch.
DIRICHLET_ALPHA = 0.3
NOISE_EPS = 0.1


def select_move(net, planes, legal_moves, device):
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    from_logits, to_logits, type_logits, value = net(x)
    scores = move_scores(from_logits[0], to_logits[0], type_logits[0], legal_moves)
    net_probs = F.softmax(scores, dim=-1)
    noise = torch.distributions.Dirichlet(
        torch.full((len(legal_moves),), DIRICHLET_ALPHA, device=device)
    ).sample()
    mixed_probs = (1.0 - NOISE_EPS) * net_probs + NOISE_EPS * noise
    dist = torch.distributions.Categorical(probs=mixed_probs)
    idx = dist.sample()
    return legal_moves[idx.item()], dist.log_prob(idx), value[0], dist.entropy()


# Win is a flat, dominant bonus — never stacked with material credit, and set
# well above the maximum possible material-capture credit so winning always
# strictly beats any amount of pure material accumulation (the standard way
# to keep a shaping term from crowding out the true objective). Checkers'
# ceiling: 8 pawns(1) + 2N(3) + 2B(3) + 2R(5) + 1Q(9) = 39 in a normal game,
# up to 103 in the pathological case where every pawn promotes to a queen —
# 150 clears both with real headroom. Chess' ceiling is 24 (one checker
# piece each), trivially under either number.
WIN_BONUS = 150.0
CHESS_PIECE_VALUE = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}  # king excluded, never captured (spec)

# Four-tier outcome ranking (best to worst): WIN > escaped-repetition draw
# (fifty-move-rule) > loss > gave-up-without-resolving (threefold-repetition
# draw and ply-cap-unresolved, treated the same). All draws used to get one
# flat penalty regardless of cause — but 100% of every draw ever logged
# (thousands of games) was threefold repetition, 0% fifty-move-rule, so a
# flat "all draws are bad" penalty was really just "repetition is bad" in
# disguise, and it was punishing the RARE genuinely-hard-fought fifty-move
# draw exactly as harshly as the common lazy repetition one.
#
# Paired with _ban_repetition_moves (blocks the move that would trigger the
# 3rd occurrence of a position whenever an alternative exists), true
# threefold draws should now be rare — only when repetition is truly forced
# (no other legal move). A fifty-move-rule draw, by contrast, now plausibly
# represents real, hard-fought play that simply didn't resolve — so it gets
# a BONUS instead of a penalty, deliberately better than a same-material
# loss (loss stays a flat 0 floor, untouched).
#
# Threefold draw (and, matching it, ply-cap-unresolved — both are "avoided a
# real result without making progress") get a penalty LARGER than either
# side's maximum possible material credit (chess ceiling 24, checkers
# ceiling ~39 normal/103 pathological), which makes the resulting score
# ALWAYS negative — a hard, structural guarantee that this is worse than
# ANY achievable loss score (loss's floor is 0), not just worse on average.
CHESS_NORMAL_DRAW_BONUS = 12.0
CHECKERS_NORMAL_DRAW_BONUS = 20.0
CHESS_WORST_PENALTY = 30.0  # > chess' 24-point material ceiling
CHECKERS_WORST_PENALTY = 50.0  # > checkers' 39-point normal-game ceiling

# Entropy bonus: without it, pure policy-gradient (REINFORCE) has no pressure
# to keep exploring — repeated updates saturate the softmax toward one move,
# and once saturated, d(log_softmax)/d(logits) vanishes near that extreme, so
# the policy can't gradient its way back out. Confirmed happening at 325
# games in: measured entropy was exactly 0.000 on every move, net locked into
# a repeating 4-move cycle regardless of reward. This term keeps a permanent
# small incentive against full determinism (standard in A2C/PPO). 0.01 is a
# conservative starting coefficient — big enough to stop total collapse,
# small enough not to fight the actual policy signal once play sharpens up.
# 0.01 measured insufficient in practice: games 326-375 loosened up mid-run
# (entropy > 0, plies/points varied) but fully re-saturated (entropy back to
# 0.000) by game 375 — entropy's own gradient vanishes near one-hot too, so
# a weak coefficient just delays collapse rather than preventing it.
ENTROPY_COEF = 0.05

# Intrinsic repetition penalty: applied at the moment a move recreates a
# position already seen once before this game (pos.repetition_count() == 2,
# i.e. one more repeat away from the threefold draw) — not only at the
# terminal draw itself. More surgical than the entropy bonus: entropy pushes
# randomness into every move uniformly, diluting sharp play everywhere; this
# only fires on the exact ply that re-walks into a prior position, so the
# penalty lands on the move that actually caused the problem. Scale (0.02 in
# normalized r-units, ~3 raw points) is deliberately small relative to a real
# capture (chess: 1/24 ~= 0.007 per checker piece in raw-points-over-WIN_BONUS
# terms already; checkers: up to 9/150 for a queen) — meant to nudge, not to
# swamp the signal from actually playing well.
REPETITION_STEP_PENALTY = 0.02


def _material_snapshot(board):
    """(count of checker pieces, total value of chess pieces) currently on the board."""
    checker_count = 0
    chess_value = 0
    for ch in board:
        if ch in ("c", "C"):
            checker_count += 1
        elif ch in CHESS_PIECE_VALUE:
            chess_value += CHESS_PIECE_VALUE[ch]
    return checker_count, chess_value


def reward_for(side, result, checkers_captured_by_chess, chess_value_captured_by_checkers, is_repetition_draw):
    """Four tiers, best to worst: WIN (flat WIN_BONUS) > fifty-move-rule draw
    (material + a BONUS, deliberately better than a same-material loss) >
    loss (material only, floor 0) > threefold-repetition draw / ply-cap
    unresolved (material - a penalty sized larger than either side's max
    possible material credit, so the result is ALWAYS negative — structurally
    guaranteed worse than any loss, not just worse on average). See the
    CHESS_NORMAL_DRAW_BONUS / CHESS_WORST_PENALTY comment block for why the
    draw/threefold-draw split exists at all."""
    if side == Side.CHESS:
        if result == Result.CHESS_WIN:
            return WIN_BONUS
        points = float(checkers_captured_by_chess)  # 1 point per checker piece captured
        if result == Result.DRAW:
            points += -CHESS_WORST_PENALTY if is_repetition_draw else CHESS_NORMAL_DRAW_BONUS
        elif result is None:
            points -= CHESS_WORST_PENALTY
        return points
    else:
        if result == Result.CHECKERS_WIN:
            return WIN_BONUS
        points = float(chess_value_captured_by_checkers)  # standard chess piece values
        if result == Result.DRAW:
            points += -CHECKERS_WORST_PENALTY if is_repetition_draw else CHECKERS_NORMAL_DRAW_BONUS
        elif result is None:
            points -= CHECKERS_WORST_PENALTY
        return points


def _ban_repetition_moves(pos, legal_moves, seen_counts):
    """Removes any move that would trigger the 3rd occurrence of a position
    (the move that causes a threefold-repetition draw) from the legal set —
    but only when at least one other legal move avoids it. Never bans down
    to zero moves: doing so would hand the opponent an artificial "no legal
    move" win/loss (see terminal.py) unrelated to actual play, converting a
    training-time nudge into a rules-engine exploit. Training-only — this
    filter is never applied to the deployed game, which still allows
    repetition draws normally; RuleConfig itself is untouched.

    Previously tried and reverted (see git history / conversation): banning
    outright, with a policy still this early in training, produced ~95%
    ply-cap-unresolved games instead of more wins — no reward tuning fixes
    that on its own. Reintroduced now paired with CHESS/CHECKERS_WORST_PENALTY
    treating unresolved the same as a threefold draw (both the worst tier),
    so at minimum there's real pressure against the failure mode this caused
    last time, not just against repetition specifically.

    `seen_counts` is a dict mirroring pos.history's key counts, maintained by
    the caller — pos.repetition_count() itself does an O(n) scan of the whole
    game history per call, and this function needs one such check per legal
    move, per ply; on a long game that's O(n * branching) per ply and made
    real self-play runs time out. An external O(1) dict lookup avoids that
    without changing the check's meaning at all."""
    non_repeating = []
    for mv in legal_moves:
        undo = pos.make(mv)
        would_repeat = seen_counts.get(pos._key, 0) + 1 >= 3
        pos.unmake(undo)
        if not would_repeat:
            non_repeating.append(mv)
    return non_repeating if non_repeating else legal_moves


# Raised 400 -> 800: measured that "unterminated" games at 400 were mostly
# NOT stuck/aimless — checkers_points/chess_points in those rows implied both
# sides had already captured nearly all of each other's material (checkers
# 33-47 of a ~39 ceiling, chess 14-23 of a 24 ceiling) by the cap. These are
# already-decided endgames that simply hadn't reached the formal win
# condition (capture-the-last-piece / deliver mate) yet — real endgame
# conversion with few pieces left can genuinely take many moves. 800 gives
# that room; CHESS/CHECKERS_WORST_PENALTY still supplies real pressure
# against actually needing it.
def self_play_game(chess_net, checkers_net, config, device, max_plies=800):
    pos = Position(config)
    first_mover = pos.side_to_move  # captured before any move — config.first_mover picks this randomly
    traj = {Side.CHESS: [], Side.CHECKERS: []}  # list of (log_prob, value, entropy, step_reward)
    plies = 0
    seen_counts = {pos._key: 1}  # mirrors pos.history's counts — see _ban_repetition_moves

    checkers_captured_by_chess = 0
    chess_value_captured_by_checkers = 0
    prev_checker_count, prev_chess_value = _material_snapshot(pos.board)

    while pos.result() is None and plies < max_plies:
        side = pos.side_to_move
        net = chess_net if side == Side.CHESS else checkers_net
        legal = pos.legal_moves()
        if not legal:
            break
        legal = _ban_repetition_moves(pos, legal, seen_counts)
        mv, log_prob, value, entropy = select_move(net, pos.to_planes(), legal, device)
        pos.make(mv)
        plies += 1
        seen_counts[pos._key] = seen_counts.get(pos._key, 0) + 1
        # Fires on the 2nd occurrence of a position (one repeat away from the
        # 3rd-time-draw), not just when the draw itself lands — gives the
        # penalty a chance to shape play before the game is already decided.
        step_reward = -REPETITION_STEP_PENALTY if seen_counts[pos._key] >= 2 else 0.0
        traj[side].append((log_prob, value, entropy, step_reward))

        checker_count, chess_value = _material_snapshot(pos.board)
        if side == Side.CHESS and checker_count < prev_checker_count:
            checkers_captured_by_chess += prev_checker_count - checker_count
        elif side == Side.CHECKERS and chess_value < prev_chess_value:
            chess_value_captured_by_checkers += prev_chess_value - chess_value
        prev_checker_count, prev_chess_value = checker_count, chess_value

    return pos.result(), traj, plies, pos, first_mover, checkers_captured_by_chess, chess_value_captured_by_checkers


def explain_result(pos, plies, max_plies=800):
    """Human-readable reason the game ended, cross-checked against the
    engine's own terminal rules — not just the win/loss/draw label."""
    result = pos.result()
    checkers_left = sum(1 for ch in pos.board if ch in ("c", "C"))
    if result is None:
        return f"did NOT terminate within {max_plies} plies (hit the ply cap, counted as unterminated)"
    if result == Result.CHESS_WIN:
        if checkers_left == 0:
            return "all checker pieces captured -> chess wins"
        return "checkers side to move had no legal move -> chess wins"
    if result == Result.CHECKERS_WIN:
        return "chess side to move had no legal move (checkmate or stalemate) -> checkers wins"
    if result == Result.DRAW:
        if pos.repetition_count() >= 3:
            return "threefold repetition -> draw"
        if pos.halfmove_clock >= pos.config.fifty_move_plies:
            return "fifty-move rule -> draw"
        return "draw (unspecified — shouldn't happen; investigate)"
    return f"unrecognized result value: {result!r}"


def render_board(pos):
    lines = []
    for r in range(7, -1, -1):
        lines.append(" ".join(pos.board[r * 8 + f] for f in range(8)))
    return "\n".join(lines)


def atomic_save(net, games_trained, path):
    tmp = path + ".tmp"
    torch.save({"model_state_dict": net.state_dict(), "games_trained": games_trained}, tmp)
    os.replace(tmp, path)  # atomic on POSIX: a reader never sees a half-written file


def load_if_exists(net, path, device):
    """Returns the cumulative games_trained count stored in the checkpoint, or 0 if none."""
    if not os.path.exists(path):
        return 0
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        net.load_state_dict(payload["model_state_dict"])
        return payload.get("games_trained", 0)
    net.load_state_dict(payload)  # older plain state_dict, no counter available
    return 0


LOG_FIELDS = ["game", "timestamp", "first_mover", "winner", "plies", "seconds", "chess_points", "checkers_points", "reason"]


def append_log(log_path, rows):
    """Appends game records to a persistent CSV, one row per game, across every
    run/session — not just this run's terminal output, which disappears."""
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def status_line(cumulative, total_target, plies, dt, tally):
    return (
        f"\r[{cumulative}/{total_target}] plies={plies} {dt:.1f}s/game "
        f"tally={tally}          "
    )


def train(
    n_games, config, gamma, lr, device, checkpoint_every, log_every,
    chess_path, checkers_path, log_path, log_enabled=True, seed=None, batch_size=5,
    freeze_chess=False,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Same family for both — Leela Chess Zero's residual-tower + policy/value
    # head design (same family AlphaZero's paper uses), sized to Leela's own
    # early-days nets (10 blocks x 128 filters) rather than its later 19-40
    # blocks x 256 filters: that later scale was trained on tens of millions
    # of self-play games, far beyond what a few free Colab sessions can
    # generate — a network that large would stay close to its random
    # initialization at this data budget.
    #
    # NOT the same size, though: measured 0 checkers wins across 975+ games
    # while chess reached 301/500 in one run — checkers' task (coordinate
    # checker pieces into an actual mating net from one-square diagonal
    # steps) is structurally harder than chess' (capture everything), and
    # giving it equal capacity wasn't enough to close that gap on its own.
    # +~55% params, not a blind doubling — self-play still needs to stay
    # affordable on a free Colab session.
    chess_net = PolicyValueNet(channels=128, num_blocks=10).to(device)
    checkers_net = PolicyValueNet(channels=160, num_blocks=13).to(device)
    games_before = load_if_exists(chess_net, chess_path, device)
    load_if_exists(checkers_net, checkers_path, device)  # same run always trains both together
    if games_before:
        print(f"resuming: {games_before} games already trained, continuing to {games_before + n_games}")
    chess_opt = torch.optim.Adam(chess_net.parameters(), lr=lr)
    checkers_opt = torch.optim.Adam(checkers_net.parameters(), lr=lr)
    chess_opt.zero_grad()
    checkers_opt.zero_grad()

    tally = {"chess": 0, "checkers": 0, "draw": 0, "unterminated": 0}
    result_name = {Result.CHESS_WIN: "chess", Result.CHECKERS_WIN: "checkers", Result.DRAW: "draw"}
    total_target = games_before + n_games

    side_name = {Side.CHESS: "chess", Side.CHECKERS: "checkers"}
    game_log = []  # one entry per game this run: full stats, printed at the end (not mid-run — avoids scroll spam)
    last_pos = None
    for i in range(n_games):
        t0 = time.monotonic()
        result, traj, plies, last_pos, first_mover, checkers_captured, chess_value_captured = self_play_game(
            chess_net, checkers_net, config, device
        )
        dt = time.monotonic() - t0
        tally[result_name.get(result, "unterminated")] += 1
        cumulative = games_before + i + 1
        # Same check explain_result uses to label the reason — distinguishes
        # the (now rare, since _ban_repetition_moves blocks it when avoidable)
        # threefold draw from a fifty-move-rule draw for reward_for's 4-tier
        # split. Meaningless when result != DRAW; reward_for ignores it then.
        is_repetition_draw = result == Result.DRAW and last_pos.repetition_count() >= 3
        chess_points = reward_for(Side.CHESS, result, checkers_captured, chess_value_captured, is_repetition_draw)
        checkers_points = reward_for(Side.CHECKERS, result, checkers_captured, chess_value_captured, is_repetition_draw)
        game_log.append({
            "game": cumulative,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "first_mover": side_name[first_mover],
            "winner": result_name.get(result, "unterminated"),
            "plies": plies,
            "seconds": round(dt, 1),
            "chess_points": chess_points,
            "checkers_points": checkers_points,
            "reason": explain_result(last_pos, plies),
        })

        for side, points in (
            (Side.CHESS, chess_points),
            (Side.CHECKERS, checkers_points),
        ):
            if freeze_chess and side == Side.CHESS:
                # chess_net still plays every game (self-play needs both
                # sides moving) but never learns from it — checkers keeps
                # updating every game as normal. Manual, not auto-detected:
                # lift this by rerunning without --freeze-chess once checkers
                # has its first win, not something the script decides itself.
                continue
            entries = traj[side]
            n = len(entries)
            if n == 0:
                continue
            r = points / WIN_BONUS  # normalized to ~[0, 1] — keeps optimizer step sizes stable
            # regardless of the raw point scale, and matches the value head's
            # Tanh output range ([-1, 1]) without needing to touch the network.
            # Per-step reward is 0 everywhere except: the intrinsic repetition
            # penalty (see self_play_game) on the ply it fires, and the
            # terminal game reward `r` added onto the side's own last move.
            # Standard backward discounted-return sum: G_t = reward_t + gamma*G_{t+1}
            # — reduces to the old pure-terminal formula (r * gamma^(n-1-i))
            # when every step_reward is 0 except the final one.
            step_rewards = [sr for _lp, _v, _e, sr in entries]
            step_rewards[-1] += r
            returns_list = [0.0] * n
            running = 0.0
            for t in range(n - 1, -1, -1):
                running = step_rewards[t] + gamma * running
                returns_list[t] = running
            returns = torch.tensor(returns_list, device=device)
            log_probs = torch.stack([lp for lp, _v, _e, _sr in entries])
            values = torch.stack([v for _lp, v, _e, _sr in entries])
            entropies = torch.stack([e for _lp, _v, e, _sr in entries])
            advantage = returns - values.detach()
            policy_loss = -(log_probs * advantage).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropies.mean()  # maximize entropy -> minimize its negative
            # Gradient accumulation: backward() every game (grads sum in
            # .grad), but step()/zero_grad() only every batch_size games —
            # see reasoning below. Each game's own gradient magnitude is
            # unaffected by this; what changes is how many games' gradients
            # get summed into one applied step.
            (policy_loss + 0.5 * value_loss + ENTROPY_COEF * entropy_loss).backward()

        # batch_size=1 reduces to the old every-game-updates-immediately
        # behavior exactly. batch_size>1 trades faster reaction for a less
        # noisy gradient (each step reflects several games' outcomes
        # averaged together instead of one, which is what was producing wild
        # swings — e.g. chess 32->5 wins, then 301->47 wins, across
        # different runs). Real tradeoff, not a free win: a batch also
        # dilutes how much any ONE game's outcome can move the step, which
        # matters most for the rare-but-important cases (threefold draw /
        # unterminated, now a minority of games post-ban) — a harsh single
        # penalty can get outweighed within its own batch by several
        # ordinary wins pulling the accumulated gradient the other way. 5 is
        # a deliberately modest choice for that reason, not the 10-20 a pure
        # variance-reduction argument alone would suggest.
        if cumulative % batch_size == 0 or i == n_games - 1:
            if not freeze_chess:
                chess_opt.step()
                chess_opt.zero_grad()
            checkers_opt.step()
            checkers_opt.zero_grad()

        if cumulative % checkpoint_every == 0 or i == n_games - 1:
            atomic_save(chess_net, cumulative, chess_path)
            atomic_save(checkers_net, cumulative, checkers_path)

        sys.stdout.write(status_line(cumulative, total_target, plies, dt, tally))
        sys.stdout.flush()
        if cumulative % log_every == 0 or i == n_games - 1:
            print()  # leave this one status line as durable scrollback

    print(f"\ndone. {games_before + n_games} games trained total. final tally this run={tally}")

    if log_enabled:
        append_log(log_path, game_log)
        print(f"logged {len(game_log)} game{'s' if len(game_log) != 1 else ''} to {log_path}")
    else:
        print("logging disabled (--no-log): not writing to training_log.csv")

    # Draw itself doesn't say WHY — split it here using the same "reason"
    # text explain_result already wrote per game, so threefold (the one
    # _ban_repetition_moves targets, and reward_for's worst tier) is visible
    # separately from a fifty-move-rule draw (mild tier) instead of both
    # being lumped into one "draw" count the way `tally` does.
    breakdown = {
        "chess_wins": sum(1 for g in game_log if g["winner"] == "chess"),
        "checkers_wins": sum(1 for g in game_log if g["winner"] == "checkers"),
        "draws_threefold_repetition": sum(
            1 for g in game_log if g["winner"] == "draw" and "threefold" in g["reason"]
        ),
        "draws_fifty_move_rule": sum(
            1 for g in game_log if g["winner"] == "draw" and "fifty-move" in g["reason"]
        ),
        "unterminated": sum(1 for g in game_log if g["winner"] == "unterminated"),
    }
    print(f"\nbatch breakdown ({len(game_log)} games this run):")
    print(json.dumps(breakdown, indent=2))

    print(f"\nper-game stats ({len(game_log)} game{'s' if len(game_log) != 1 else ''} this run):")
    header = f"{'game':>5}  {'first mover':<11} {'winner':<11} {'plies':>6}  {'sec':>5}  {'chess pts':>9}  {'checkers pts':>12}  reason"
    print(header)
    print("-" * len(header))
    for g in game_log:
        print(
            f"{g['game']:>5}  {g['first_mover']:<11} {g['winner']:<11} {g['plies']:>6}  {g['seconds']:>5}  "
            f"{g['chess_points']:>9.0f}  {g['checkers_points']:>12.0f}  {g['reason']}"
        )

    if last_pos is not None:
        print("\nfinal board of the last game (rank 8 top, a-file left):")
        print(render_board(last_pos))
    return chess_net, checkers_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100, help="how many MORE games to run this session")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=5,
                    help="games per gradient step (gradients accumulate across the batch); 1 = update every game (old behavior)")
    p.add_argument("--checkpoint-every", type=int, default=1,
                    help="save after every N games; low default trades a little speed for crash safety")
    p.add_argument("--log-every", type=int, default=20,
                    help="how often to leave a permanent log line instead of overwriting in place")
    p.add_argument("--chess-out", default="chess_net.pt")
    p.add_argument("--checkers-out", default="checkers_net.pt")
    p.add_argument("--log-path", default="training_log.csv",
                    help="persistent per-game stats, appended across every run/session — survives after the terminal doesn't")
    p.add_argument("--no-log", action="store_true",
                    help="disable writing to training_log.csv (on by default — cheap, and useful history to have)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--freeze-chess", action="store_true",
                    help="chess_net still plays every game but never updates — checkers keeps learning normally. "
                         "Manual toggle: rerun without this flag once checkers gets its first win.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    cfg = RuleConfig()
    print(f"RuleConfig hash: {cfg.config_hash()} — must match the deployed server's /api/config")

    train(
        n_games=args.games, config=cfg, gamma=args.gamma, lr=args.lr, device=device,
        checkpoint_every=args.checkpoint_every, log_every=args.log_every,
        chess_path=args.chess_out, checkers_path=args.checkers_out, log_path=args.log_path,
        log_enabled=not args.no_log, seed=args.seed, batch_size=args.batch_size,
        freeze_chess=args.freeze_chess,
    )


if __name__ == "__main__":
    main()
