"""Stateless FastAPI service. Spec section 11.

No database, no session store: the client holds the state string and posts it
back with every request (spec 11.2). The server never trusts client-supplied
state without validating it first.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.board import Position, Side, Result
from engine.config import RuleConfig
from engine.serialize import serialize, deserialize, move_to_str
from engine.moves import move_to_uci
from agents.random_agent import RandomAgent
from agents.greedy import GreedyAgent
from agents.alphabeta import AlphaBetaAgent
from agents.evals import chess_side_eval, checkers_side_eval

# A single active RuleConfig for the whole deployment (spec 7.1: a model/client
# trained or built under one ruleset must never silently see another).
_config_path = os.environ.get("CHESSKERS_CONFIG")
if _config_path:
    import json
    with open(_config_path) as f:
        ACTIVE_CONFIG = RuleConfig.from_json(json.load(f))
else:
    ACTIVE_CONFIG = RuleConfig()

app = FastAPI(title="I am playing chess while you are playing checkers")


class NewGameRequest(BaseModel):
    config: dict | None = None
    first_mover: str | None = None


class MovesRequest(BaseModel):
    state: str


class MoveRequest(BaseModel):
    state: str
    move: int


class AiMoveRequest(BaseModel):
    state: str
    agent: str = "random"
    difficulty: int | None = None


def _load_state(state: str) -> Position:
    try:
        return deserialize(state, ACTIVE_CONFIG)
    except (ValueError, IndexError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"invalid state: {e}")


def _legal_moves_payload(pos: Position) -> list[dict]:
    return [
        {"move": mv, "uci": move_to_uci(mv), "notation": move_to_str(pos, mv)}
        for mv in pos.legal_moves()
    ]


def _result_payload(result: Result | None) -> str | None:
    if result is None:
        return None
    return {Result.CHESS_WIN: "chess", Result.CHECKERS_WIN: "checkers", Result.DRAW: "draw"}[result]


def _make_agent(name: str, side: Side, difficulty: int | None):
    eval_fn = chess_side_eval if side == Side.CHESS else checkers_side_eval
    name = (name or "random").lower()
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    if name == "alphabeta":
        depth = difficulty if difficulty else 3
        return AlphaBetaAgent(depth=depth, eval_fn=eval_fn, time_limit_seconds=5.0)
    if name == "neural":
        from agents.neural_agent import NeuralAgent
        try:
            return NeuralAgent(side=side)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=400, detail=f"unknown agent: {name}")


@app.post("/api/new")
def new_game(req: NewGameRequest):
    if req.config is not None and req.config != ACTIVE_CONFIG.to_json():
        raise HTTPException(status_code=400, detail="server does not support per-request rule overrides")

    pos = Position(ACTIVE_CONFIG)
    if req.first_mover in ("chess", "checkers"):
        pos.side_to_move = Side.CHESS if req.first_mover == "chess" else Side.CHECKERS
        pos._key = pos._compute_key()
        pos.history = [pos._key]

    return {
        "state": serialize(pos),
        "legal_moves": _legal_moves_payload(pos),
        "result": _result_payload(pos.result()),
    }


@app.post("/api/moves")
def list_moves(req: MovesRequest):
    pos = _load_state(req.state)
    return {
        "legal_moves": _legal_moves_payload(pos),
        "result": _result_payload(pos.result()),
        "must_continue": pos.jump_from is not None,
    }


@app.post("/api/move")
def make_move(req: MoveRequest):
    pos = _load_state(req.state)
    legal = pos.legal_moves()
    if req.move not in legal:
        raise HTTPException(status_code=400, detail="illegal move for this position")
    pos.make(req.move)
    return {
        "state": serialize(pos),
        "legal_moves": _legal_moves_payload(pos),
        "result": _result_payload(pos.result()),
        "must_continue": pos.jump_from is not None,
    }


@app.post("/api/ai_move")
def ai_move(req: AiMoveRequest):
    pos = _load_state(req.state)
    if pos.result() is not None:
        raise HTTPException(status_code=400, detail="game already over")
    agent = _make_agent(req.agent, pos.side_to_move, req.difficulty)
    mv = agent.select(pos)
    pos.make(mv)
    return {
        "state": serialize(pos),
        "move": mv,
        "legal_moves": _legal_moves_payload(pos),
        "result": _result_payload(pos.result()),
        "must_continue": pos.jump_from is not None,
    }


@app.get("/api/config")
def get_config():
    return ACTIVE_CONFIG.to_json()


app.mount("/", StaticFiles(directory="static", html=True), name="static")
