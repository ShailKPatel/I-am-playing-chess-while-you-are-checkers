"use strict";

const CHESS_GLYPH = { P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔" };

const game = {
  state: null,
  legalMoves: [],
  result: null,
  mustContinue: false,
  selected: null,
  mode: "hvh",       // "hvh" | "hvb"
  humanSide: "chess", // side the human plays in hvb mode
  difficulty: 3,
  moveList: [],
  highlightMovable: true,
  firstMover: null, // "chess" | "checkers" — this game's actual first mover, defines White
  firstMoverChoice: "random", // "random" | "chess" | "checkers" — user's setting for the *next* new game
};

function squareName(idx) {
  const file = idx % 8, rank = Math.floor(idx / 8);
  return "abcdefgh"[file] + (rank + 1);
}

function decodeBoard(boardStr) {
  const board = new Array(64).fill(".");
  const rows = boardStr.split("|"); // rows[0] = rank 8 ... rows[7] = rank 1
  for (let i = 0; i < 8; i++) {
    const rank = 7 - i;
    let file = 0;
    for (const ch of rows[i]) {
      if (ch >= "0" && ch <= "9") {
        file += parseInt(ch, 10);
      } else {
        board[rank * 8 + file] = ch;
        file += 1;
      }
    }
  }
  return board;
}

function sideToMove(state) {
  return state.split("/")[1] === "w" ? "chess" : "checkers";
}

async function api(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `${path} failed: ${resp.status}`);
  }
  return resp.json();
}

function applyResponse(resp) {
  game.state = resp.state;
  game.legalMoves = resp.legal_moves;
  game.result = resp.result;
  game.mustContinue = !!resp.must_continue;
  game.selected = null;
  render();
  maybePlayBot();
}

async function newGame() {
  const body = {};
  if (game.firstMoverChoice === "chess" || game.firstMoverChoice === "checkers") {
    body.first_mover = game.firstMoverChoice;
  }
  const resp = await api("/api/new", body);
  game.moveList = [];
  game.firstMover = sideToMove(resp.state);
  applyResponse(resp);
}

async function loadState(stateStr) {
  // Console helper: jump straight to a given state string, e.g.
  // loadState("cccccccc|...|.../w/KQ/-/0/1/-/8e68f26e85")
  const resp = await api("/api/moves", { state: stateStr });
  game.moveList = [];
  game.firstMover = sideToMove(stateStr); // best-effort: unknown true history
  applyResponse({ state: stateStr, ...resp });
}
window.loadState = loadState;

async function playMove(mv, notation) {
  const resp = await api("/api/move", { state: game.state, move: mv });
  game.moveList.push(notation);
  applyResponse(resp);
}

async function maybePlayBot() {
  if (game.mode !== "hvb" || game.result) return;
  const turn = sideToMove(game.state);
  const botSide = game.humanSide === "chess" ? "checkers" : "chess";
  if (turn !== botSide) return;
  setStatus("bot is thinking...");
  const preMoveLegal = game.legalMoves;
  const resp = await api("/api/ai_move", {
    state: game.state,
    agent: "alphabeta",
    difficulty: game.difficulty,
  });
  const played = preMoveLegal.find((m) => m.move === resp.move);
  game.moveList.push(played ? played.notation : String(resp.move));
  applyResponse(resp);
}

function setStatus(extra) {
  const el = document.getElementById("status");
  if (!game.state) { el.textContent = "Loading..."; return; }
  if (game.result) {
    if (game.result === "draw") {
      el.innerHTML = `<span class="over">Game over: draw</span>`;
    } else {
      const color = colorOf(game.result);
      const label = color ? `${game.result} (${color})` : game.result;
      el.innerHTML = `<span class="over">Game over: ${label} wins</span>`;
    }
    return;
  }
  const turn = sideToMove(game.state);
  const army = turn === "chess" ? "Chess" : "Checkers";
  const color = colorOf(turn);
  let text = color ? `${army} (${color}) to move` : `${army} to move`;
  if (game.mustContinue) text += " (jump chain in progress)";
  if (extra) text += ` — ${extra}`;
  el.textContent = text;
}

function isFlipped() {
  if (!game.state) return false;
  if (game.mode === "hvb") {
    // Fixed to the human's own side for the whole game — nobody else is
    // looking at this screen, so there's nothing to rotate for.
    return game.humanSide === "checkers";
  }
  // Human vs human: White (whoever moved first, decided once at game start)
  // always sits at the bottom, fixed for the whole game.
  return game.firstMover === "checkers";
}

function colorOf(army) {
  if (!game.firstMover) return "";
  return army === game.firstMover ? "White" : "Black";
}

function render() {
  const boardEl = document.getElementById("board");
  boardEl.innerHTML = "";
  if (!game.state) return;
  const board = decodeBoard(game.state.split("/")[0]);
  const flipped = isFlipped();

  const destSquares = new Set();
  if (game.selected !== null) {
    for (const m of game.legalMoves) {
      if (m.uci.slice(0, 2) === squareName(game.selected)) {
        destSquares.add(m.uci.slice(2, 4));
      }
    }
  }

  const movableSquares = new Set(game.legalMoves.map((m) => m.uci.slice(0, 2)));

  for (let row = 0; row < 8; row++) {
    const rank = flipped ? row : 7 - row;
    for (let col = 0; col < 8; col++) {
      const file = flipped ? 7 - col : col;
      const idx = rank * 8 + file;
      const div = document.createElement("div");
      const dark = (file + rank) % 2 === 0;
      div.className = "sq " + (dark ? "dark" : "light");
      div.dataset.sq = idx;
      if (game.selected === idx) div.classList.add("selected");
      if (destSquares.has(squareName(idx))) {
        div.classList.add("dest");
        if (board[idx] !== ".") div.classList.add("has-piece");
      }
      if (game.highlightMovable && !game.result && movableSquares.has(squareName(idx))) {
        div.classList.add("movable");
      }

      const piece = board[idx];
      if (piece !== ".") {
        if (piece === "c" || piece === "C") {
          const c = document.createElement("div");
          c.className = "checker" + (piece === "C" ? " king" : "");
          div.appendChild(c);
        } else {
          const span = document.createElement("span");
          span.className = "chess-piece";
          span.textContent = CHESS_GLYPH[piece] || piece;
          div.appendChild(span);
        }
      }
      div.addEventListener("click", () => onSquareClick(idx));
      boardEl.appendChild(div);
    }
  }

  setStatus();
  const ml = document.getElementById("move-list");
  ml.innerHTML = game.moveList.map((m, i) => `${i + 1}. ${m}`).join("<br>");
}

async function onSquareClick(idx) {
  if (game.result) return;
  if (game.mode === "hvb") {
    const turn = sideToMove(game.state);
    if (turn !== game.humanSide) return;
  }

  const name = squareName(idx);
  if (game.selected !== null) {
    const match = game.legalMoves.find(
      (m) => m.uci.slice(0, 2) === squareName(game.selected) && m.uci.slice(2, 4) === name
    );
    if (match) {
      await playMove(match.move, match.notation);
      return;
    }
  }

  if (game.mustContinue) {
    const only = game.legalMoves[0];
    if (only && only.uci.slice(0, 2) === name) {
      game.selected = idx;
      render();
    }
    return;
  }

  const hasOwnMoves = game.legalMoves.some((m) => m.uci.slice(0, 2) === name);
  game.selected = hasOwnMoves ? idx : null;
  render();
}

function setupControls() {
  document.getElementById("mode-select").addEventListener("change", (e) => {
    game.mode = e.target.value;
    document.getElementById("side-row").style.display = game.mode === "hvb" ? "flex" : "none";
    document.getElementById("difficulty-row").style.display = game.mode === "hvb" ? "flex" : "none";
  });
  document.getElementById("side-select").addEventListener("change", (e) => {
    game.humanSide = e.target.value;
  });
  document.getElementById("difficulty-select").addEventListener("change", (e) => {
    game.difficulty = parseInt(e.target.value, 10);
  });
  document.getElementById("first-mover-select").addEventListener("change", (e) => {
    game.firstMoverChoice = e.target.value;
  });
  document.getElementById("new-game-btn").addEventListener("click", () => newGame());
  document.getElementById("highlight-movable-toggle").addEventListener("change", (e) => {
    game.highlightMovable = e.target.checked;
    render();
  });
}

window.addEventListener("DOMContentLoaded", () => {
  setupControls();
  newGame();
});
