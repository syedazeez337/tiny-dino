const boardEl = document.getElementById("board");
const bannerEl = document.getElementById("banner");
const statusEl = document.getElementById("status");
const mmoveEl = document.getElementById("mmove");
const latencyEl = document.getElementById("latency");
const probsEl = document.getElementById("probs");
const humanSel = document.getElementById("human");
const newBtn = document.getElementById("new");
const cardEl = document.getElementById("card");

const state = {
  runId: null,
  board: Array(9).fill(0),
  turn: null,
  human: "x",
  status: "idle",
  model_move: null,
  probabilities: null,
  latency_ms: null,
};

const MARK = { 1: "X", 2: "O" };
const sideOf = (n) => (n === 1 ? "x" : n === 2 ? "o" : null);

function setState(json) {
  state.runId = json.run_id;
  state.board = json.board;
  state.turn = json.turn;
  state.human = json.human;
  state.status = json.status;
  state.model_move = json.model_move;
  state.probabilities = json.probabilities;
  state.latency_ms = json.latency_ms;
}

function render() {
  const myTurn = state.status === "playing" && state.turn === state.human;
  boardEl.innerHTML = "";
  state.board.forEach((value, index) => {
    const btn = document.createElement("button");
    btn.className = "cell";
    if (value) {
      btn.classList.add("taken", sideOf(value));
      btn.textContent = MARK[value];
    } else {
      btn.textContent = String(index + 1);
    }
    if (index === state.model_move) btn.classList.add("last-model");
    btn.disabled = Boolean(value) || !myTurn;
    btn.addEventListener("click", () => play(index));
    boardEl.appendChild(btn);
  });

  const banners = {
    idle: "Choose a side and start a new game.",
    playing: myTurn ? "Your move." : "Model is thinking…",
    human_win: "You win! (unexpected — the model is certified never to lose)",
    model_win: "The model wins.",
    draw: "Draw — the model never loses.",
  };
  bannerEl.textContent = banners[state.status] ?? "";
  bannerEl.className =
    "banner" +
    (state.status === "human_win" ? " win" : state.status === "model_win" ? " lose" : "");

  statusEl.textContent = `you are ${state.human.toUpperCase()} · ${state.status}`;
  mmoveEl.textContent = state.model_move == null ? "–" : String(state.model_move + 1);
  latencyEl.textContent =
    state.latency_ms == null ? "–" : `${state.latency_ms.toFixed(2)} ms`;

  renderProbs();
}

function renderProbs() {
  probsEl.innerHTML = "";
  for (let index = 0; index < 9; index++) {
    const wrap = document.createElement("div");
    wrap.className = "prob";
    const bar = document.createElement("div");
    bar.className = "bar";
    const value = state.probabilities ? state.probabilities[index] : 0;
    bar.style.height = `${Math.max(2, Math.round(value * 60))}px`;
    const label = document.createElement("span");
    label.textContent = String(index + 1);
    wrap.append(bar, label);
    probsEl.appendChild(wrap);
  }
}

async function newGame() {
  const res = await fetch("/api/play/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ human: humanSel.value }),
  });
  setState(await res.json());
  render();
}

async function play(cell) {
  if (!state.runId) return;
  const res = await fetch("/api/play/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: state.runId, cell }),
  });
  if (!res.ok) return;
  setState(await res.json());
  render();
}

async function loadModel() {
  try {
    const model = await (await fetch("/api/model")).json();
    const cert = model.certification;
    const kb = (model.bytes / 1024).toFixed(1);
    const arch = model.architecture;
    cardEl.innerHTML = `
      <b>${model.name}</b> &middot; MLP ${arch.hidden}×${arch.layers}<br>
      ${model.params.toLocaleString()} parameters &middot; ${kb} KB &middot; FP32<br>
      sha256 <code>${model.checkpoint_sha256.slice(0, 16)}…</code><br>
      ${
        cert
          ? `${cert.perfect_play ? '<span class="ok">CERTIFIED perfect play</span>' : "not certified"} &middot; ${cert.optimal}/${cert.decisions} optimal moves over ${cert.positions.toLocaleString()} positions`
          : ""
      }`;
  } catch (error) {
    cardEl.textContent = "model unavailable";
  }
}

newBtn.addEventListener("click", newGame);
humanSel.addEventListener("change", newGame);

loadModel().then(newGame);
