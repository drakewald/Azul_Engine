# Azul Engine

A complete implementation of the board game [Azul](https://boardgamegeek.com/boardgame/230802/azul) for 2–4 players, written in Rust. The engine compiles to WebAssembly for in-browser play and also ships as a suite of native CLI tools for AI simulation, training, and game replay.

## Features

- **Complete 2–4 player ruleset** — all phases: drafting, pattern lines, wall tiling, scoring, end-game bonuses
- **Six AI agents** — from random to neural-network MCTS (see [AI Agents](#ai-agents))
- **Headless simulation** — parallel AI vs. AI matches with compressed, replayable game logs
- **Interactive replay viewer** — step through any recorded game turn-by-turn in the terminal
- **AlphaZero-style training pipeline** — self-play data generation, candle (pure-Rust) neural network training, and automated model evaluation with a promotion gate
- **Web UI** — play against any AI in the browser via a compiled Wasm module

---

## Prerequisites

- **Rust toolchain** — [install via rustup](https://rustup.rs)
- **wasm-pack** — only needed for the web build

```bash
cargo install wasm-pack
```

> The neural network training uses [candle](https://github.com/huggingface/candle), a pure-Rust ML framework. No external C++ libraries (e.g., LibTorch) are required.

---

## Running the Web UI

```bash
# 1. Compile to WebAssembly
wasm-pack build --target web

# 2. Serve locally
python3 -m http.server
```

Open `http://localhost:8000` in your browser.

---

## Headless Simulation

Run AI vs. AI matches and collect statistics. Results are saved to `stats/<timestamp>/`.

```bash
cargo run --release --features="native" --bin headless -- --players <agents...> [--games N]
```

**Agent strings:**

| String | Description |
|---|---|
| `simpleai` | Random legal move |
| `heuristicai` | Rule-based heuristics |
| `greedyai` | Greedy immediate-score maximiser |
| `convergenceai` | Convergence-focused heuristic |
| `mctsnn:<iters>:<model_path>` | MCTS with neural network policy (default 800 iters) |

**Examples:**

```bash
# 2-player match, 100 games
cargo run --release --features="native" --bin headless -- \
  --players heuristicai greedyai

# 4-player match, 500 games
cargo run --release --features="native" --bin headless -- \
  --players heuristicai greedyai convergenceai simpleai --games 500

# Evaluate a trained model head-to-head
cargo run --release --features="native" --bin headless -- \
  --players mctsnn:200:release_models/azul_alpha.safetensors heuristicai
```

**Output files** in `stats/<timestamp>/`:

| File | Contents |
|---|---|
| `summary_stats.json` | Win rates, ties, total time |
| `game_logs.json.zst` | Full move-by-move game logs (zstd-compressed) |

---

## Replay Viewer

Step through any recorded game interactively in the terminal.

```bash
cargo run --release --features="native" --bin replay -- <path/to/game_logs.json.zst> [--game N]
```

**Options:**

| Flag | Description |
|---|---|
| `--game N` / `-g N` | Start at game N (1-based, default 1) |

**Navigation:**

| Key | Action |
|---|---|
| `Enter` or `n` | Next step |
| `p` | Previous step |
| `g<N>` | Jump to game N (e.g. `g3`) |
| `q` | Quit |

The display shows factories, center, and all player boards with ANSI-colored tiles. Empty wall slots show their expected tile color dimmed. The current player is marked with `◄`.

Factory state is reconstructed exactly at each round boundary using recorded layout data, so the replay is fully accurate across all rounds.

---

## Neural Network Training Pipeline

The `mctsnn` agent uses an MLP trained via AlphaZero-style self-play. Models are saved as `.safetensors` files.

### Step 1 — Generate Training Data

```bash
# 2-player self-play, auto-loads latest model from training_models/
cargo run --release --features="native" --bin headless -- \
  --self-play --players mctsnn:200 --games 100

# 4-player self-play
cargo run --release --features="native" --bin headless -- \
  --self-play --self-play-players 4 --players mctsnn:200 --games 50
```

Data is saved to `training_data/data_<timestamp>.json`.

### Step 2 — Train

```bash
cargo run --release --features="native" --bin train
```

The trainer:
1. Loads the last 5 data files (rolling replay buffer)
2. Fine-tunes (or initialises) the model for 10 epochs with AdamW
3. Evaluates the new model vs. the previous one (300 games, 200 MCTS iters)
4. Deploys to `release_models/azul_alpha.safetensors` only if the new model achieves ≥ 55% adjusted win rate

Versioned checkpoints are saved to `training_models/azul_model_vN.safetensors`.

### Step 3 — Evaluate & Repeat

```bash
cargo run --release --features="native" --bin headless -- \
  --players mctsnn:200:release_models/azul_alpha.safetensors heuristicai
```

Repeat from Step 1 to continue improving the model.

---

## AI Agents

| Agent | CLI Name | Notes |
|---|---|---|
| `SimpleAI` | `simpleai` | Picks a random legal move |
| `HeuristicAI` | `heuristicai` | Priority-ordered rule-based strategy, stateless |
| `GreedyAI` | `greedyai` | Maximises immediate scoring gain |
| `ConvergenceAI` | `convergenceai` | Focuses on completing wall rows and columns |
| `MctsNnAI` | `mctsnn:<iters>:<model>` | MCTS with a neural network policy + value head |

MCTS agents use PUCT selection (exploration constant 1.41). The NN architecture is a two-hidden-layer MLP (`583 → 256 → 256 → policy + value`).

---

## Project Structure

```
src/
  lib.rs               # Core game logic (GameState, PlayerBoard, Move) + Wasm wrapper
  ai/
    simple_ai.rs
    heuristic_ai.rs
    greedy_ai.rs
    convergence_ai.rs
    mcts_nn_ai.rs      # native only
    mcts_lib.rs        # Generic MCTS infrastructure
    nn.rs              # Neural network definition (candle)
  bin/
    headless.rs        # CLI: parallel simulation + self-play data generation
    train.rs           # CLI: NN training + evaluation + deployment
    replay.rs          # CLI: interactive game log viewer
index.html / index.js  # Web UI
```
