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
| `mctsnn:<iters>:<model_path>` | MCTS with neural network policy (default 300 iters) |

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
  --players mctsnn:300:release_models/azul_alpha.safetensors heuristicai
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

The `mctsnn` agent uses a residual network trained via AlphaZero-style self-play. Models are saved as `.safetensors` files.

### Quick start — automated loop

```bash
# Self-play → train → evaluate, 10 iterations (250 games each)
./util/train_loop.sh

# Custom: 20 iterations, 500 games, also run 3p and 4p self-play
./util/train_loop.sh 20 500 --multi
```

### Bootstrap — new architecture or fresh start

Use this when switching to a new NN topology so existing training data isn't wasted.

```bash
./util/bootstrap.sh
```

Clears `training_models/`, trains two fresh models on all existing `training_data/`, runs a 400-game tournament, and saves the winner as `azul_model_v1.safetensors`. Then continue with `train_loop.sh` as normal.

### Manual steps

**Step 1 — Generate training data**

```bash
# 2-player self-play, auto-loads latest model from training_models/
cargo run --release --features="native" --bin headless -- \
  --self-play --players mctsnn:300 --games 250

# 4-player self-play
cargo run --release --features="native" --bin headless -- \
  --self-play --self-play-players 4 --players mctsnn:300 --games 250
```

Data is saved to `training_data/data_<timestamp>.json`.

**Step 2 — Train**

```bash
cargo run --release --features="native" --bin train
```

The trainer:
1. Loads the last 5 data files (~90k samples, rolling replay buffer)
2. Fine-tunes (or initialises) the model for 10 epochs with AdamW (lr=1e-4)
3. Evaluates the new model vs. the previous one (400 games, 300 MCTS iters)
4. Deploys to `release_models/azul_alpha.safetensors` only if the new model achieves ≥ 52% win rate

Versioned checkpoints are saved to `training_models/azul_model_vN.safetensors`.

**Step 3 — Evaluate & repeat**

```bash
cargo run --release --features="native" --bin headless -- \
  --players mctsnn:300:release_models/azul_alpha.safetensors heuristicai
```

---

## AI Agents

| Agent | CLI Name | Notes |
|---|---|---|
| `SimpleAI` | `simpleai` | Picks a random legal move |
| `HeuristicAI` | `heuristicai` | Priority-ordered rule-based strategy, stateless |
| `GreedyAI` | `greedyai` | Maximises immediate scoring gain |
| `ConvergenceAI` | `convergenceai` | Focuses on completing wall rows and columns |
| `MctsNnAI` | `mctsnn:<iters>:<model>` | MCTS with a neural network policy + value head |

MCTS uses PUCT selection (exploration constant 1.41) with progressive widening. The NN architecture (`TacticianNet`) uses a residual trunk: stem (256) → 2× ResBlock → bottleneck (128), with separate policy and value heads.

---

## Project Structure

```
src/
  lib.rs               # Core game logic (GameState, PlayerBoard, Move) + Wasm wrapper
  config.rs            # Hardware config (mcts_batch_size)
  ai/
    simple_ai.rs
    heuristic_ai.rs
    greedy_ai.rs
    convergence_ai.rs
    mcts_lib.rs        # Generic MCTS infrastructure
    mcts_nn_ai.rs      # MCTS + NN agent
    nn/
      mod.rs           # AzulNetwork trait, encode_state, ResBlock
      tactician.rs     # TacticianNet — current active species
  bin/
    headless.rs        # CLI: parallel simulation + self-play data generation
    train.rs           # CLI: NN training + evaluation + deployment
    bootstrap.rs       # CLI: train fresh models on all data, tournament picks winner
    replay.rs          # CLI: interactive game log viewer
util/
  train_loop.sh        # Automated self-play → train → evaluate loop
  bootstrap.sh         # Bootstrap a new model architecture
index.html / index.js  # Web UI
hardware_config.json   # Runtime config (mcts_batch_size)
```
