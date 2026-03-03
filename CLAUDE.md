# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A complete implementation of the board game Azul (2–4 players) in Rust, with two deployment targets:
- **WebAssembly** – compiled with `wasm-pack` for an in-browser interactive UI
- **Native binary** – compiled with `cargo` for headless AI simulation, neural network training, and game replay

## Build Commands

### WebAssembly (browser UI)
```bash
wasm-pack build --target web
python3 -m http.server   # then open http://localhost:8000
```

### Native binaries (require `--features="native"`)
```bash
# AI vs AI simulation
cargo run --release --features="native" --bin headless -- --players heuristicai greedyai

# 4-player match, 500 games
cargo run --release --features="native" --bin headless -g 500 -p heuristicai greedyai convergenceai simpleai

# Self-play data generation
cargo run --release --features="native" --bin headless -- --self-play --players mctsnn:200 --games 100

# Train neural network
cargo run --release --features="native" --bin train

# Interactive game replay
cargo run --release --features="native" --bin replay -- stats/<timestamp>/game_logs.json.zst
```

### Standard cargo
```bash
cargo build                  # builds lib + main (Wasm-compatible subset)
cargo test                   # run tests
cargo check                  # fast type-check without compiling
```

## Feature Flags

The `native` feature gates all dependencies incompatible with Wasm (`candle-core`, `candle-nn`, `clap`, `rayon`, `serde_json`, `chrono`, `anyhow`, `zstd`). Modules gated by this feature use `#[cfg(feature = "native")]`. The `MctsNnAI` and `nn` modules are compiled only under `native`.

## Architecture

### Core game logic – `src/lib.rs`
All game rules live here as pure Rust structs. Key types:
- `GameState` – full game state (factories, center, players, bag, discard)
- `PlayerBoard` – pattern lines, wall, floor line, score
- `Move` / `MoveSource` / `MoveDestination` – represent a legal action
- `Tile` – enum for the 5 tile colors
- `WALL_LAYOUT` – the fixed 5×5 color layout shared by all players

Key methods on `GameState`: `new`, `get_legal_moves`, `apply_move`, `is_round_over`, `run_tiling_phase`, `refill_factories`, `apply_end_game_scoring`.

The `WasmGame` wrapper (also in `lib.rs`) exposes these methods to JavaScript via `#[wasm_bindgen]`.

### AI agents – `src/ai/`
All agents implement the `AIAgent` trait (`get_move`, `as_any`).

| Agent | File | Notes |
|---|---|---|
| `SimpleAI` | `simple_ai.rs` | Random legal move |
| `HeuristicAI` | `heuristic_ai.rs` | Prioritized rule-based heuristics, stateless |
| `GreedyAI` | `greedy_ai.rs` | Maximises immediate score gain each turn |
| `ConvergenceAI` | `convergence_ai.rs` | Focuses on completing wall rows/columns |
| `MctsNnAI` | `mcts_nn_ai.rs` | MCTS using a neural network policy (native only) |
| `HumanAgent` | `human_agent.rs` | Placeholder for human input from JS |

MCTS infrastructure is in `mcts_lib.rs` as a generic `Mcts<P: MctsPolicy>`. The `MctsPolicy` trait requires an `evaluate` method returning `(value: f32, policy: HashMap<Move, f32>)`. PUCT selection with exploration constant 1.41 is used.

### Neural network – `src/ai/nn.rs`
MLP using ReLU activations, implemented with `candle` (pure-Rust, no C++ dependency). Architecture: `[INPUT_SIZE=583, 256, 256]` with a policy head (`POLICY_SIZE` outputs) and value head (1 output). Models saved/loaded as `.safetensors` files.

### Binaries – `src/bin/`
- **`headless.rs`** – CLI using `clap`. Two modes:
  - Normal: runs parallel games with `rayon`, saves `stats/<timestamp>/summary_stats.json` and `stats/<timestamp>/game_logs.json.zst` (zstd-compressed)
  - `--self-play`: generates `TrainingData` (state vector, MCTS policy, outcome), saves to `training_data/`
- **`train.rs`** – Loads last 5 `training_data/*.json` files, fine-tunes or initializes the model with AdamW, evaluates new vs old (300 games, ≥55% win rate required), saves versioned model to `training_models/azul_model_vN.safetensors` and deploys to `release_models/azul_alpha.safetensors`.
- **`replay.rs`** – Interactive terminal viewer for `game_logs.json.zst` files. Navigate with `n`/`p`/`g<N>`/`q`.

### Web UI – `index.html` / `index.js`
JavaScript front-end that imports the Wasm pkg. Interacts with `WasmGame` methods: `getState`, `getLegalMoves`, `applyMove`, `runAiTurn`, `handleRoundEnd`, `applyEndGameScoring`, `getWallLayout`.

WasmGame agent type mapping: `0`=Human, `1`=SimpleAI, `2`=HeuristicAI, `3`=ConvergenceAI, `4`=MctsNnAI (falls back to SimpleAI in Wasm), `5`=GreedyAI, `6`=ConvergenceAI.

### Headless CLI agent syntax
```
simpleai
heuristicai
greedyai
convergenceai
mctsnn:<iterations>:<model_path>    # default 800 iterations, optional path
```

## Training Pipeline

1. Generate data: `headless --self-play` → `training_data/data_<timestamp>.json`
2. Train: `train` binary loads last 5 data files, runs 10 epochs, saves `training_models/azul_model_vN.safetensors`; deploys to `release_models/azul_alpha.safetensors` only if new model wins ≥55% vs previous
3. Evaluate: `headless --players mctsnn:200:release_models/azul_alpha.safetensors heuristicai`
4. Repeat from step 1 with improved model

The `headless --self-play` auto-discovers the latest `.safetensors` file in `training_models/` when no model path is specified in the agent string.
