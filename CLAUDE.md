# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A complete implementation of the board game Azul (2–4 players) in Rust, with two deployment targets:
- **WebAssembly** – compiled with `wasm-pack` for an in-browser interactive UI
- **Native binary** – compiled with `cargo` for headless AI simulation and neural network training

## Build Commands

### WebAssembly (browser UI)
```bash
wasm-pack build --target web
python3 -m http.server   # then open http://localhost:8000
```

### Native binaries (require `--features="native"`)
```bash
# AI vs AI simulation
cargo run --release --features="native" --bin headless -- --players heuristicai mctsheuristic:1000

# 4-player match, 500 games
cargo run --release --features="native" --bin headless -g 500 -p mctsheuristic mctsheuristic heuristicai simpleai

# Self-play data generation
cargo run --release --features="native" --bin headless -- --self-play --players mctsnn:200 --games 100

# Train neural network
cargo run --release --features="native" --bin train
```

### Standard cargo
```bash
cargo build                  # builds lib + main (Wasm-compatible subset)
cargo test                   # run tests
cargo check                  # fast type-check without compiling
```

## Feature Flags

The `native` feature gates all dependencies incompatible with Wasm (`tch`, `clap`, `rayon`, `serde_json`, `chrono`, `anyhow`, `tempfile`). Modules gated by this feature use `#[cfg(feature = "native")]`. The `MctsNnAI` and `nn` modules are compiled only under `native`.

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
| `MctsHeuristicAI` | `mcts_heuristic_ai.rs` | MCTS using heuristic evaluation |
| `MctsNnAI` | `mcts_nn_ai.rs` | MCTS using a neural network policy (native only) |
| `HumanAgent` | `human_agent.rs` | Placeholder for human input from JS |

MCTS infrastructure is shared in `mcts_lib.rs` as a generic `Mcts<P: MctsPolicy>` that works with any policy. The `MctsPolicy` trait requires an `evaluate` method returning `(value: f32, policy: HashMap<Move, f32>)`. PUCT selection with exploration constant 1.41 is used.

### Neural network – `src/ai/nn.rs`
A hand-rolled MLP using `tanh` activations. Architecture: `[INPUT_SIZE=583, 256, 256, POLICY_SIZE+1=51]`. The policy head outputs move probabilities over a flat action space; the value head outputs a scalar. Loaded/saved via `tch` (LibTorch). `NeuralNetwork::from_bytes` exists but is currently a placeholder.

### Binaries – `src/bin/`
- **`headless.rs`** – CLI using `clap`. Two modes:
  - Normal: runs parallel games with `rayon`, saves `stats/<timestamp>/` JSON
  - `--self-play`: generates `TrainingData` (state vector, MCTS policy, outcome), saves to `training_data/`
- **`train.rs`** – Loads latest `training_data/*.json`, fine-tunes or initializes the `Net` (a `tch` model with identical architecture to `NeuralNetwork`), saves versioned model to `training_models/azul_model_vN.ot` and deploys to `release_models/azul_alpha.ot`.

### Web UI – `index.html` / `index.js`
JavaScript front-end that imports the Wasm pkg. Interacts with `WasmGame` methods: `getState`, `getLegalMoves`, `applyMove`, `runAiTurn`, `handleRoundEnd`, `applyEndGameScoring`, `getWallLayout`.

### Headless CLI agent syntax
```
simpleai
heuristicai
mctsheuristic:<iterations>          # default 5000
mctsnn:<iterations>:<model_path>    # default 800 iterations, optional path
```

## Training Pipeline

1. Generate data: `headless --self-play` → `training_data/data_<timestamp>.json`
2. Train: `train` binary loads latest data + model, runs 10 epochs, saves `training_models/azul_model_vN.ot` and `release_models/azul_alpha.ot`
3. Evaluate: `headless --players mctsnn:200:release_models/azul_alpha.ot mctsheuristic:200`
4. Repeat from step 1 with improved model

The `headless --self-play` auto-discovers the latest `.ot` file in `training_models/` when no model path is specified in the agent string.

## LibTorch Dependency

The `tch` crate requires a locally installed LibTorch. Follow the [tch-rs setup instructions](https://github.com/LaurentMazare/tch-rs) before running any `--features="native"` build. The `LIBTORCH` and `LD_LIBRARY_PATH` environment variables typically need to be set.
