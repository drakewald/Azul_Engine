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
cargo run --release --features="native" --bin headless -- --self-play --players mctsnn:600 --games 250

# Train neural network
cargo run --release --features="native" --bin train

# Bootstrap new architecture from existing data
cargo run --release --features="native" --bin bootstrap

# Interactive game replay
cargo run --release --features="native" --bin replay -- stats/<timestamp>/game_logs.json.zst
```

### Training scripts
```bash
# Full self-play → train → evaluate loop (default: 10 loops, 250 games, 2p)
./util/train_loop.sh
./util/train_loop.sh 20 500          # custom loops and games
./util/train_loop.sh --multi         # also runs 3p and 4p self-play each iteration

# Bootstrap: train two fresh models on all existing data, tournament picks winner
./util/bootstrap.sh
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

MCTS infrastructure is in `mcts_lib.rs` as a generic `Mcts<P: MctsPolicy>`. The `MctsPolicy` trait requires an `evaluate` method returning `(value: f32, Vec<(Move, f32)>)`. PUCT selection with exploration constant 1.41 is used. Batch size is runtime-configurable via `hardware_config.json`.

### Neural network – `src/ai/nn/`
Expandable module where each network architecture ("species") lives in its own file and implements the shared `AzulNetwork` trait.

```
src/ai/nn/
  mod.rs        ← AzulNetwork trait, INPUT_SIZE, POLICY_SIZE, ResBlock
  tactician.rs  ← TacticianConfig, TacticianNet (current species)
```

**`AzulNetwork` trait** (inference only, used for dyn dispatch in MCTS):
```rust
fn forward(&self, input: &[f32]) -> (Vec<f32>, f32);
fn forward_batch(&self, inputs: &[Vec<f32>]) -> Vec<(Vec<f32>, f32)>;
```

**`TacticianNet`** — current species. Topology defined by `TacticianConfig::default()`:
- Input → stem (256) → LayerNorm → ReLU → 2× ResBlock → bottleneck (128) → ReLU
- Policy head: 128 → POLICY_SIZE (logits)
- Value head: 128 → 64 → ReLU → 1 → tanh → scalar

`TacticianConfig` fields: `input_dim`, `trunk_width` (256), `num_res_blocks` (2), `mid_dim` (128), `value_hidden_dim` (64), `policy_dim`.

`TacticianNet` provides `forward_train`, `trainable_vars`, and `save` for the training pipeline. Models saved/loaded as `.safetensors` files.

To add a new species: create `src/ai/nn/<name>.rs`, implement `AzulNetwork`, re-export from `mod.rs`.

### Hardware config – `src/config.rs`
Runtime settings in `hardware_config.json` (falls back to defaults if missing):
```json
{ "mcts_batch_size": 1 }
```
- `mcts_batch_size`: MCTS leaf evaluations per batch. Use 1 on CPU (batching adds overhead). Increase for GPU.

NN topology is **not** in the config file — it is defined in each species file (e.g. `TacticianConfig::default()`).

### Binaries – `src/bin/`
- **`headless.rs`** – CLI using `clap`. Two modes:
  - Normal: runs parallel games with `rayon`, saves `stats/<timestamp>/summary_stats.json` and `stats/<timestamp>/game_logs.json.zst` (zstd-compressed)
  - `--self-play`: generates `TrainingData` (state vector, MCTS policy, outcome), saves to `training_data/`
- **`train.rs`** – Loads last 5 `training_data/*.json` files (~90k samples), fine-tunes or initializes the model with AdamW, evaluates new vs old (400 games, ≥52% win rate required), saves versioned model to `training_models/azul_model_vN.safetensors` and deploys to `release_models/azul_alpha.safetensors`.
- **`bootstrap.rs`** – Trains two independent models from random weights on **all** existing training data, runs a 400-game tournament, saves the winner as the next versioned model. Use when switching to a new NN topology so existing data isn't wasted.
- **`replay.rs`** – Interactive terminal viewer for `game_logs.json.zst` files. Navigate with `n`/`p`/`g<N>`/`q`.
- **`bench_nn.rs`** – Benchmarks single and batched forward passes across res-block counts.

### Web UI – `index.html` / `index.js`
JavaScript front-end that imports the Wasm pkg. Interacts with `WasmGame` methods: `getState`, `getLegalMoves`, `applyMove`, `runAiTurn`, `handleRoundEnd`, `applyEndGameScoring`, `getWallLayout`.

WasmGame agent type mapping: `0`=Human, `1`=SimpleAI, `2`=HeuristicAI, `3`=ConvergenceAI, `4`=MctsNnAI (falls back to SimpleAI in Wasm), `5`=GreedyAI, `6`=ConvergenceAI.

### Headless CLI agent syntax
```
simpleai
heuristicai
greedyai
convergenceai
mctsnn:<iterations>:<model_path>    # default 600 iterations, optional path
```

## Training Pipeline

### Normal loop (ongoing improvement)
```bash
./util/train_loop.sh          # 10 iterations, 250 games, 2p, 300 MCTS iters
./util/train_loop.sh --multi  # also runs 3p/4p self-play each iteration
```
Each iteration: self-play → train (last 5 files, 10 epochs, AdamW lr=1e-4) → deploy if ≥52% win rate.

### Bootstrap (new topology or fresh start)
Use when the existing model weights are incompatible with a topology change:
```bash
./util/bootstrap.sh
```
Clears `training_models/`, trains two fresh models on **all** existing `training_data/`, tournament picks the winner as `azul_model_v1.safetensors`. Then run `./util/train_loop.sh` as normal.

### Manual steps
1. Generate data: `headless --self-play` → `training_data/data_<timestamp>.json`
2. Train: `train` binary → `training_models/azul_model_vN.safetensors` → deploys to `release_models/azul_alpha.safetensors` if ≥52% win rate
3. Evaluate: `headless --players mctsnn:600:release_models/azul_alpha.safetensors heuristicai`

The `headless --self-play` auto-discovers the latest `.safetensors` file in `training_models/` when no model path is specified.
