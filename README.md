Azul Game Engine in Rust and WebAssembly

This project is a complete implementation of the board game Azul for 2-4 players, written in Rust. The core game logic is compiled to WebAssembly (Wasm) to run in any modern web browser, and it also includes a headless simulation runner for benchmarking and training AI agents.

The engine enforces all rules of the game, including tile drafting, pattern line placement, wall tiling, scoring, and end-game conditions.
Key Features

    Complete 2-4 Player Ruleset: Implements all phases of the game according to the official rulebook for up to four players.

    Multiple AI Agents: Includes several built-in AIs (Simple, Heuristic, MCTS with heuristics, and a self-learning MCTS with a neural network) that can play against humans or each other.

    Headless Simulation & Training Runner: A dedicated command-line interface for running AI vs. AI games, generating training data via self-play, and training the neural network.

    Interactive Web UI: A clickable user interface built with HTML, CSS, and JavaScript that allows for human or AI play.

    Performant Rust Engine: The game logic is written in Rust for performance and memory safety, then compiled to a Wasm module to run at near-native speed in the browser.

Prerequisites

Before you begin, ensure you have the following installed:

    Rust Toolchain: Includes rustup, cargo, and the Rust compiler. (Install Rust)

    wasm-pack: The primary tool for building Rust-generated WebAssembly.

    cargo install wasm-pack

    A Local Web Server (for the web UI only): Python's built-in server is a simple option.

    PyTorch C++ Library (LibTorch): Required for the training functionality. See the tch-rs crate documentation for detailed setup instructions.

How to Run

There are two primary ways to run this project: the interactive web version and the headless simulation/training.
1. Web Version (Interactive UI)

This is the primary way to play the game against an AI or another human.

Step 1: Build the WebAssembly Package
This command compiles the Rust library (lib.rs) into a Wasm module and creates a pkg directory.

wasm-pack build --target web

Step 2: Start a Local Web Server
From the project's root directory, run a simple web server.

python3 -m http.server

Step 3: Open in Browser
Navigate to http://localhost:8000 in your web browser to play.
2. Headless Simulation (AI vs. AI)

This is used for running AI matchups and collecting performance statistics. The simulation is controlled via command-line arguments.
Usage

    --players or -p: (Required) A space-separated list of 2 to 4 AI agents.

        Valid names: simpleai, heuristicai, mctsheuristic, mctsnn.

        For MCTS agents, you can specify iterations with a colon (e.g., mctsheuristic:1000).

        For mctsnn, you can specify a model to load (e.g., mctsnn:200:release_models/azul_alpha.ot).

    --games or -g: (Optional) The number of games to simulate. Defaults to 100.

Examples

Run a 2-player match for 100 games:

cargo run --release --features="native" --bin headless -- --players heuristicai mctsheuristic

Run a 4-player match for 500 games:

cargo run --release --features="native" --bin headless -g 500 -p mctsheuristic mctsheuristic heuristicai simpleai

3. Training the Neural Network AI

This is a cyclical process to make the mctsnn agent smarter over time.
Step 1: Generate Training Data (Self-Play)

Have the current best version of the mctsnn AI play against itself to generate a dataset.

# The agent string tells the AI to use 200 iterations. The runner will auto-find the latest model in training_models/.
cargo run --release --features="native" --bin headless -- --self-play --players mctsnn:200 --games 100

    --self-play: Activates data generation mode.

    --self-play-players 4 (Optional): Generates data from 4-player games instead of the default 2.

    cargo run --release --features="native" --bin headless -- --self-play --self-play-players 3 --players mctsnn:200 --games 50

This will create a new data file in the training_data/ directory.
Step 2: Train a New Model

Run the train binary. It will automatically find the latest dataset in training_data/ and the latest model in training_models/, fine-tune it, and save the result as the next version.

cargo run --release --features="native" --bin train

This will create a new, smarter model (e.g., training_models/azul_model_v2.ot) and also deploy a copy for the web app to release_models/azul_alpha.ot.
Step 3: Evaluate and Repeat

Test your new model against another AI to see if it has improved.

# Use the newly created release model in a head-to-head match.
cargo run --release --features="native" --bin headless -- --players mctsnn:200:release_models/azul_alpha.ot mctsheuristic:200

If the win rate has improved, you can repeat the cycle, starting again from Step 1 to generate even higher-quality data with your new, smarter AI.