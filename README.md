Azul Game Engine in Rust and WebAssembly

This project is a complete implementation of the board game Azul for 2-4 players, written in Rust. The core game logic is compiled to WebAssembly (Wasm) to run in any modern web browser, and it also includes a headless simulation runner for benchmarking AI agents.

The engine enforces all rules of the game, including tile drafting, pattern line placement, wall tiling, scoring, and end-game conditions.

Key Features

    Complete 2-4 Player Ruleset: Implements all phases of the game according to the official rulebook for up to four players.

    Multiple AI Agents: Includes several built-in AIs (Simple, Heuristic, and MCTS) that can play against humans or each other.

    Headless Simulation Runner: A dedicated command-line interface for running thousands of AI vs. AI games to collect statistics and benchmark performance.

    Interactive Web UI: A clickable user interface built with HTML, CSS, and JavaScript that allows for human or AI play.

    Performant Rust Engine: The game logic is written in Rust for performance, memory safety, and reliability, then compiled to a Wasm module to run at near-native speed in the browser.

Prerequisites

Before you begin, ensure you have the following installed:

    Rust Toolchain: Includes rustup, cargo, and the Rust compiler. (Install Rust)

    wasm-pack: The primary tool for building Rust-generated WebAssembly.
    Bash

    cargo install wasm-pack

    A Local Web Server (for the web UI only): Python's built-in server is a simple option.

How to Run

There are two primary ways to run this project: the interactive web version and the headless simulation.

1. Web Version (Interactive UI)

This is the primary way to play the game against an AI or another human.

Step 1: Build the WebAssembly Package
This command compiles the Rust library (lib.rs) into a Wasm module and creates a pkg directory.
Bash

wasm-pack build --target web

Step 2: Start a Local Web Server
From the project's root directory, run a simple web server.
Bash

python3 -m http.server

Step 3: Open in Browser
Navigate to http://localhost:8000 in your web browser to play.

2. Headless Simulation (AI vs. AI)

This is used for running AI matchups and collecting performance statistics. The simulation is controlled via command-line arguments.

Usage

    --players or -p: (Required) A space-separated list of 2 to 4 AI agents to play against each other.

        Valid agent names are: SimpleAI, HeuristicAI, MctsAI.

    --games or -g: (Optional) The number of games to simulate. Defaults to 100.

Examples

Run a 2-player match for 100 games (default):
Bash

cargo run --release --bin headless -- --players HeuristicAI MctsAI

Run a 4-player match for 500 games:
Bash

cargo run --release --bin headless -g 500 -p MctsAI MctsAI HeuristicAI SimpleAI

Viewing the Results

After the simulation completes, two files will be saved in the stats/ directory:

    summary_stats.json: Contains the high-level win/loss statistics.

    game_logs.json: Contains a detailed turn-by-turn history of every game, which can be used as a dataset for training machine learning models.