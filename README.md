# Azul Game Engine in Rust and WebAssembly

This project is a complete implementation of the board game Azul, written in Rust. The core game logic is compiled to WebAssembly (Wasm), allowing it to run in any modern web browser with a simple HTML and JavaScript frontend.

The engine enforces all the rules for a 2-player game, including tile drafting, pattern line placement, wall tiling, scoring, and end-game conditions.

## Features

- **Complete 2-Player Ruleset**: Implements all phases of the game according to the official rulebook.
- **Interactive Web UI**: A clickable user interface built with HTML, CSS, and JavaScript that allows for human vs. human play.
- **Performant Rust Engine**: The game logic is written in Rust for performance, memory safety, and reliability.
- **WebAssembly Compilation**: The Rust engine is compiled to a Wasm module, allowing it to run at near-native speed directly in the browser.
- **Command-Line Runner**: Includes a separate command-line interface (`main.rs`) for quick, text-based testing of the engine's logic.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

1.  **Rust Toolchain**: Includes `rustup`, `cargo`, and the Rust compiler. [Install Rust](https://www.rust-lang.org/tools/install).
2.  **`wasm-pack`**: The primary tool for building Rust-generated WebAssembly.
    ```bash
    cargo install wasm-pack
    ```
3.  **A Local Web Server**: Required to serve the files to your browser. Python's built-in server is a simple option.

## How to Run

There are two ways to run this project: the interactive web version and the command-line test version.

### 1. Web Version (Main UI)

This is the primary way to play the game.

**Step 1: Build the WebAssembly Package**
Navigate to the project's root directory in your terminal and run the following command. This compiles the Rust library (`lib.rs`) into a Wasm module and creates a `pkg` directory.

```bash
wasm-pack build --target web

python3 -m http.server

