#!/usr/bin/env bash
# train_loop.sh — self-play → train → evaluate loop
#
# Usage:
#   ./util/train_loop.sh [loops] [games_per_player_count] [--multi]
#
# Defaults: 10 loops, 100 games per player count per iteration, 2p only.
# --multi: self-play runs 2p, 3p, and 4p in random order each iteration.

set -euo pipefail

cd "$(dirname "$0")/.."

LOOPS=10
GAMES=250
MULTI_PLAYER=false

# Separate --multi flag from positional args
args=()
for arg in "$@"; do
    if [[ "$arg" == "--multi" ]]; then
        MULTI_PLAYER=true
    else
        args+=("$arg")
    fi
done

LOOPS="${args[0]:-$LOOPS}"
GAMES="${args[1]:-$GAMES}"

MCTS_ITERS=300
EVAL_GAMES=100

HEADLESS="./target/release/headless"
TRAIN="./target/release/train"

echo "Building release binaries..."
cargo build --release --features="native" --bins
echo ""

if [ "$MULTI_PLAYER" = true ]; then
    echo "Starting training loop: $LOOPS iterations, $GAMES games per player count (2p/3p/4p random order)"
else
    echo "Starting training loop: $LOOPS iterations, $GAMES games (2p)"
fi
echo ""

for i in $(seq 1 "$LOOPS"); do
    echo "════════════════════════════════════════"
    echo "  Iteration $i / $LOOPS"
    echo "════════════════════════════════════════"

    # Self-play
    if [ "$MULTI_PLAYER" = true ]; then
        PLAYER_COUNTS=$(printf '2\n3\n4' | shuf)
    else
        PLAYER_COUNTS="2"
    fi

    for p in $PLAYER_COUNTS; do
        echo ""
        echo "── Self-play: ${p}p, $GAMES games ──"
        "$HEADLESS" \
            --self-play \
            --players "mctsnn:${MCTS_ITERS}" \
            --games "$GAMES" \
            --self-play-players "$p"
    done

    # Train on accumulated data
    echo ""
    echo "── Training ──"
    "$TRAIN"

    # Evaluate only once a release model exists
    if [ -f "release_models/azul_alpha.safetensors" ]; then
        echo ""
        echo "── Evaluation: mctsnn vs heuristicai ($EVAL_GAMES games) ──"
        "$HEADLESS" \
            --players "mctsnn:${MCTS_ITERS}:release_models/azul_alpha.safetensors" heuristicai \
            --games "$EVAL_GAMES"
    fi

    echo ""
done

echo "Training loop complete."
