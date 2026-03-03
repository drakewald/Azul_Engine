#!/usr/bin/env bash
# bootstrap.sh — train two fresh models on all existing data, tournament decides the base model.
#
# Usage:
#   ./bootstrap.sh
#
# Clears training_models/ before running so the new model starts as v1.
# Does NOT touch training_data/ — all existing data is reused.

set -euo pipefail

echo "Building release binary..."
cargo build --release --features="native" --bin bootstrap
echo ""

echo "Clearing old training models..."
rm -f training_models/*.safetensors
echo ""

./target/release/bootstrap
