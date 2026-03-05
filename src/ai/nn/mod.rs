use candle_core::Tensor;
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};
use crate::{GameState, MoveSource, Tile};

// Encoding constants shared across all architectures.
// These describe the state/action representation, not any particular network topology.
pub const INPUT_SIZE: usize = (9 * 5 * 4) + (27 * 5) + (4 * (1 + 25 + 25 + 7 + 1)) + 1;
pub const POLICY_SIZE: usize = (9 * 5) + 5;

pub const NUM_FACTORIES: usize = 9;
pub const NUM_COLORS: usize = 5;
pub const MAX_CENTER_TILES: usize = 27;
pub const MAX_PLAYERS: usize = 4;
const PATTERN_LINE_SLOTS: usize = 5 * 5;
const WALL_SLOTS: usize = 5 * 5;
const FLOOR_SLOTS: usize = 7;

fn color_to_index(tile: Tile) -> usize {
    match tile {
        Tile::Blue => 0, Tile::Yellow => 1, Tile::Red => 2,
        Tile::Black => 3, Tile::White => 4,
    }
}

/// Map a (tile, source) pair to its index in the flat policy vector.
/// Destination is not encoded — the policy selects *what and where from*,
/// not which pattern line to place into.
pub fn move_to_policy_index(tile: Tile, source: &MoveSource) -> Option<usize> {
    let color_idx = color_to_index(tile);
    match source {
        MoveSource::Factory(idx) => Some(*idx * NUM_COLORS + color_idx),
        MoveSource::Center => Some(NUM_FACTORIES * NUM_COLORS + color_idx),
    }
}

/// Encode a game state into the flat float vector used as NN input.
pub fn encode_state(state: &GameState) -> Vec<f32> {
    let mut input = vec![0.0f32; INPUT_SIZE];
    let mut offset = 0;
    for factory_idx in 0..NUM_FACTORIES {
        if let Some(factory) = state.factories.get(factory_idx) {
            for tile in factory {
                let color_idx = color_to_index(*tile);
                for slot in 0..4 {
                    let index = offset + (color_idx * 4) + slot;
                    if input[index] == 0.0 { input[index] = 1.0; break; }
                }
            }
        }
        offset += NUM_COLORS * 4;
    }
    for (i, tile) in state.center.iter().enumerate().take(MAX_CENTER_TILES) {
        let color_idx = color_to_index(*tile);
        input[offset + (i * NUM_COLORS) + color_idx] = 1.0;
    }
    offset += MAX_CENTER_TILES * NUM_COLORS;
    for player_idx in 0..MAX_PLAYERS {
        if let Some(player) = state.players.get(player_idx) {
            input[offset] = player.score as f32 / 100.0;
            offset += 1;
            for (row_idx, line) in player.pattern_lines.iter().enumerate() {
                for i in 0..line.len() { input[offset + (row_idx * 5) + i] = 1.0; }
            }
            offset += PATTERN_LINE_SLOTS;
            for (row_idx, row) in player.wall.iter().enumerate() {
                for (col_idx, tile_option) in row.iter().enumerate() {
                    if tile_option.is_some() { input[offset + (row_idx * 5) + col_idx] = 1.0; }
                }
            }
            offset += WALL_SLOTS;
            for i in 0..player.floor_line.len().min(FLOOR_SLOTS) { input[offset + i] = 1.0; }
            offset += FLOOR_SLOTS;
            if player.has_first_player_marker { input[offset] = 1.0; }
            offset += 1;
        } else {
            offset += 1 + PATTERN_LINE_SLOTS + WALL_SLOTS + FLOOR_SLOTS + 1;
        }
    }
    input[offset] = (state.current_player_idx as f32 + 1.0) / MAX_PLAYERS as f32;
    input
}

pub mod tactician;
pub use tactician::{TacticianConfig, TacticianNet};

/// Inference interface shared by all Azul network architectures.
/// Object-safe — usable as `Rc<dyn AzulNetwork>` inside MCTS.
/// Training-specific concerns (VarMap, save, backward) live on concrete types only.
pub trait AzulNetwork {
    fn forward(&self, input: &[f32]) -> (Vec<f32>, f32);
    fn forward_batch(&self, inputs: &[Vec<f32>]) -> Vec<(Vec<f32>, f32)>;
}

/// A residual block: FC → LN → ReLU → FC → LN, with a skip connection and final ReLU.
/// Shared building block available to any species in this module.
pub(super) struct ResBlock {
    fc1: Linear,
    ln1: LayerNorm,
    fc2: Linear,
    ln2: LayerNorm,
}

impl ResBlock {
    pub(super) fn new(width: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(width, width, vb.pp("fc1"))?,
            ln1: layer_norm(width, 1e-5, vb.pp("ln1"))?,
            fc2: linear(width, width, vb.pp("fc2"))?,
            ln2: layer_norm(width, 1e-5, vb.pp("ln2"))?,
        })
    }

    pub(super) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.fc1.forward(x)?.apply(&self.ln1)?.relu()?;
        let h = self.fc2.forward(&h)?.apply(&self.ln2)?;
        (x + h)?.relu()
    }
}
