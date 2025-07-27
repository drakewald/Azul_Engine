// MODIFIED: This entire module will only be compiled when the "native" feature is enabled.
#![cfg(feature = "native")]

use crate::{
    ai::{
        mcts_lib::{Mcts, MctsPolicy},
        nn::NeuralNetwork,
        AIAgent,
    },
    GameState, Move, MoveSource, Tile,
};
use std::any::Any;
use std::collections::HashMap;

// --- Constants for Network Architecture ---
const NUM_FACTORIES: usize = 9;
const NUM_COLORS: usize = 5;
const MAX_CENTER_TILES: usize = 27;
const MAX_PLAYERS: usize = 4;
const PATTERN_LINE_SLOTS: usize = 5 * 5;
const WALL_SLOTS: usize = 5 * 5;
const FLOOR_SLOTS: usize = 7;

const INPUT_SIZE: usize = (NUM_FACTORIES * NUM_COLORS * 4)
                        + (MAX_CENTER_TILES * NUM_COLORS)
                        + (MAX_PLAYERS * (1 + PATTERN_LINE_SLOTS + WALL_SLOTS + FLOOR_SLOTS + 1))
                        + 1;
const POLICY_SIZE: usize = (NUM_FACTORIES * NUM_COLORS) + NUM_COLORS;

// --- Helper Functions ---
fn color_to_index(tile: Tile) -> usize {
    match tile {
        Tile::Blue => 0, Tile::Yellow => 1, Tile::Red => 2,
        Tile::Black => 3, Tile::White => 4,
    }
}

fn move_to_policy_index(move_tile: Tile, move_source: &MoveSource) -> Option<usize> {
    let color_idx = color_to_index(move_tile);
    match move_source {
        MoveSource::Factory(idx) => Some(*idx * NUM_COLORS + color_idx),
        MoveSource::Center => Some(NUM_FACTORIES * NUM_COLORS + color_idx),
    }
}

#[derive(Clone)]
struct NnPolicy {
    nn: NeuralNetwork,
}

impl MctsPolicy for NnPolicy {
    fn evaluate(&self, game_state: &GameState) -> (f32, HashMap<Move, f32>) {
        let input = self.state_to_input(game_state);
        let nn_output = self.nn.forward(&input);
        let value = *nn_output.last().unwrap_or(&0.0);
        let raw_policy = &nn_output[..POLICY_SIZE];
        let legal_moves = game_state.get_legal_moves();
        let policy_map = self.mask_and_normalize_policy(&legal_moves, raw_policy);
        (value, policy_map)
    }
}

impl NnPolicy {
    fn state_to_input(&self, game_state: &GameState) -> Vec<f32> {
        let mut input = vec![0.0; INPUT_SIZE];
        let mut offset = 0;
        for factory_idx in 0..NUM_FACTORIES {
            if let Some(factory) = game_state.factories.get(factory_idx) {
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
        for (i, tile) in game_state.center.iter().enumerate().take(MAX_CENTER_TILES) {
            let color_idx = color_to_index(*tile);
            input[offset + (i * NUM_COLORS) + color_idx] = 1.0;
        }
        offset += MAX_CENTER_TILES * NUM_COLORS;
        for player_idx in 0..MAX_PLAYERS {
            if let Some(player) = game_state.players.get(player_idx) {
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
        input[offset] = (game_state.current_player_idx as f32 + 1.0) / MAX_PLAYERS as f32;
        input
    }

    fn mask_and_normalize_policy(&self, legal_moves: &[Move], raw_policy: &[f32]) -> HashMap<Move, f32> {
        let mut masked_policy = HashMap::new();
        let mut total_prob = 0.0;
        let unique_takes: HashMap<(MoveSource, Tile), ()> = legal_moves.iter().map(|m| ((m.source.clone(), m.tile), ())).collect();
        for (source, tile) in unique_takes.keys() {
            if let Some(index) = move_to_policy_index(*tile, source) {
                if let Some(prob) = raw_policy.get(index) {
                    let positive_prob = prob.max(0.0);
                    masked_policy.insert((source.clone(), *tile), positive_prob);
                    total_prob += positive_prob;
                }
            }
        }
        let mut final_policy = HashMap::new();
        if total_prob > 0.0 {
            for m in legal_moves {
                if let Some(prob) = masked_policy.get(&(m.source.clone(), m.tile)) {
                    final_policy.insert(m.clone(), prob / total_prob);
                }
            }
        }
        if final_policy.is_empty() && !legal_moves.is_empty() {
            let prob = 1.0 / legal_moves.len() as f32;
            for m in legal_moves { final_policy.insert(m.clone(), prob); }
        }
        final_policy
    }
}

pub struct MctsNnAI {
    mcts: Option<Mcts<NnPolicy>>,
    iterations: u32,
    model_path: Option<String>,
    model_bytes: Option<Vec<u8>>,
}

impl MctsNnAI {
    pub fn new(iterations: u32, model_path: Option<String>, model_bytes: Option<Vec<u8>>) -> Self {
        Self { mcts: None, iterations, model_path, model_bytes }
    }

    pub fn get_mcts_policy(&self) -> Option<Vec<f32>> {
        if let Some(mcts) = &self.mcts {
            let root = &mcts.tree[0];
            if root.visit_count == 0 { return None; }
            let mut policy_vec = vec![0.0; POLICY_SIZE];
            for (mv, child_idx) in &root.children {
                if let Some(policy_idx) = move_to_policy_index(mv.tile, &mv.source) {
                    let child_visits = mcts.tree[*child_idx].visit_count;
                    policy_vec[policy_idx] = child_visits as f32 / root.visit_count as f32;
                }
            }
            return Some(policy_vec);
        }
        None
    }

    pub fn state_to_input(&self, game_state: &GameState) -> Option<Vec<f32>> {
        self.mcts.as_ref().map(|mcts| mcts.policy_handler.state_to_input(game_state))
    }
}

impl AIAgent for MctsNnAI {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move> {
        if self.mcts.is_none() {
            let hidden_size = 256;
            let value_size = 1;
            
            let nn = if let Some(bytes) = &self.model_bytes {
                NeuralNetwork::from_bytes(bytes).unwrap_or_else(|e| {
                    println!("Failed to load model from bytes: {}, creating new.", e);
                    NeuralNetwork::new(&[INPUT_SIZE, hidden_size, POLICY_SIZE + value_size])
                })
            } else if let Some(path) = &self.model_path {
                println!("Attempting to load model from path: {} (placeholder)", path);
                NeuralNetwork::new(&[INPUT_SIZE, hidden_size, POLICY_SIZE + value_size])
            } else {
                NeuralNetwork::new(&[INPUT_SIZE, hidden_size, POLICY_SIZE + value_size])
            };

            let policy_handler = NnPolicy { nn };
            self.mcts = Some(Mcts::new(game_state.clone(), policy_handler));
        }

        let mcts = self.mcts.as_mut().unwrap();
        mcts.sync_tree_with_state(game_state);
        mcts.run_search(self.iterations);
        mcts.best_move()
    }

    fn as_any(&mut self) -> &mut dyn Any { self }
}
