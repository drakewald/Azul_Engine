use crate::{GameState, Move};

// Module declarations
pub mod simple_ai;
pub mod heuristic_ai;
pub mod human_agent;
pub mod nn;
pub mod mcts_lib;
pub mod mcts_heuristic_ai;
pub mod mcts_nn_ai; 

// The trait that all AI agents will implement.
pub trait AIAgent {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move>;
}
