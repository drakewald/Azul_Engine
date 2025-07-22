pub mod simple_ai;
pub mod mcts_ai;
pub mod heuristic_ai;

use crate::{GameState, Move};

//common interface for all AI agents
pub trait AIAgent {
    fn get_move(&self, game_state: &GameState) -> Option<Move>;
}