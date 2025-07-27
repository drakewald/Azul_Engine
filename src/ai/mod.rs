use crate::{GameState, Move};
use std::any::Any;

pub mod simple_ai;
pub mod heuristic_ai;
pub mod human_agent;
pub mod mcts_lib;
pub mod mcts_heuristic_ai;

// These modules will only be compiled when the "native" feature is enabled.
#[cfg(feature = "native")]
pub mod nn;
#[cfg(feature = "native")]
pub mod mcts_nn_ai;


pub trait AIAgent {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move>;
    fn as_any(&mut self) -> &mut dyn Any;
}
