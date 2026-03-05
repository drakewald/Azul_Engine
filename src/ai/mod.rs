use crate::{GameState, Move};
use std::any::Any;

pub mod heuristic_utils;
pub mod simple_ai;
pub mod heuristic_ai;
pub mod greedy_ai;
pub mod convergence_ai;
pub mod cluster_ai;
pub mod human_agent;
pub mod mcts_lib;

pub mod nn;
pub mod mcts_nn_ai;


pub trait AIAgent {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move>;
    fn as_any(&mut self) -> &mut dyn Any;
}
