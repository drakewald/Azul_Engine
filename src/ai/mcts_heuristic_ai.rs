use crate::{
    ai::{
        heuristic_ai::HeuristicAI,
        mcts_lib::{Mcts, MctsPolicy},
        AIAgent,
    },
    GameState, Move,
};
use std::any::Any;
use std::collections::HashMap;

#[derive(Clone)]
struct HeuristicPolicy;

impl MctsPolicy for HeuristicPolicy {
    // MODIFIED: This function now runs a simulation to get a value,
    // which is required by the new AlphaGo-style search algorithm.
    fn evaluate(&self, game_state: &GameState) -> (f32, HashMap<Move, f32>) {
        // The policy part remains the same: give all legal moves an equal chance.
        let legal_moves = game_state.get_legal_moves();
        let probability = if legal_moves.is_empty() { 0.0 } else { 1.0 / legal_moves.len() as f32 };
        let policy = legal_moves.into_iter().map(|m| (m, probability)).collect();

        // The value part: run one simulation to estimate the value of this position.
        let scores = self.run_simulation(game_state);
        let value = scores[game_state.current_player_idx];
        
        (value, policy)
    }
}

// Added a helper function for the simulation logic.
impl HeuristicPolicy {
    fn run_simulation(&self, game_state: &GameState) -> Vec<f32> {
        let mut sim_state = game_state.clone();
        let mut simulation_agent = HeuristicAI;
        while !sim_state.end_game_triggered {
            if sim_state.is_round_over() {
                sim_state.run_tiling_phase();
                sim_state.refill_factories();
                continue;
            }
            if let Some(best_move) = simulation_agent.get_move(&sim_state) {
                sim_state.apply_move(&best_move);
            } else {
                break;
            }
        }
        sim_state.run_tiling_phase();
        sim_state.apply_end_game_scoring();
        sim_state.players.iter().map(|p| p.score as f32).collect()
    }
}

pub struct MctsHeuristicAI {
    mcts: Option<Mcts<HeuristicPolicy>>,
    iterations: u32,
}

impl MctsHeuristicAI {
    pub fn new(iterations: u32) -> Self {
        Self {
            mcts: None,
            iterations,
        }
    }
}

impl AIAgent for MctsHeuristicAI {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move> {
        if self.mcts.is_none() {
            self.mcts = Some(Mcts::new(game_state.clone(), HeuristicPolicy));
        }

        let mcts = self.mcts.as_mut().unwrap();
        
        mcts.sync_tree_with_state(game_state);
        
        mcts.run_search(self.iterations);
        mcts.best_move()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
