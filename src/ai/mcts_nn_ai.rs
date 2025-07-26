use crate::{
    ai::{
        mcts_lib::{Mcts, MctsPolicy},
        nn::NeuralNetwork,
        AIAgent,
    },
    GameState, Move,
};
use std::collections::HashMap;

// MODIFIED: Added `Clone` to the struct.
#[derive(Clone)]
struct NnPolicy {
    nn: NeuralNetwork,
}

impl MctsPolicy for NnPolicy {
    fn evaluate(&self, game_state: &GameState) -> (f32, HashMap<Move, f32>) {
        let input = self.state_to_input(game_state);
        let nn_output = self.nn.forward(&input);
        
        let value = nn_output.last().cloned().unwrap_or(0.0);
        
        let policy_vec = &nn_output[..nn_output.len() - 1];
        let legal_moves = game_state.get_legal_moves();
        
        let policy_map = legal_moves.into_iter().zip(policy_vec.iter().cloned()).collect();

        (value, policy_map)
    }

    fn simulate(&self, game_state: &GameState) -> Vec<f32> {
        let (value, _) = self.evaluate(game_state);
        vec![value; game_state.players.len()]
    }
}

impl NnPolicy {
    fn state_to_input(&self, _game_state: &GameState) -> Vec<f32> {
        vec![0.0; 181]
    }
}

pub struct MctsNnAI {
    mcts: Option<Mcts<NnPolicy>>,
    iterations: u32,
}

impl MctsNnAI {
    pub fn new(iterations: u32) -> Self {
        Self {
            mcts: None,
            iterations,
        }
    }
}

impl AIAgent for MctsNnAI {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move> {
        if self.mcts.is_none() {
            let input_size = 181;
            let hidden_size = 128;
            let policy_size = 1;
            let value_size = 1;
            let nn = NeuralNetwork::new(&[input_size, hidden_size, policy_size + value_size]);
            let policy_handler = NnPolicy { nn };
            self.mcts = Some(Mcts::new(game_state.clone(), policy_handler));
        }

        let mcts = self.mcts.as_mut().unwrap();
        
        mcts.sync_tree_with_state(game_state);
        
        mcts.run_search(self.iterations);
        mcts.best_move()
    }
}
