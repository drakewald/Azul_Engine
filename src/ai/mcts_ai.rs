use crate::{ai::AIAgent, GameState, Move};
use rand::seq::SliceRandom; 

// --- Data Structures for the MCTS Tree ---

struct Node {
    game_state: GameState,
    parent: Option<usize>,
    children: Vec<usize>,
    action: Option<Move>, 
    visits: u32,
    scores: Vec<f64>, 
}

pub struct MctsAI {
    nodes: Vec<Node>, 
}

impl MctsAI {
    pub fn new(initial_state: GameState) -> Self {
        let num_players = initial_state.players.len();
        let root_node = Node {
            game_state: initial_state,
            parent: None,
            children: Vec::new(),
            action: None,
            visits: 0,
            scores: vec![0.0; num_players],
        };
        MctsAI { nodes: vec![root_node] }
    }

    fn run_search(&mut self, iterations: u32) {
        for _ in 0..iterations {
            let leaf_index = self.select();
            let expanded_children = self.expand(leaf_index);
            
            let node_to_simulate_from = if !expanded_children.is_empty() {
                *expanded_children.choose(&mut rand::thread_rng()).unwrap()
            } else {
                leaf_index
            };
            
            let final_scores = self.simulate(node_to_simulate_from);
            self.backpropagate(node_to_simulate_from, &final_scores);
        }
    }

    fn select(&self) -> usize {
        let mut current_index = 0;
        loop {
            let node = &self.nodes[current_index];
            if node.children.is_empty() {
                return current_index;
            }
            
            let uct = |child_index: usize| -> f64 {
                let child = &self.nodes[child_index];
                if child.visits == 0 {
                    return f64::INFINITY;
                }
                let exploitation = child.scores[node.game_state.current_player_idx] / child.visits as f64;
                let exploration = 2.0 * ( (node.visits as f64).ln() / child.visits as f64 ).sqrt();
                exploitation + exploration
            };
            
            current_index = *node.children.iter().max_by(|a, b| uct(**a).partial_cmp(&uct(**b)).unwrap()).unwrap();
        }
    }

    fn expand(&mut self, node_index: usize) -> Vec<usize> {
        let legal_moves = self.nodes[node_index].game_state.get_legal_moves();
        if legal_moves.is_empty() {
            return Vec::new();
        }
        
        let mut new_children_indices = Vec::new();
        for action in legal_moves {
            let mut new_state = self.nodes[node_index].game_state.clone();
            new_state.apply_move(&action);
            
            let new_node = Node {
                game_state: new_state,
                parent: Some(node_index),
                children: Vec::new(),
                action: Some(action),
                visits: 0,
                scores: vec![0.0; self.nodes[0].game_state.players.len()],
            };
            
            let new_index = self.nodes.len();
            self.nodes.push(new_node);
            self.nodes[node_index].children.push(new_index);
            new_children_indices.push(new_index);
        }
        new_children_indices
    }

    fn simulate(&self, node_index: usize) -> Vec<u32> {
        let mut sim_state = self.nodes[node_index].game_state.clone();
        
        while !sim_state.end_game_triggered {
            let moves = sim_state.get_legal_moves();
            if moves.is_empty() {
                let mut discard_pile_ref = std::mem::take(&mut sim_state.discard_pile);
                for p in &mut sim_state.players {
                    p.run_tiling_phase(&mut discard_pile_ref);
                }
                sim_state.discard_pile = discard_pile_ref;
                sim_state.refill_factories();
                continue;
            }
            let random_move = moves.choose(&mut rand::thread_rng()).unwrap();
            sim_state.apply_move(random_move);
        }
        
        let mut discard_pile_ref = std::mem::take(&mut sim_state.discard_pile);
        for p in &mut sim_state.players {
            p.run_tiling_phase(&mut discard_pile_ref);
        }
        sim_state.discard_pile = discard_pile_ref;
        
        sim_state.apply_end_game_scoring();
        
        sim_state.players.iter().map(|p| p.score).collect()
    }

    fn backpropagate(&mut self, start_index: usize, final_scores: &[u32]) {
        let mut current_index = Some(start_index);
        while let Some(index) = current_index {
            let node = &mut self.nodes[index];
            node.visits += 1;
            for (i, score) in final_scores.iter().enumerate() {
                node.scores[i] += *score as f64;
            }
            current_index = node.parent;
        }
    }
}

impl AIAgent for MctsAI {
    fn get_move(&self, game_state: &GameState) -> Option<Move> {
        let mut mcts = MctsAI::new(game_state.clone());
        
        mcts.run_search(1000); 

        let best_child_index = mcts.nodes[0].children.iter()
            .max_by_key(|&&child_index| mcts.nodes[child_index].visits)?;
            
        mcts.nodes[*best_child_index].action.clone()
    }
}