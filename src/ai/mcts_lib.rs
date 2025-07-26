use crate::{GameState, Move};
use std::collections::HashMap;

// MODIFIED: Added `Clone` as a requirement for any MctsPolicy.
pub trait MctsPolicy: Clone {
    fn evaluate(&self, game_state: &GameState) -> (f32, HashMap<Move, f32>);
    fn simulate(&self, game_state: &GameState) -> Vec<f32>;
}

pub struct Node {
    pub parent: Option<usize>,
    pub children: Vec<(Move, usize)>,
    pub untried_moves: Vec<Move>,
    pub visit_count: u32,
    pub total_action_value: f32,
    pub prior_probability: f32,
    pub game_state: GameState,
}

impl Node {
    fn new(parent: Option<usize>, prior: f32, game_state: GameState) -> Self {
        let untried_moves = game_state.get_legal_moves();
        Self {
            parent,
            children: Vec::new(),
            untried_moves,
            visit_count: 0,
            total_action_value: 0.0,
            prior_probability: prior,
            game_state,
        }
    }

    pub fn mean_action_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_action_value / self.visit_count as f32
        }
    }
}

pub struct Mcts<P: MctsPolicy> {
    pub tree: Vec<Node>,
    pub policy_handler: P,
}

// MODIFIED: The generic parameter P is now also constrained by Clone.
impl<P: MctsPolicy + Clone> Mcts<P> {
    pub fn new(initial_state: GameState, policy_handler: P) -> Self {
        Self {
            tree: vec![Node::new(None, 1.0, initial_state)],
            policy_handler,
        }
    }
    
    pub fn sync_tree_with_state(&mut self, current_game_state: &GameState) {
        let new_root_child_idx = self.tree[0].children.iter()
            .find(|(_, child_idx)| self.tree[*child_idx].game_state.players == current_game_state.players)
            .map(|(_, child_idx)| *child_idx);

        if let Some(child_idx) = new_root_child_idx {
            let new_root_state = self.tree[child_idx].game_state.clone();
            // MODIFIED: Pass a clone of the policy handler.
            *self = Mcts::new(new_root_state, self.policy_handler.clone());
        } else {
            // MODIFIED: Pass a clone of the policy handler.
            *self = Mcts::new(current_game_state.clone(), self.policy_handler.clone());
        }
    }

    pub fn best_move(&self) -> Option<Move> {
        if self.tree.is_empty() { return None; }
        
        let root = &self.tree[0];
        root.children.iter()
            .max_by_key(|(_, child_idx)| self.tree[*child_idx].visit_count)
            .map(|(m, _)| m.clone())
    }

    pub fn run_search(&mut self, iterations: u32) {
        for _ in 0..iterations {
            let node_to_simulate_idx = self.select_and_expand();
            let sim_game_state = self.tree[node_to_simulate_idx].game_state.clone();
            let scores = self.policy_handler.simulate(&sim_game_state);
            self.backpropagation(node_to_simulate_idx, &scores);
        }
    }

    fn select_and_expand(&mut self) -> usize {
        let mut current_idx = 0;
        loop {
            if !self.tree[current_idx].untried_moves.is_empty() {
                let node = &mut self.tree[current_idx];
                let next_move = node.untried_moves.pop().unwrap();
                let mut new_state = node.game_state.clone();
                new_state.apply_move(&next_move);
                let prior_prob = 1.0 / (node.children.len() + 1) as f32;
                let new_node = Node::new(Some(current_idx), prior_prob, new_state);
                let new_node_idx = self.tree.len();
                self.tree.push(new_node);
                self.tree[current_idx].children.push((next_move, new_node_idx));
                return new_node_idx;
            }

            let node = &self.tree[current_idx];
            if node.children.is_empty() {
                return current_idx;
            }

            let best_child_idx = node.children.iter()
                .map(|(_, child_idx)| *child_idx)
                .max_by(|&a_idx, &b_idx| {
                    let a_score = self.puct_score(a_idx, node.visit_count);
                    let b_score = self.puct_score(b_idx, node.visit_count);
                    a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            
            current_idx = best_child_idx;
        }
    }

    fn backpropagation(&mut self, start_idx: usize, scores: &[f32]) {
        let mut current_idx = Some(start_idx);
        while let Some(idx) = current_idx {
            let node = &mut self.tree[idx];
            node.visit_count += 1;
            let player_idx = node.game_state.current_player_idx;
            node.total_action_value += scores[player_idx];
            current_idx = node.parent;
        }
    }

    fn puct_score(&self, node_idx: usize, parent_visit_count: u32) -> f32 {
        let node = &self.tree[node_idx];
        let exploration_constant = 1.41;
        let q_value = node.mean_action_value();
        let p_value = node.prior_probability;
        let exploration_term = exploration_constant * p_value * (parent_visit_count as f32).sqrt() / (1.0 + node.visit_count as f32);
        q_value + exploration_term
    }
}
