use crate::{
    ai::{heuristic_ai::HeuristicAI, AIAgent},
    GameState, Move,
};

/// A node within the Monte Carlo Search Tree.
struct Node {
    game_state: GameState,
    parent: Option<usize>,
    children: Vec<usize>,
    action: Option<Move>,      // The move that resulted in this node's state.
    untried_actions: Vec<Move>,
    visits: u32,
    scores: Vec<f64>, // A score for each player in the game.
}

/// A stateful Monte Carlo Tree Search AI. It builds and reuses a single search
/// tree throughout the game, making it significantly stronger and more performant
/// than recreating the tree on every turn.
pub struct MctsAI {
    nodes: Vec<Node>,
    /// The index of the node that represents the current, actual game state.
    root_idx: usize,
}

impl MctsAI {
    /// Creates a new, empty MCTS agent. The tree is initialized on the first move.
    pub fn new() -> Self {
        MctsAI {
            nodes: Vec::new(),
            root_idx: 0,
        }
    }

    /// Runs the MCTS algorithm for a specified number of iterations.
    fn run_search(&mut self, iterations: u32) {
        for _ in 0..iterations {
            // Phase 1: Selection & Expansion
            let node_index = self.select_and_expand();
            // Phase 2: Simulation
            let final_scores = self.simulate(node_index);
            // Phase 3: Backpropagation
            self.backpropagate(node_index, &final_scores);
        }
    }

    /// Aligns the tree's root with the current actual game state.
    /// This allows the AI to reuse work done on previous turns.
    fn sync_with_game_state(&mut self, game_state: &GameState) {
        if self.nodes.is_empty() {
            self.create_root(game_state);
            return;
        }

        // Phase 1: FIND the index of the child that matches the new state.
        // We do this without modifying `self` so the borrow is fine.
        let new_root_child_idx: Option<usize> = self.nodes[self.root_idx]
            .children
            .iter()
            .find(|&&child_idx| self.nodes[child_idx].game_state.players == game_state.players)
            .copied(); // .copied() turns Option<&usize> into Option<usize>

        // Phase 2: MODIFY `self` based on the result of the find operation.
        if let Some(child_idx) = new_root_child_idx {
            self.root_idx = child_idx;
            self.nodes[self.root_idx].parent = None;
        } else {
            // If no matching child was found, reset the tree.
            self.create_root(game_state);
        }
    }

    /// Helper to create a new root node for the search tree.
    fn create_root(&mut self, game_state: &GameState) {
        let root_node = Node {
            untried_actions: game_state.get_legal_moves(),
            game_state: game_state.clone(),
            parent: None,
            children: Vec::new(),
            action: None,
            visits: 0,
            scores: vec![0.0; game_state.players.len()],
        };
        self.nodes = vec![root_node];
        self.root_idx = 0;
    }

    /// Phase 1: Finds the most promising node to expand.
    fn select_and_expand(&mut self) -> usize {
        let mut current_index = self.root_idx;
        loop {
            let node = &self.nodes[current_index];
            if !node.untried_actions.is_empty() {
                // If there are untried actions, expand this node.
                return self.expand(current_index);
            }
            if node.children.is_empty() {
                // This is a terminal leaf node.
                return current_index;
            }
            // Use UCT to select the best child to explore further.
            let uct = |child_index: usize| -> f64 {
                let child = &self.nodes[child_index];
                if child.visits == 0 { return f64::INFINITY; }
                
                // Exploitation: Average score of the child node.
                let exploitation = child.scores[node.game_state.current_player_idx] / child.visits as f64;
                // Exploration: Encourages visiting less-explored nodes.
                let exploration = 2.0 * ((node.visits as f64).ln() / child.visits as f64).sqrt();
                
                exploitation + exploration
            };
            current_index = *self.nodes[current_index].children.iter()
                .max_by(|a, b| uct(**a).partial_cmp(&uct(**b)).unwrap())
                .unwrap();
        }
    }
    
    /// Expands the tree with a new node.
    fn expand(&mut self, node_index: usize) -> usize {
        // Pop an untried action to create a new state.
        let action_to_try = self.nodes[node_index].untried_actions.pop().unwrap();
        let mut new_state = self.nodes[node_index].game_state.clone();
        new_state.apply_move(&action_to_try);

        let new_node = Node {
            untried_actions: new_state.get_legal_moves(),
            game_state: new_state,
            parent: Some(node_index),
            children: Vec::new(),
            action: Some(action_to_try),
            visits: 0,
            scores: vec![0.0; self.nodes[0].game_state.players.len()],
        };
        
        // Add the new node to the tree.
        let new_index = self.nodes.len();
        self.nodes.push(new_node);
        self.nodes[node_index].children.push(new_index);
        new_index
    }

    /// Phase 2: Simulates a random game from a node to its conclusion.
    fn simulate(&self, node_index: usize) -> Vec<u32> {
        let mut sim_state = self.nodes[node_index].game_state.clone();
        // A smarter simulation policy (like a simpler heuristic) leads to better results
        // than pure random choices.
        let mut simulation_agent = HeuristicAI;
        
        // Play until the final round is triggered.
        while !sim_state.end_game_triggered {
            if sim_state.is_round_over() {
                sim_state.run_tiling_phase();
                sim_state.refill_factories();
                continue;
            }
            
            // Get a move from the simple heuristic agent.
            if let Some(best_move) = simulation_agent.get_move(&sim_state) {
                sim_state.apply_move(&best_move);
            } else {
                // This branch should ideally not be taken if get_move is implemented correctly
                break;
            }
        }
        
        // Run the final tiling and scoring phase.
        sim_state.run_tiling_phase();
        sim_state.apply_end_game_scoring();
        
        sim_state.players.iter().map(|p| p.score).collect()
    }
    
    /// Phase 3: Propagates the simulation results back up the tree.
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
    /// The main entry point for the AI to select a move.
    fn get_move(&mut self, game_state: &GameState) -> Option<Move> {
        // 1. Update the tree to reflect the current game state.
        self.sync_with_game_state(game_state);
        
        // 2. Run the search to gather intelligence. The number of iterations
        // is the primary knob for controlling AI strength vs. thinking time.
        self.run_search(5000); 

        // 3. Select the best move from the children of the root node.
        // The most robust choice is the one most visited, not necessarily the one with the highest score.
        let best_child_index = self.nodes[self.root_idx].children.iter()
            .max_by_key(|&&child_idx| self.nodes[child_idx].visits)?;
            
        self.nodes[*best_child_index].action.clone()
    }
}