use crate::{GameState, Move};
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use std::collections::HashMap;

pub trait MctsPolicy: Clone {
    fn evaluate(&self, game_state: &GameState) -> (f32, HashMap<Move, f32>);
}

pub struct Node {
    pub parent: Option<usize>,
    /// Child nodes that have already been created (visited at least once).
    pub children: Vec<(Move, usize)>,
    /// Prior probabilities returned by the policy network.
    /// `None`  — node has never been evaluated (it is an unvisited leaf).
    /// `Some(map)` — evaluated; the map covers all legal moves including those
    ///               not yet expanded into child nodes.
    pub priors: Option<HashMap<Move, f32>>,
    pub visit_count: u32,
    pub total_action_value: f32,
    pub prior_probability: f32,
    /// The move that was applied to reach this node from its parent.
    /// `None` for the root node.
    pub applied_move: Option<Move>,
    /// The player whose turn it is at this node.
    pub current_player_idx: usize,
}

impl Node {
    fn new(
        parent: Option<usize>,
        prior: f32,
        applied_move: Option<Move>,
        current_player_idx: usize,
    ) -> Self {
        Self {
            parent,
            children: Vec::new(),
            priors: None,
            visit_count: 0,
            total_action_value: 0.0,
            prior_probability: prior,
            applied_move,
            current_player_idx,
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
    pub add_noise: bool,
    /// The game state at the root of the current search tree.
    pub root_state: GameState,
}

impl<P: MctsPolicy + Clone> Mcts<P> {
    pub fn new(initial_state: GameState, policy_handler: P, add_noise: bool) -> Self {
        let root_player_idx = initial_state.current_player_idx;
        Self {
            tree: vec![Node::new(None, 1.0, None, root_player_idx)],
            policy_handler,
            add_noise,
            root_state: initial_state,
        }
    }

    /// Reconstruct the game state for a node by replaying its moves from the root.
    fn get_state(&self, node_idx: usize) -> GameState {
        // Walk up to the root, collecting moves in reverse order.
        let mut moves = Vec::new();
        let mut idx = node_idx;
        while let Some(parent) = self.tree[idx].parent {
            moves.push(self.tree[idx].applied_move.clone().unwrap());
            idx = parent;
        }
        // Replay from the root state.
        let mut state = self.root_state.clone();
        for mv in moves.into_iter().rev() {
            state.apply_move(&mv);
        }
        state
    }

    /// Rebuild the tree rooted at `current_game_state`.
    pub fn sync_tree_with_state(&mut self, current_game_state: &GameState) {
        *self = Mcts::new(current_game_state.clone(), self.policy_handler.clone(), self.add_noise);
    }

    /// Greedy best move (highest visit count). Used during evaluation.
    pub fn best_move(&self) -> Option<Move> {
        if self.tree.is_empty() {
            return None;
        }
        let root = &self.tree[0];
        root.children
            .iter()
            .max_by_key(|(_, child_idx)| self.tree[*child_idx].visit_count)
            .map(|(m, _)| m.clone())
    }

    /// Temperature-weighted move sampling. At `temperature == 0.0` this reduces to `best_move`.
    /// Used during self-play to produce diverse training data.
    pub fn sample_move(&self, temperature: f32) -> Option<Move> {
        if temperature == 0.0 {
            return self.best_move();
        }
        let root = &self.tree[0];
        if root.children.is_empty() {
            return None;
        }
        let weights: Vec<f32> = root
            .children
            .iter()
            .map(|(_, idx)| (self.tree[*idx].visit_count as f32).powf(1.0 / temperature))
            .collect();
        let total: f32 = weights.iter().sum();
        if total == 0.0 {
            return self.best_move();
        }
        let mut rng = rand::thread_rng();
        let threshold = rng.gen::<f32>() * total;
        let mut cumsum = 0.0;
        for ((mv, _), w) in root.children.iter().zip(weights.iter()) {
            cumsum += w;
            if cumsum >= threshold {
                return Some(mv.clone());
            }
        }
        root.children.last().map(|(mv, _)| mv.clone())
    }

    pub fn run_search(&mut self, iterations: u32) {
        for _ in 0..iterations {
            let leaf_idx = self.selection();
            let value = self.expansion(leaf_idx);
            self.backpropagation(leaf_idx, value);
        }
    }

    /// Traverse the tree picking the best PUCT action at each step.
    ///
    /// A node is a leaf when its `priors` are `None` (never evaluated) or empty
    /// (terminal state). Otherwise, `best_action` selects among both existing
    /// children *and* moves that have priors but no child node yet.  When an
    /// unexpanded move is chosen, its child node is created here (lazy expansion).
    fn selection(&mut self) -> usize {
        let mut current_idx = 0;
        loop {
            let is_leaf = self.tree[current_idx]
                .priors
                .as_ref()
                .map_or(true, |p| p.is_empty());

            if is_leaf {
                return current_idx;
            }

            let (best_move, child_idx_opt) = self.best_action(current_idx);

            if let Some(child_idx) = child_idx_opt {
                // Child node already exists — keep descending.
                current_idx = child_idx;
            } else {
                // Move has a prior but no node yet: create the child now.
                let prior = self.tree[current_idx]
                    .priors
                    .as_ref()
                    .unwrap()
                    .get(&best_move)
                    .copied()
                    .unwrap_or(0.0);
                // Derive child player index arithmetically — apply_move always
                // advances by (current + 1) % num_players (lib.rs).
                let num_players = self.root_state.players.len();
                let child_player_idx =
                    (self.tree[current_idx].current_player_idx + 1) % num_players;
                let child_node =
                    Node::new(Some(current_idx), prior, Some(best_move.clone()), child_player_idx);
                let child_idx = self.tree.len();
                self.tree.push(child_node);
                self.tree[current_idx].children.push((best_move, child_idx));
                return child_idx; // New node — always an unevaluated leaf.
            }
        }
    }

    /// Evaluate the leaf with the policy and store the resulting priors on the
    /// node. No child nodes are created — that is deferred to `selection`.
    fn expansion(&mut self, leaf_idx: usize) -> f32 {
        let leaf_state = self.get_state(leaf_idx);
        let (value, policy) = self.policy_handler.evaluate(&leaf_state);
        self.tree[leaf_idx].priors = Some(policy);

        // After the root's first evaluation, mix in Dirichlet noise.
        if leaf_idx == 0 && self.add_noise {
            self.apply_dirichlet_noise(0.3, 0.25);
        }

        value
    }

    /// Mix Dirichlet(alpha) noise into the root's stored priors.
    fn apply_dirichlet_noise(&mut self, alpha: f32, epsilon: f32) {
        let priors = match &self.tree[0].priors {
            Some(p) if !p.is_empty() => p,
            _ => return,
        };
        let n = priors.len();
        let dist = Gamma::new(alpha as f64, 1.0).expect("invalid gamma params");
        let mut rng = rand::thread_rng();
        let mut samples: Vec<f32> = (0..n).map(|_| dist.sample(&mut rng) as f32).collect();
        let sum: f32 = samples.iter().sum();
        if sum > 0.0 {
            samples.iter_mut().for_each(|x| *x /= sum);
        }
        // Collect keys first to avoid a simultaneous immutable + mutable borrow.
        let keys: Vec<Move> = priors.keys().cloned().collect();
        let priors_mut = self.tree[0].priors.as_mut().unwrap();
        for (mv, eta) in keys.iter().zip(samples.iter()) {
            if let Some(p) = priors_mut.get_mut(mv) {
                *p = (1.0 - epsilon) * *p + epsilon * eta;
            }
        }
    }

    fn backpropagation(&mut self, start_idx: usize, value: f32) {
        let player_at_leaf = self.tree[start_idx].current_player_idx;
        let mut current_idx = Some(start_idx);
        while let Some(idx) = current_idx {
            let node = &mut self.tree[idx];
            node.visit_count += 1;
            if node.current_player_idx == player_at_leaf {
                node.total_action_value += value;
            } else {
                node.total_action_value -= value;
            }
            current_idx = node.parent;
        }
    }

    /// PUCT over all priors (both expanded children and unexpanded moves).
    /// Returns `(best_move, Some(child_idx))` if already a node, else `(best_move, None)`.
    fn best_action(&self, node_idx: usize) -> (Move, Option<usize>) {
        let node = &self.tree[node_idx];
        let priors = node.priors.as_ref().unwrap();
        let exploration_constant = 1.41;
        let parent_visit_count = node.visit_count;

        // O(1) lookup: move → child index for already-created children.
        let child_map: HashMap<&Move, usize> =
            node.children.iter().map(|(m, idx)| (m, *idx)).collect();

        priors
            .iter()
            .map(|(mv, &prior)| {
                let (q, n) = if let Some(&child_idx) = child_map.get(mv) {
                    let child = &self.tree[child_idx];
                    (-child.mean_action_value(), child.visit_count)
                } else {
                    (0.0_f32, 0_u32)
                };
                let score = q
                    + exploration_constant * prior * (parent_visit_count as f32).sqrt()
                        / (1.0 + n as f32);
                (mv, child_map.get(mv).copied(), score)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(mv, child_idx, _)| (mv.clone(), child_idx))
            .unwrap() // Safe: called only when priors is non-empty (checked in selection).
    }
}
