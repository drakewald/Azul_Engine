use crate::{GameState, Move};
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use std::cmp::Ordering::Equal;

#[cfg(feature = "native")]
use rustc_hash::FxHashMap;
#[cfg(not(feature = "native"))]
use std::collections::HashMap as FxHashMap;

#[cfg(feature = "native")]
use smallvec::SmallVec;

#[cfg(feature = "native")]
type ChildVec = SmallVec<[(Move, usize); 8]>;
#[cfg(not(feature = "native"))]
type ChildVec = Vec<(Move, usize)>;

pub trait MctsPolicy: Clone {
    fn evaluate(&self, game_state: &GameState) -> (f32, Vec<(Move, f32)>);

    fn evaluate_batch(&self, states: &[&GameState]) -> Vec<(f32, Vec<(Move, f32)>)> {
        states.iter().map(|s| self.evaluate(s)).collect()
    }
}

pub struct Node {
    pub parent: Option<usize>,
    /// Child nodes that have already been created (visited at least once).
    pub children: ChildVec,
    /// Prior probabilities returned by the policy network, sorted descending by value.
    /// `None`  — node has never been evaluated (it is an unvisited leaf).
    /// `Some(vec)` — evaluated; covers all legal moves including unexpanded ones.
    pub priors: Option<Vec<(Move, f32)>>,
    pub visit_count: u32,
    pub total_action_value: f32,
    pub prior_probability: f32,
    /// The move that was applied to reach this node from its parent.
    /// `None` for the root node.
    pub applied_move: Option<Move>,
    /// The player whose turn it is at this node.
    pub current_player_idx: usize,
    /// Cached game state at this node — eliminates O(depth) replay each iteration.
    pub state: GameState,
    /// Virtual loss applied during batch selection to discourage re-selection.
    pub virtual_visits: u32,
    pub virtual_loss: f32,
}

impl Node {
    fn new(
        parent: Option<usize>,
        prior: f32,
        applied_move: Option<Move>,
        current_player_idx: usize,
        state: GameState,
    ) -> Self {
        Self {
            parent,
            children: ChildVec::new(),
            priors: None,
            visit_count: 0,
            total_action_value: 0.0,
            prior_probability: prior,
            applied_move,
            current_player_idx,
            state,
            virtual_visits: 0,
            virtual_loss: 0.0,
        }
    }

    #[inline]
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
    pub batch_size: usize,
}

impl<P: MctsPolicy + Clone> Mcts<P> {
    pub fn new(initial_state: GameState, policy_handler: P, add_noise: bool, batch_size: usize) -> Self {
        let root_player_idx = initial_state.current_player_idx;
        let root_node = Node::new(None, 1.0, None, root_player_idx, initial_state);
        Self {
            tree: vec![root_node],
            policy_handler,
            add_noise,
            batch_size,
        }
    }

    /// Rebuild the tree rooted at `current_game_state`.
    pub fn sync_tree_with_state(&mut self, current_game_state: &GameState) {
        *self = Mcts::new(current_game_state.clone(), self.policy_handler.clone(), self.add_noise, self.batch_size);
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
        let mut done = 0u32;
        while done < iterations {
            let batch = (self.batch_size as u32).min(iterations - done) as usize;
            let mut leaves = Vec::with_capacity(batch);
            for _ in 0..batch {
                let leaf = self.selection();
                self.apply_virtual_loss(leaf);
                leaves.push(leaf);
            }
            // Clone states into owned values so we can release the tree borrow.
            let states: Vec<GameState> = leaves.iter()
                .map(|&i| self.tree[i].state.clone())
                .collect();
            let state_refs: Vec<&GameState> = states.iter().collect();
            let evals = self.policy_handler.evaluate_batch(&state_refs);
            for (&leaf_idx, (value, mut policy)) in leaves.iter().zip(evals) {
                self.remove_virtual_loss(leaf_idx);
                if self.tree[leaf_idx].priors.is_none() {
                    policy.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Equal));
                    self.tree[leaf_idx].priors = Some(policy);
                    if leaf_idx == 0 && self.add_noise {
                        self.apply_dirichlet_noise(0.3, 0.25);
                    }
                }
                self.backpropagation(leaf_idx, value);
            }
            done += batch as u32;
        }
    }

    fn apply_virtual_loss(&mut self, leaf_idx: usize) {
        let mut idx = Some(leaf_idx);
        while let Some(i) = idx {
            self.tree[i].virtual_visits += 1;
            self.tree[i].virtual_loss += 1.0;
            idx = self.tree[i].parent;
        }
    }

    fn remove_virtual_loss(&mut self, leaf_idx: usize) {
        let mut idx = Some(leaf_idx);
        while let Some(i) = idx {
            self.tree[i].virtual_visits -= 1;
            self.tree[i].virtual_loss -= 1.0;
            idx = self.tree[i].parent;
        }
    }

    /// Traverse the tree picking the best PUCT action at each step.
    ///
    /// A node is a leaf when its `priors` are `None` (never evaluated) or empty
    /// (terminal state). Otherwise, `best_action` selects among both existing
    /// children *and* moves that have priors but no child node yet. When an
    /// unexpanded move is chosen, its child node is created here (lazy expansion).
    #[inline]
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
                // Clone the parent state, apply the move, derive child state.
                let child_state = {
                    let mut s = self.tree[current_idx].state.clone();
                    s.apply_move(&best_move);
                    s
                };
                let child_player_idx = child_state.current_player_idx;
                let prior = self.tree[current_idx]
                    .priors
                    .as_ref()
                    .unwrap()
                    .iter()
                    .find(|(m, _)| m == &best_move)
                    .map(|(_, p)| *p)
                    .unwrap_or(0.0);
                let child_node = Node::new(
                    Some(current_idx),
                    prior,
                    Some(best_move.clone()),
                    child_player_idx,
                    child_state,
                );
                let child_idx = self.tree.len();
                self.tree.push(child_node);
                self.tree[current_idx].children.push((best_move, child_idx));
                return child_idx; // New node — always an unevaluated leaf.
            }
        }
    }

    /// Mix Dirichlet(alpha) noise into the root's stored priors, then re-sort.
    fn apply_dirichlet_noise(&mut self, alpha: f32, epsilon: f32) {
        let n = match &self.tree[0].priors {
            Some(p) if !p.is_empty() => p.len(),
            _ => return,
        };
        let dist = Gamma::new(alpha as f64, 1.0).expect("invalid gamma params");
        let mut rng = rand::thread_rng();
        let mut samples: Vec<f32> = (0..n).map(|_| dist.sample(&mut rng) as f32).collect();
        let sum: f32 = samples.iter().sum();
        if sum > 0.0 {
            samples.iter_mut().for_each(|x| *x /= sum);
        }
        let priors_mut = self.tree[0].priors.as_mut().unwrap();
        for (i, eta) in samples.iter().enumerate() {
            priors_mut[i].1 = (1.0 - epsilon) * priors_mut[i].1 + epsilon * eta;
        }
        priors_mut.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Equal));
    }

    #[inline]
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

    /// PUCT over priors with progressive widening and virtual loss.
    /// Returns `(best_move, Some(child_idx))` if already a node, else `(best_move, None)`.
    #[inline]
    fn best_action(&self, node_idx: usize) -> (Move, Option<usize>) {
        let node = &self.tree[node_idx];
        let priors = node.priors.as_ref().unwrap();
        let exploration_constant = 1.41;
        let eff_visits = node.visit_count + node.virtual_visits;

        // Progressive widening: expand ceil(2 * sqrt(visits)) candidates, min 1.
        let max_candidates = ((2.0 * (eff_visits as f32).sqrt()).ceil() as usize)
            .max(node.children.len())
            .max(1)
            .min(priors.len());

        // FxHashMap: move → child_idx for already-expanded children.
        let child_map: FxHashMap<&Move, usize> =
            node.children.iter().map(|(m, idx)| (m, *idx)).collect();

        priors[..max_candidates]
            .iter()
            .map(|(mv, prior)| {
                let (q, n) = if let Some(&child_idx) = child_map.get(mv) {
                    let child = &self.tree[child_idx];
                    let eff_n = child.visit_count + child.virtual_visits;
                    let eff_q = if eff_n == 0 {
                        0.0
                    } else {
                        -(child.total_action_value - child.virtual_loss) / eff_n as f32
                    };
                    (eff_q, eff_n)
                } else {
                    (0.0_f32, 0_u32)
                };
                let score: f32 = q + exploration_constant * prior * (eff_visits as f32).sqrt()
                    / (1.0 + n as f32);
                (mv, child_map.get(mv).copied(), score)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Equal))
            .map(|(mv, child_idx, _): (&Move, Option<usize>, f32)| (mv.clone(), child_idx))
            .unwrap() // Safe: called only when priors is non-empty (checked in selection).
    }
}
