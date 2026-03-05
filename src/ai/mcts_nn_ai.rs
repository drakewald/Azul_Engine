use crate::{
    ai::{
        mcts_lib::{Mcts, MctsPolicy},
        nn::{encode_state, move_to_policy_index, AzulNetwork, TacticianConfig, TacticianNet, POLICY_SIZE},
        AIAgent,
    },
    GameState, Move, MoveSource, Tile,
};
#[cfg(feature = "native")]
use crate::config::HardwareConfig;
use rustc_hash::{FxHashMap, FxHasher};
use std::any::Any;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

fn hash_vec_f32(v: &[f32]) -> u64 {
    let mut h = FxHasher::default();
    for &f in v { f.to_bits().hash(&mut h); }
    h.finish()
}

// Rc gives cheap Clone (ref-count increment) without requiring AzulNetwork: Sync.
// The transposition table is shared across clones (e.g. after sync_tree_with_state).
#[derive(Clone)]
struct NnPolicy {
    net: Rc<dyn AzulNetwork>,
    transposition_table: Rc<RefCell<FxHashMap<u64, (f32, Vec<f32>)>>>,
}

impl MctsPolicy for NnPolicy {
    fn evaluate(&self, game_state: &GameState) -> (f32, Vec<(Move, f32)>) {
        let input = self.state_to_input(game_state);
        let hash = hash_vec_f32(&input);
        let legal_moves = game_state.get_legal_moves();

        let (value, raw_policy) = {
            let table = self.transposition_table.borrow();
            if let Some(&(v, ref p)) = table.get(&hash) {
                (v, p.clone())
            } else {
                drop(table);
                let (p, v) = self.net.forward(&input);
                let mut table = self.transposition_table.borrow_mut();
                if table.len() >= 65_536 { table.clear(); }
                table.insert(hash, (v, p.clone()));
                (v, p)
            }
        };

        (value, self.mask_and_normalize_policy(&legal_moves, &raw_policy))
    }

    fn evaluate_batch(&self, states: &[&GameState]) -> Vec<(f32, Vec<(Move, f32)>)> {
        let mut results: Vec<Option<(f32, Vec<(Move, f32)>)>> = vec![None; states.len()];
        let mut uncached_indices: Vec<usize> = Vec::new();
        let mut uncached_inputs: Vec<Vec<f32>> = Vec::new();

        for (i, &state) in states.iter().enumerate() {
            let input = self.state_to_input(state);
            let hash = hash_vec_f32(&input);
            if let Some(&(v, ref p)) = self.transposition_table.borrow().get(&hash) {
                let legal = state.get_legal_moves();
                results[i] = Some((v, self.mask_and_normalize_policy(&legal, p)));
            } else {
                uncached_indices.push(i);
                uncached_inputs.push(input);
            }
        }

        if !uncached_inputs.is_empty() {
            let batch_results = self.net.forward_batch(&uncached_inputs);
            for (j, &state_i) in uncached_indices.iter().enumerate() {
                let (raw_policy, value) = &batch_results[j];
                let legal = states[state_i].get_legal_moves();
                let policy = self.mask_and_normalize_policy(&legal, raw_policy);
                {
                    let mut table = self.transposition_table.borrow_mut();
                    if table.len() >= 65_536 { table.clear(); }
                    let hash = hash_vec_f32(&uncached_inputs[j]);
                    table.insert(hash, (*value, raw_policy.clone()));
                }
                results[state_i] = Some((*value, policy));
            }
        }

        results.into_iter().map(|r| r.unwrap()).collect()
    }
}

impl NnPolicy {
    fn state_to_input(&self, game_state: &GameState) -> Vec<f32> {
        encode_state(game_state)
    }

    fn mask_and_normalize_policy(&self, legal_moves: &[Move], raw_policy: &[f32]) -> Vec<(Move, f32)> {
        // Convert raw logits to probabilities via numerically-stable softmax.
        let max_logit = raw_policy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = raw_policy.iter().map(|&x| (x - max_logit).exp()).sum();

        // Mask to legal (source, tile) pairs and sum their softmax probabilities.
        // Multiple destinations for the same (source, tile) share the same policy index,
        // so we deduplicate here to get a proper masked sum.
        let mut unique_probs: FxHashMap<(MoveSource, Tile), f32> = FxHashMap::default();
        let mut masked_sum = 0.0f32;
        for m in legal_moves {
            let key = (m.source.clone(), m.tile);
            if unique_probs.contains_key(&key) { continue; }
            if let Some(index) = move_to_policy_index(m.tile, &m.source) {
                let prob = (raw_policy.get(index).copied().unwrap_or(f32::NEG_INFINITY) - max_logit).exp() / exp_sum;
                unique_probs.insert(key, prob);
                masked_sum += prob;
            }
        }

        let mut result = Vec::with_capacity(legal_moves.len());
        if masked_sum > 0.0 {
            for m in legal_moves {
                if let Some(&prob) = unique_probs.get(&(m.source.clone(), m.tile)) {
                    result.push((m.clone(), prob / masked_sum));
                }
            }
        }
        if result.is_empty() && !legal_moves.is_empty() {
            let prob = 1.0 / legal_moves.len() as f32;
            for m in legal_moves { result.push((m.clone(), prob)); }
        }
        result
    }
}

pub struct MctsNnAI {
    mcts: Option<Mcts<NnPolicy>>,
    iterations: u32,
    model_path: Option<String>,
    model_bytes: Option<Vec<u8>>,
    /// Temperature for move selection (0.0 = greedy, 1.0 = proportional to visit counts).
    temperature: f32,
    /// Whether to inject Dirichlet noise at the root (should be true during self-play only).
    add_noise: bool,
    batch_size: usize,
}

impl MctsNnAI {
    pub fn new(iterations: u32, model_path: Option<String>, model_bytes: Option<Vec<u8>>) -> Self {
        #[cfg(feature = "native")]
        let batch_size = HardwareConfig::load().mcts_batch_size;
        #[cfg(not(feature = "native"))]
        let batch_size = 1usize;
        Self {
            mcts: None, iterations, model_path, model_bytes,
            temperature: 0.0, add_noise: false,
            batch_size,
        }
    }

    /// Switch to training mode: enables temperature sampling and Dirichlet root noise.
    pub fn set_training_mode(&mut self) {
        self.temperature = 1.0;
        self.add_noise = true;
    }

    pub fn get_mcts_policy(&self) -> Option<Vec<f32>> {
        if let Some(mcts) = &self.mcts {
            let root = &mcts.tree[0];
            if root.visit_count == 0 { return None; }
            let mut policy_vec = vec![0.0; POLICY_SIZE];
            for (mv, child_idx) in &root.children {
                if let Some(policy_idx) = move_to_policy_index(mv.tile, &mv.source) {
                    let child_visits = mcts.tree[*child_idx].visit_count;
                    policy_vec[policy_idx] = child_visits as f32 / root.visit_count as f32;
                }
            }
            return Some(policy_vec);
        }
        None
    }

    pub fn state_to_input(&self, game_state: &GameState) -> Option<Vec<f32>> {
        self.mcts.as_ref().map(|mcts| mcts.policy_handler.state_to_input(game_state))
    }
}

impl AIAgent for MctsNnAI {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move> {
        if self.mcts.is_none() {
            let net: Rc<dyn AzulNetwork> = if let Some(bytes) = &self.model_bytes {
                Rc::new(TacticianNet::load_from_bytes(bytes, TacticianConfig::default()).unwrap_or_else(|e| {
                    println!("Failed to load model from bytes: {}, using random weights.", e);
                    TacticianNet::new_random(TacticianConfig::default())
                }))
            } else if let Some(path) = &self.model_path {
                Rc::new(TacticianNet::load_from_path(path, TacticianConfig::default()).unwrap_or_else(|e| {
                    println!("Failed to load model from '{}': {}, using random weights.", path, e);
                    TacticianNet::new_random(TacticianConfig::default())
                }))
            } else {
                Rc::new(TacticianNet::new_random(TacticianConfig::default()))
            };

            let policy_handler = NnPolicy {
                net,
                transposition_table: Rc::new(RefCell::new(FxHashMap::default())),
            };
            self.mcts = Some(Mcts::new(game_state.clone(), policy_handler, self.add_noise, self.batch_size));
        }

        let mcts = self.mcts.as_mut().unwrap();
        mcts.sync_tree_with_state(game_state);
        mcts.run_search(self.iterations);
        mcts.sample_move(self.temperature)
    }

    fn as_any(&mut self) -> &mut dyn Any { self }
}
