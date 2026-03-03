#![cfg(feature = "native")]

use serde::{Deserialize, Serialize};
use std::fs;

const CONFIG_PATH: &str = "hardware_config.json";

/// Hardware-tunable runtime settings.
/// Edit hardware_config.json to change these without recompiling.
/// NN topology is defined in each species file (e.g. nn/tactician.rs), not here.
/// A missing or unparseable file silently falls back to defaults.
#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// MCTS leaf evaluations per batch. Use 1 on CPU (batching adds overhead).
    /// Increase for GPU where batching amortises inference cost.
    pub mcts_batch_size: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self { mcts_batch_size: 1 }
    }
}

impl HardwareConfig {
    pub fn load() -> Self {
        match fs::read_to_string(CONFIG_PATH) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                eprintln!("Warning: failed to parse {CONFIG_PATH}: {e}. Using defaults.");
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }
}
