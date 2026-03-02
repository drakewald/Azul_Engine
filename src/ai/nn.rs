// This entire module will only be compiled when the "native" feature is enabled.
#![cfg(feature = "native")]

use anyhow;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};

// Architecture constants shared with mcts_nn_ai.rs and train.rs.
pub const INPUT_SIZE: usize = (9 * 5 * 4) + (27 * 5) + (4 * (1 + 25 + 25 + 7 + 1)) + 1;
pub const POLICY_SIZE: usize = (9 * 5) + 5;

const HIDDEN_SIZE: usize = 256;

/// A candle-based network matching the architecture saved by train.rs:
/// two ReLU hidden layers with separate policy and value heads.
pub struct AzulNet {
    fc1: Linear,
    fc2: Linear,
    policy_head: Linear,
    value_head: Linear,
    device: Device,
}

impl AzulNet {
    fn build(vb: VarBuilder) -> candle_core::Result<Self> {
        let device = vb.device().clone();
        let fc1 = linear(INPUT_SIZE, HIDDEN_SIZE, vb.pp("fc1"))?;
        let fc2 = linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("fc2"))?;
        let policy_head = linear(HIDDEN_SIZE, POLICY_SIZE, vb.pp("policy_head"))?;
        let value_head = linear(HIDDEN_SIZE, 1, vb.pp("value_head"))?;
        Ok(Self { fc1, fc2, policy_head, value_head, device })
    }

    pub fn new_random() -> Self {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::build(vb).expect("AzulNet build failed")
    }

    /// Load weights from a `.safetensors` file saved by the train binary.
    pub fn load_from_path(path: &str) -> Result<Self, anyhow::Error> {
        let device = Device::Cpu;
        // Pre-populate VarMap from the saved file, then build layers on top of it.
        // VarBuilder::from_varmap will reuse existing variables rather than creating new ones.
        let mut varmap = VarMap::new();
        varmap.load(path)?;
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Ok(Self::build(vb)?)
    }

    /// Load weights from raw bytes (e.g. bundled model data).
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, anyhow::Error> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(bytes.to_vec(), DType::F32, &device)?;
        Ok(Self::build(vb)?)
    }

    /// Run a forward pass and return (policy_logits, value).
    /// policy_logits has POLICY_SIZE elements; value is in [-1, 1].
    pub fn forward(&self, input: &[f32]) -> (Vec<f32>, f32) {
        let xs = Tensor::from_slice(input, (1, INPUT_SIZE), &self.device)
            .expect("failed to create input tensor");
        let h = self.fc1.forward(&xs).and_then(|t| t.relu())
            .and_then(|t| self.fc2.forward(&t))
            .and_then(|t| t.relu())
            .expect("hidden layers failed");
        let policy = self.policy_head.forward(&h)
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_vec1::<f32>())
            .expect("policy head failed");
        let value = self.value_head.forward(&h)
            .and_then(|t| t.tanh())
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_scalar::<f32>())
            .expect("value head failed");
        (policy, value)
    }
}
