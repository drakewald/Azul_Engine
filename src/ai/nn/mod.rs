use candle_core::Tensor;
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

// Encoding constants shared across all architectures.
// These describe the state/action representation, not any particular network topology.
pub const INPUT_SIZE: usize = (9 * 5 * 4) + (27 * 5) + (4 * (1 + 25 + 25 + 7 + 1)) + 1;
pub const POLICY_SIZE: usize = (9 * 5) + 5;

pub mod tactician;
pub use tactician::{TacticianConfig, TacticianNet};

/// Inference interface shared by all Azul network architectures.
/// Object-safe — usable as `Rc<dyn AzulNetwork>` inside MCTS.
/// Training-specific concerns (VarMap, save, backward) live on concrete types only.
pub trait AzulNetwork {
    fn forward(&self, input: &[f32]) -> (Vec<f32>, f32);
    fn forward_batch(&self, inputs: &[Vec<f32>]) -> Vec<(Vec<f32>, f32)>;
}

/// A residual block: FC → LN → ReLU → FC → LN, with a skip connection and final ReLU.
/// Shared building block available to any species in this module.
pub(super) struct ResBlock {
    fc1: Linear,
    ln1: LayerNorm,
    fc2: Linear,
    ln2: LayerNorm,
}

impl ResBlock {
    pub(super) fn new(width: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(width, width, vb.pp("fc1"))?,
            ln1: layer_norm(width, 1e-5, vb.pp("ln1"))?,
            fc2: linear(width, width, vb.pp("fc2"))?,
            ln2: layer_norm(width, 1e-5, vb.pp("ln2"))?,
        })
    }

    pub(super) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.fc1.forward(x)?.apply(&self.ln1)?.relu()?;
        let h = self.fc2.forward(&h)?.apply(&self.ln2)?;
        (x + h)?.relu()
    }
}
