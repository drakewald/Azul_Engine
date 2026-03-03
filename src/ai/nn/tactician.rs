use anyhow;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{linear, LayerNorm, Linear, VarBuilder, VarMap};

use super::{AzulNetwork, ResBlock, INPUT_SIZE, POLICY_SIZE};

#[derive(Debug, Clone)]
pub struct TacticianConfig {
    pub input_dim: usize,
    pub trunk_width: usize,
    pub num_res_blocks: usize,
    pub mid_dim: usize,
    pub value_hidden_dim: usize,
    pub policy_dim: usize,
}

impl Default for TacticianConfig {
    fn default() -> Self {
        Self {
            input_dim: INPUT_SIZE,
            trunk_width: 256,
            num_res_blocks: 2,
            mid_dim: 128,
            value_hidden_dim: 64,
            policy_dim: POLICY_SIZE,
        }
    }
}

pub struct TacticianNet {
    cfg: TacticianConfig,
    stem: Linear,
    stem_ln: LayerNorm,
    blocks: Vec<ResBlock>,
    bottleneck: Linear,
    policy_head: Linear,
    value_hidden: Linear,
    value_head: Linear,
    // Some when built from VarMap (trainable/saveable); None when loaded from bytes.
    varmap: Option<VarMap>,
    device: Device,
}

type Layers = (Linear, LayerNorm, Vec<ResBlock>, Linear, Linear, Linear, Linear);

impl TacticianNet {
    fn build(cfg: &TacticianConfig, vb: VarBuilder) -> candle_core::Result<Layers> {
        use candle_nn::layer_norm;
        let stem = linear(cfg.input_dim, cfg.trunk_width, vb.pp("stem"))?;
        let stem_ln = layer_norm(cfg.trunk_width, 1e-5, vb.pp("stem_ln"))?;
        let blocks = (0..cfg.num_res_blocks)
            .map(|i| ResBlock::new(cfg.trunk_width, vb.pp(format!("block{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let bottleneck = linear(cfg.trunk_width, cfg.mid_dim, vb.pp("bottleneck"))?;
        let policy_head = linear(cfg.mid_dim, cfg.policy_dim, vb.pp("policy_head"))?;
        let value_hidden = linear(cfg.mid_dim, cfg.value_hidden_dim, vb.pp("value_hidden"))?;
        let value_head = linear(cfg.value_hidden_dim, 1, vb.pp("value_head"))?;
        Ok((stem, stem_ln, blocks, bottleneck, policy_head, value_hidden, value_head))
    }

    fn from_parts(cfg: TacticianConfig, layers: Layers, varmap: Option<VarMap>, device: Device) -> Self {
        let (stem, stem_ln, blocks, bottleneck, policy_head, value_hidden, value_head) = layers;
        Self { cfg, stem, stem_ln, blocks, bottleneck, policy_head, value_hidden, value_head, varmap, device }
    }

    pub fn new_random(cfg: TacticianConfig) -> Self {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let layers = Self::build(&cfg, vb).expect("TacticianNet build failed");
        Self::from_parts(cfg, layers, Some(varmap), device)
    }

    pub fn load_from_path(path: &str, cfg: TacticianConfig) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        varmap.load(path)?;
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let layers = Self::build(&cfg, vb)?;
        Ok(Self::from_parts(cfg, layers, Some(varmap), device))
    }

    pub fn load_from_bytes(bytes: &[u8], cfg: TacticianConfig) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(bytes.to_vec(), DType::F32, &device)?;
        let layers = Self::build(&cfg, vb)?;
        Ok(Self::from_parts(cfg, layers, None, device))
    }

    fn run_forward(&self, xs: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        use candle_nn::Module;
        let h = self.stem.forward(xs)?.apply(&self.stem_ln)?.relu()?;
        let h = self.blocks.iter().try_fold(h, |acc, block| block.forward(&acc))?;
        let h = self.bottleneck.forward(&h)?.relu()?;
        let policy = self.policy_head.forward(&h)?;
        // Squeeze the trailing 1 so value shape is (batch,) not (batch, 1).
        // forward() can then squeeze(0) → scalar; forward_batch() can to_vec1() directly.
        let value = self.value_hidden.forward(&h)?.relu()
            .and_then(|t| self.value_head.forward(&t))
            .and_then(|t| t.tanh())
            .and_then(|t| t.squeeze(1))?;
        Ok((policy, value))
    }

    /// Training forward pass — returns raw tensors for loss computation.
    pub fn forward_train(&self, xs: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        self.run_forward(xs)
    }

    /// All trainable variables for optimizer construction.
    pub fn trainable_vars(&self) -> Vec<Var> {
        self.varmap.as_ref().map_or_else(Vec::new, VarMap::all_vars)
    }

    /// Save weights to a safetensors file.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        self.varmap
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no VarMap — model was loaded from bytes"))?
            .save(path)?;
        Ok(())
    }
}

impl AzulNetwork for TacticianNet {
    fn forward(&self, input: &[f32]) -> (Vec<f32>, f32) {
        let xs = Tensor::from_slice(input, (1, self.cfg.input_dim), &self.device)
            .expect("input tensor");
        let (policy_t, value_t) = self.run_forward(&xs).expect("forward failed");
        let policy = policy_t.squeeze(0).and_then(|t| t.to_vec1::<f32>()).expect("policy vec");
        let value = value_t.squeeze(0).and_then(|t| t.to_scalar::<f32>()).expect("value scalar");
        (policy, value)
    }

    fn forward_batch(&self, inputs: &[Vec<f32>]) -> Vec<(Vec<f32>, f32)> {
        let n = inputs.len();
        let flat: Vec<f32> = inputs.iter().flat_map(|v| v.iter().copied()).collect();
        let xs = Tensor::from_slice(&flat, (n, self.cfg.input_dim), &self.device)
            .expect("batch input tensor");
        let (policies_t, values_t) = self.run_forward(&xs).expect("batch forward failed");
        let policies = policies_t.to_vec2::<f32>().expect("policies");
        let values = values_t.to_vec1::<f32>().expect("values");
        policies.into_iter().zip(values).collect()
    }
}
