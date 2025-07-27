use azul_engine::TrainingData;
use serde_json;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use tch::{nn, Device, Tensor, nn::OptimizerConfig};

// --- Network Architecture Constants ---
const NUM_FACTORIES: usize = 9;
const NUM_COLORS: usize = 5;
const MAX_CENTER_TILES: usize = 27;
const MAX_PLAYERS: usize = 4;
const PATTERN_LINE_SLOTS: usize = 5 * 5;
const WALL_SLOTS: usize = 5 * 5;
const FLOOR_SLOTS: usize = 7;

const INPUT_SIZE: usize = (NUM_FACTORIES * NUM_COLORS * 4)
                        + (MAX_CENTER_TILES * NUM_COLORS)
                        + (MAX_PLAYERS * (1 + PATTERN_LINE_SLOTS + WALL_SLOTS + FLOOR_SLOTS + 1))
                        + 1;
const POLICY_SIZE: usize = (NUM_FACTORIES * NUM_COLORS) + NUM_COLORS;


#[derive(Debug)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Self {
        let hidden_size = 256;
        let fc1 = nn::linear(vs / "fc1", INPUT_SIZE as i64, hidden_size, Default::default());
        let fc2 = nn::linear(vs / "fc2", hidden_size, hidden_size, Default::default());
        let policy_head = nn::linear(vs / "policy_head", hidden_size, POLICY_SIZE as i64, Default::default());
        let value_head = nn::linear(vs / "value_head", hidden_size, 1, Default::default());
        Self { fc1, fc2, policy_head, value_head }
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor) {
        let xs = xs.apply(&self.fc1).relu().apply(&self.fc2).relu();
        let policy = xs.apply(&self.policy_head);
        let value = xs.apply(&self.value_head).tanh();
        (policy, value)
    }
}

fn main() -> anyhow::Result<()> {
    // --- 1. Load Data ---
    let data_dir = "training_data";
    fs::create_dir_all(data_dir)?;

    let latest_data_file = fs::read_dir(data_dir)?
        .filter_map(Result::ok)
        .max_by_key(|entry| entry.metadata().unwrap().created().unwrap());

    let data: Vec<TrainingData> = if let Some(entry) = latest_data_file {
        let path = entry.path();
        println!("Loading latest data file: {:?}", path);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)?
    } else {
        Vec::new()
    };
    
    println!("Loaded {} training samples.", data.len());

    if data.is_empty() {
        println!("No training data found. Run headless in --self-play mode to generate data.");
        return Ok(());
    }

    // --- 2. Set up Model and Optimizer ---
    let mut vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&vs.root());

    // --- MODIFIED SECTION: Fine-tuning Logic ---
    let training_models_dir = "training_models";
    fs::create_dir_all(training_models_dir)?;

    let latest_model = fs::read_dir(training_models_dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "ot"))
        .max_by_key(|entry| entry.metadata().unwrap().created().unwrap());

    let mut next_version = 1;
    if let Some(entry) = latest_model {
        let path = entry.path();
        println!("Loading model for fine-tuning: {:?}", path);
        vs.load(&path)?;

        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if let Some(version_str) = stem.strip_prefix("azul_model_v") {
                if let Ok(version) = version_str.parse::<u32>() {
                    next_version = version + 1;
                }
            }
        }
    } else {
        println!("No existing model found in 'training_models/'. Training a new model (v1) from scratch.");
    }
    // --- END MODIFIED SECTION ---

    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;

    // --- 3. Training Loop ---
    let epochs = 10;
    let batch_size = 64;
    println!("Starting training for {} epochs...", epochs);

    for epoch in 1..=epochs {
        // In a real implementation, you would shuffle the data here.
        for batch_start in (0..data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(data.len());
            if batch_start >= batch_end { continue; }
            let batch = &data[batch_start..batch_end];

            let states: Vec<Tensor> = batch.iter().map(|d| Tensor::from_slice(&d.state_input)).collect();
            let policies: Vec<Tensor> = batch.iter().map(|d| Tensor::from_slice(&d.mcts_policy)).collect();
            let outcomes: Vec<Tensor> = batch.iter().map(|d| Tensor::from_slice(&[d.outcome])).collect();

            let state_tensor = Tensor::stack(&states, 0).to_device(vs.device());
            let policy_tensor = Tensor::stack(&policies, 0).to_device(vs.device());
            let outcome_tensor = Tensor::stack(&outcomes, 0).to_device(vs.device());

            let (policy_logits, value_pred) = net.forward(&state_tensor);

            let value_loss = value_pred.mse_loss(&outcome_tensor, tch::Reduction::Mean);
            let policy_loss = policy_logits.mse_loss(&policy_tensor, tch::Reduction::Mean);
            let total_loss = value_loss + policy_loss;

            opt.zero_grad();
            total_loss.backward();
            opt.step();
        }
        println!("Epoch {} complete.", epoch);
    }

    // --- 4. Save Model ---
    let release_models_dir = "release_models";
    fs::create_dir_all(release_models_dir)?;

    let new_training_model_path = format!("{}/azul_model_v{}.ot", training_models_dir, next_version);
    let release_model_path = format!("{}/azul_alpha.ot", release_models_dir);

    // Save the new versioned model for continued training.
    vs.save(&new_training_model_path)?;
    println!("Training complete. New version saved to '{}'", new_training_model_path);

    // Save a copy to the release directory for the web app.
    vs.save(&release_model_path)?;
    println!("Model deployed for release to '{}'", release_model_path);

    Ok(())
}
