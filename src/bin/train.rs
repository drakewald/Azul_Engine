use azul_engine::ai::{mcts_nn_ai::MctsNnAI, AIAgent};
use azul_engine::ai::nn::{TacticianConfig, TacticianNet, INPUT_SIZE, POLICY_SIZE};
use azul_engine::TrainingData;
use azul_engine::GameState;
use candle_core::{Device, Tensor};
use candle_nn::{loss, ops, Optimizer};
use candle_nn::optim::{AdamW, ParamsAdamW};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde_json;
use std::fs;
use std::fs::File;
use std::io::BufReader;

/// Number of most-recent data files included in each training run.
const REPLAY_BUFFER_FILES: usize = 5;
/// Weight applied to policy loss relative to value loss.
const POLICY_LOSS_WEIGHT: f64 = 1.0;
/// Evaluation games run between new and old model after training.
const EVAL_GAMES: u32 = 400;
/// Minimum new-model win rate required to deploy to release_models/.
const WIN_RATE_THRESHOLD: f64 = 0.52;
/// MCTS iterations used during evaluation games.
const EVAL_MCTS_ITERS: u32 = 200;

/// Determine the winner using score, then complete rows as a tiebreaker.
/// Returns `None` on a genuine tie (both metrics equal), matching `headless.rs`.
fn determine_winner(game: &GameState) -> Option<usize> {
    let best = game.players.iter().enumerate().max_by(|(_, a), (_, b)| {
        a.score.cmp(&b.score).then(a.count_complete_rows().cmp(&b.count_complete_rows()))
    });
    let (winner_idx, winner) = best?;
    let is_tie = game.players.iter().any(|p| {
        p != winner
            && p.score == winner.score
            && p.count_complete_rows() == winner.count_complete_rows()
    });
    if is_tie { None } else { Some(winner_idx) }
}

/// Play one 2-player game between `new_path` and `old_path` models.
/// Returns 1.0 (new wins), 0.5 (tie), or 0.0 (new loses).
fn run_eval_game(new_path: &str, old_path: &str, new_goes_first: bool) -> f32 {
    let (p0_path, p1_path) = if new_goes_first {
        (new_path, old_path)
    } else {
        (old_path, new_path)
    };
    let mut agents: Vec<Box<dyn AIAgent>> = vec![
        Box::new(MctsNnAI::new(EVAL_MCTS_ITERS, Some(p0_path.to_string()), None)),
        Box::new(MctsNnAI::new(EVAL_MCTS_ITERS, Some(p1_path.to_string()), None)),
    ];
    let mut game = GameState::new(2);
    while !game.end_game_triggered {
        while !game.is_round_over() {
            let idx = game.current_player_idx;
            if let Some(mv) = agents[idx].get_move(&game) {
                game.apply_move(&mv);
            } else {
                break;
            }
        }
        game.run_tiling_phase();
        if !game.end_game_triggered {
            game.refill_factories();
        }
    }
    game.apply_end_game_scoring();
    let new_player_idx = if new_goes_first { 0 } else { 1 };
    match determine_winner(&game) {
        Some(w) if w == new_player_idx => 1.0,
        None => 0.5,
        _ => 0.0,
    }
}

fn main() -> anyhow::Result<()> {
    // --- 1. Load Data (rolling replay buffer: last N files) ---
    let data_dir = "training_data";
    fs::create_dir_all(data_dir)?;

    let mut data_files: Vec<_> = fs::read_dir(data_dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
        .collect();
    data_files.sort_by_key(|e| e.metadata().unwrap().modified().unwrap());

    let mut data: Vec<TrainingData> = data_files
        .iter()
        .rev()
        .take(REPLAY_BUFFER_FILES)
        .flat_map(|entry| {
            println!("Loading data file: {:?}", entry.path());
            let file = File::open(entry.path()).expect("failed to open data file");
            let reader = BufReader::new(file);
            let samples: Vec<TrainingData> =
                serde_json::from_reader(reader).unwrap_or_default();
            samples
        })
        .collect();

    println!(
        "Loaded {} training samples from up to {} file(s).",
        data.len(),
        REPLAY_BUFFER_FILES
    );

    if data.is_empty() {
        println!("No training data found. Run headless in --self-play mode to generate data.");
        return Ok(());
    }

    // Shuffle to break temporal correlation between batches.
    data.shuffle(&mut rand::thread_rng());

    // --- 2. Set up Model ---
    let device = Device::Cpu;

    let training_models_dir = "training_models";
    fs::create_dir_all(training_models_dir)?;

    let latest_model = fs::read_dir(training_models_dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
        .max_by_key(|entry| entry.metadata().unwrap().modified().unwrap());

    let mut next_version = 1;
    let mut old_model_path: Option<String> = None;

    if let Some(entry) = latest_model {
        let path = entry.path();
        println!("Loading model for fine-tuning: {:?}", path);
        old_model_path = Some(path.to_string_lossy().to_string());

        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if let Some(version_str) = stem.strip_prefix("azul_model_v") {
                if let Ok(version) = version_str.parse::<u32>() {
                    next_version = version + 1;
                }
            }
        }
    } else {
        println!(
            "No existing model found in 'training_models/'. Training a new model (v1) from scratch."
        );
    }

    let net = if let Some(ref path) = old_model_path {
        TacticianNet::load_from_path(path, TacticianConfig::default())?
    } else {
        TacticianNet::new_random(TacticianConfig::default())
    };

    let mut opt = AdamW::new(
        net.trainable_vars(),
        ParamsAdamW { lr: 1e-4, weight_decay: 0.0, ..Default::default() },
    )?;

    // --- 3. Training Loop ---
    let epochs = 10;
    let batch_size = 64;
    println!("Starting training for {} epochs...", epochs);

    for epoch in 1..=epochs {
        for batch_start in (0..data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(data.len());
            if batch_start >= batch_end {
                continue;
            }
            let batch = &data[batch_start..batch_end];
            let n = batch.len();

            let mut state_flat = Vec::with_capacity(n * INPUT_SIZE);
            let mut policy_flat = Vec::with_capacity(n * POLICY_SIZE);
            let mut outcome_flat = Vec::with_capacity(n);
            for d in batch {
                state_flat.extend_from_slice(&d.state_input);
                policy_flat.extend_from_slice(&d.mcts_policy);
                outcome_flat.push(d.outcome);
            }

            let state_t = Tensor::from_slice(&state_flat, (n, INPUT_SIZE), &device)?;
            let policy_t = Tensor::from_slice(&policy_flat, (n, POLICY_SIZE), &device)?;
            let outcome_t = Tensor::from_slice(&outcome_flat, (n,), &device)?;

            let (policy_logits, value_pred) = net.forward_train(&state_t)?;

            let value_loss = loss::mse(&value_pred, &outcome_t)?;

            // Soft cross-entropy: -sum(target * log_softmax(logits)) / batch
            let log_probs = ops::log_softmax(&policy_logits, 1)?;
            let policy_loss = policy_t.mul(&log_probs)?.mean_all()?.neg()?;

            let total_loss = (value_loss + (policy_loss * POLICY_LOSS_WEIGHT)?)?;

            opt.backward_step(&total_loss)?;
        }
        println!("Epoch {} complete.", epoch);
    }

    // --- 4. Save New Model ---
    let release_models_dir = "release_models";
    fs::create_dir_all(release_models_dir)?;

    let new_training_model_path = format!(
        "{}/azul_model_v{}.safetensors",
        training_models_dir, next_version
    );
    net.save(&new_training_model_path)?;
    println!("Saved new model to '{}'", new_training_model_path);

    // --- 5. Evaluate and Gate Deployment ---
    let release_model_path = format!("{}/azul_alpha.safetensors", release_models_dir);

    let deploy = if let Some(ref old_path) = old_model_path {
        println!(
            "\nEvaluating new model vs previous ({} games, {} MCTS iters)...",
            EVAL_GAMES, EVAL_MCTS_ITERS
        );
        let new_path = new_training_model_path.clone();
        let old_path = old_path.clone();
        let (wins, ties): (u32, u32) = (0..EVAL_GAMES)
            .into_par_iter()
            .map(|i| {
                let r = run_eval_game(&new_path, &old_path, i % 2 == 0);
                (if r == 1.0 { 1u32 } else { 0 }, if r == 0.5 { 1u32 } else { 0 })
            })
            .reduce(|| (0, 0), |(w1, t1), (w2, t2)| (w1 + w2, t1 + t2));
        let losses = EVAL_GAMES - wins - ties;
        let win_rate = (wins as f64 + ties as f64 * 0.5) / EVAL_GAMES as f64;
        println!(
            "New model: {} W / {} D / {} L — {:.1}% adjusted win rate (threshold {:.0}%)",
            wins, ties, losses, win_rate * 100.0, WIN_RATE_THRESHOLD * 100.0
        );
        win_rate >= WIN_RATE_THRESHOLD
    } else {
        true // First model ever — always deploy.
    };

    if deploy {
        net.save(&release_model_path)?;
        println!("Model deployed for release to '{}'", release_model_path);
    } else {
        println!(
            "New model did not reach the {:.0}% win-rate threshold. Not deployed.",
            WIN_RATE_THRESHOLD * 100.0
        );
    }

    Ok(())
}
