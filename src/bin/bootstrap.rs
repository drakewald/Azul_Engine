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

const POLICY_LOSS_WEIGHT: f64 = 1.0;
const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 64;
const TOURNAMENT_GAMES: u32 = 400;
const TOURNAMENT_MCTS_ITERS: u32 = 600;

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

fn run_game(path_a: &str, path_b: &str, a_goes_first: bool) -> f32 {
    let (p0_path, p1_path) = if a_goes_first { (path_a, path_b) } else { (path_b, path_a) };
    let mut agents: Vec<Box<dyn AIAgent>> = vec![
        Box::new(MctsNnAI::new(TOURNAMENT_MCTS_ITERS, Some(p0_path.to_string()), None)),
        Box::new(MctsNnAI::new(TOURNAMENT_MCTS_ITERS, Some(p1_path.to_string()), None)),
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
    let a_idx = if a_goes_first { 0 } else { 1 };
    match determine_winner(&game) {
        Some(w) if w == a_idx => 1.0,
        None => 0.5,
        _ => 0.0,
    }
}

fn train_net(net: &TacticianNet, data: &[TrainingData], device: &Device, label: &str) -> anyhow::Result<()> {
    let mut opt = AdamW::new(
        net.trainable_vars(),
        ParamsAdamW { lr: 1e-4, weight_decay: 0.0, ..Default::default() },
    )?;
    for epoch in 1..=EPOCHS {
        for batch_start in (0..data.len()).step_by(BATCH_SIZE) {
            let batch = &data[batch_start..(batch_start + BATCH_SIZE).min(data.len())];
            let n = batch.len();
            let mut state_flat = Vec::with_capacity(n * INPUT_SIZE);
            let mut policy_flat = Vec::with_capacity(n * POLICY_SIZE);
            let mut outcome_flat = Vec::with_capacity(n);
            for d in batch {
                state_flat.extend_from_slice(&d.state_input);
                policy_flat.extend_from_slice(&d.mcts_policy);
                outcome_flat.push(d.outcome);
            }
            let state_t = Tensor::from_slice(&state_flat, (n, INPUT_SIZE), device)?;
            let policy_t = Tensor::from_slice(&policy_flat, (n, POLICY_SIZE), device)?;
            let outcome_t = Tensor::from_slice(&outcome_flat, (n,), device)?;
            let (policy_logits, value_pred) = net.forward_train(&state_t)?;
            let value_loss = loss::mse(&value_pred, &outcome_t)?;
            let log_probs = ops::log_softmax(&policy_logits, 1)?;
            let policy_loss = policy_t.mul(&log_probs)?.mean_all()?.neg()?;
            let total_loss = (value_loss + (policy_loss * POLICY_LOSS_WEIGHT)?)?;
            opt.backward_step(&total_loss)?;
        }
        println!("  [Model {label}] Epoch {epoch}/{EPOCHS} complete.");
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // --- 1. Load ALL training data ---
    let data_dir = "training_data";
    fs::create_dir_all(data_dir)?;

    let mut data_files: Vec<_> = fs::read_dir(data_dir)?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
        .collect();
    data_files.sort_by_key(|e| e.metadata().unwrap().modified().unwrap());

    let mut data: Vec<TrainingData> = data_files.iter()
        .flat_map(|entry| {
            println!("Loading: {:?}", entry.path());
            let file = File::open(entry.path()).expect("open failed");
            serde_json::from_reader::<_, Vec<TrainingData>>(BufReader::new(file)).unwrap_or_default()
        })
        .collect();

    println!("\nLoaded {} training samples from {} file(s).\n", data.len(), data_files.len());
    if data.is_empty() {
        println!("No training data found. Run headless --self-play first.");
        return Ok(());
    }

    // --- 2. Determine next version number ---
    let training_models_dir = "training_models";
    fs::create_dir_all(training_models_dir)?;

    let next_version = fs::read_dir(training_models_dir)?
        .filter_map(Result::ok)
        .filter_map(|e| {
            e.path().file_stem()
                .and_then(|s| s.to_str()?.strip_prefix("azul_model_v")?.parse::<u32>().ok())
        })
        .max()
        .map(|v| v + 1)
        .unwrap_or(1);

    let device = Device::Cpu;
    let path_a = format!("{training_models_dir}/bootstrap_a.safetensors");
    let path_b = format!("{training_models_dir}/bootstrap_b.safetensors");

    // --- 3. Train model A ---
    println!("=== Training Model A (random weights) ===");
    let net_a = TacticianNet::new_random(TacticianConfig::default());
    data.shuffle(&mut rand::thread_rng());
    train_net(&net_a, &data, &device, "A")?;
    net_a.save(&path_a)?;
    println!("Model A saved.\n");

    // --- 4. Train model B ---
    println!("=== Training Model B (random weights) ===");
    let net_b = TacticianNet::new_random(TacticianConfig::default());
    data.shuffle(&mut rand::thread_rng());
    train_net(&net_b, &data, &device, "B")?;
    net_b.save(&path_b)?;
    println!("Model B saved.\n");

    // --- 5. Tournament ---
    println!("=== Tournament: {TOURNAMENT_GAMES} games, {TOURNAMENT_MCTS_ITERS} MCTS iters ===");
    let (a_wins, ties): (u32, u32) = (0..TOURNAMENT_GAMES)
        .into_par_iter()
        .map(|i| {
            let r = run_game(&path_a, &path_b, i % 2 == 0);
            (if r == 1.0 { 1u32 } else { 0 }, if r == 0.5 { 1u32 } else { 0 })
        })
        .reduce(|| (0, 0), |(w1, t1), (w2, t2)| (w1 + w2, t1 + t2));
    let b_wins = TOURNAMENT_GAMES - a_wins - ties;
    let a_rate = (a_wins as f64 + ties as f64 * 0.5) / TOURNAMENT_GAMES as f64;
    println!("A: {a_wins} W / {ties} D / {b_wins} L — {:.1}% adjusted win rate", a_rate * 100.0);

    // --- 6. Save winner as next versioned model ---
    let winner_path = format!("{training_models_dir}/azul_model_v{next_version}.safetensors");
    let winner_label = if a_rate >= 0.5 { "A" } else { "B" };
    let winner_net = if a_rate >= 0.5 { &net_a } else { &net_b };
    println!("\nWinner: Model {winner_label} — saving to '{winner_path}'");
    winner_net.save(&winner_path)?;

    fs::remove_file(&path_a)?;
    fs::remove_file(&path_b)?;

    println!("Bootstrap complete. New base model: '{winner_path}'");
    Ok(())
}
