use azul_engine::ai::{
    simple_ai::SimpleAI, 
    heuristic_ai::HeuristicAI, 
    mcts_heuristic_ai::MctsHeuristicAI,
    mcts_nn_ai::MctsNnAI,
    AIAgent
};
use azul_engine::{GameState, Move, TileBagSummary, TurnState, TrainingData};
use chrono::prelude::*;
use clap::Parser;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short, long, num_args = 1.., value_delimiter = ' ', required = true)]
    players: Vec<String>,
    #[arg(short, long, default_value_t = 100)]
    games: u32,
    #[arg(long)]
    self_play: bool,
    #[arg(long, default_value_t = 2)]
    self_play_players: usize,
}

#[derive(Serialize)]
struct GameTurn {
    player_index: usize,
    state_before_move: TurnState,
    chosen_move: Move,
}

#[derive(Serialize)]
struct GameRound {
    round_number: usize,
    tile_bag_at_start_of_round: TileBagSummary,
    turns: Vec<GameTurn>,
}

#[derive(Serialize)]
struct GameLog {
    matchup: Vec<String>,
    history: Vec<GameRound>,
    final_scores: Vec<u32>,
}

#[derive(Serialize)]
struct GameStats {
    agent_wins: HashMap<String, u32>,
    total_games: u32,
    ties: u32,
    simulation_time_seconds: f64,
}

impl GameStats {
    fn new() -> Self {
        Self {
            agent_wins: HashMap::new(),
            total_games: 0,
            ties: 0,
            simulation_time_seconds: 0.0,
        }
    }

    fn record_game(&mut self, final_state: &GameState, agent_names: &[String]) {
        self.total_games += 1;
        let winner = final_state.players.iter().enumerate().max_by(|(_, a), (_, b)| {
            let score_cmp = a.score.cmp(&b.score);
            if score_cmp != std::cmp::Ordering::Equal { return score_cmp; }
            a.count_complete_rows().cmp(&b.count_complete_rows())
        });

        if let Some((winner_idx, winner_player)) = winner {
            let is_tie = final_state.players.iter().any(|p| {
                p != winner_player &&
                p.score == winner_player.score &&
                p.count_complete_rows() == winner_player.count_complete_rows()
            });

            if !is_tie {
                let winner_name = &agent_names[winner_idx];
                *self.agent_wins.entry(winner_name.clone()).or_insert(0) += 1;
            } else {
                self.ties += 1;
            }
        }
    }

    fn print_summary(&self) {
        println!("\n--- Simulation Complete ---");
        println!("Total Games: {}", self.total_games);
        println!("Total Time: {:.2} seconds", self.simulation_time_seconds);
        println!("Ties: {}", self.ties);
        println!("Wins by Agent:");
        for (name, wins) in &self.agent_wins {
            let win_rate = (*wins as f64 / self.total_games as f64) * 100.0;
            println!("  - {}: {} ({:.2}%)", name, wins, win_rate);
        }
    }
}

fn create_agent(name: &str) -> Box<dyn AIAgent> {
    let parts: Vec<&str> = name.split(':').collect();
    let agent_type = parts[0].to_lowercase();

    match agent_type.as_str() {
        "simpleai" => Box::new(SimpleAI),
        "heuristicai" => Box::new(HeuristicAI),
        "mctsheuristic" => {
            let iterations = if parts.len() > 1 { parts[1].parse::<u32>().unwrap_or(5000) } else { 5000 };
            Box::new(MctsHeuristicAI::new(iterations))
        }
        "mctsnn" => {
            let iterations = if parts.len() > 1 { parts[1].parse::<u32>().unwrap_or(800) } else { 800 };
            let model_path = if parts.len() > 2 { Some(parts[2].to_string()) } else { None };
            Box::new(MctsNnAI::new(iterations, model_path, None))
        }
        _ => panic!("Unknown AI type: {}", name),
    }
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    if cli.self_play {
        run_self_play(cli)?;
    } else {
        run_simulations(cli)?;
    }
    Ok(())
}

fn run_self_play(cli: Cli) -> std::io::Result<()> {
    let num_games = cli.games;
    let mut agent_config = cli.players[0].clone();
    let num_players = cli.self_play_players;

    if !(2..=4).contains(&num_players) {
        eprintln!("Error: Self-play player count must be between 2 and 4.");
        return Ok(());
    }

    // --- MODIFIED SECTION: Auto-find latest model for self-play ---
    let parts: Vec<&str> = agent_config.split(':').collect();
    if parts[0].to_lowercase() == "mctsnn" && parts.len() < 3 {
        let training_models_dir = "training_models";
        fs::create_dir_all(training_models_dir)?;
        let latest_model = fs::read_dir(training_models_dir)?
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "ot"))
            .max_by_key(|entry| entry.metadata().unwrap().created().unwrap());

        if let Some(entry) = latest_model {
            let path_str = entry.path().to_string_lossy().to_string();
            println!("Found latest model for self-play: {}", path_str);
            // Append the path to the agent config string
            agent_config = format!("{}:{}", agent_config, path_str);
        } else {
            println!("No existing model found. Starting self-play with a random brain.");
        }
    }
    // --- END MODIFIED SECTION ---

    println!("Running {} {}-player self-play games to generate training data...", num_games, num_players);
    let start_time = Instant::now();

    let all_training_data: Vec<TrainingData> = (0..num_games)
        .into_par_iter()
        .flat_map(|_| {
            let mut agents: Vec<Box<dyn AIAgent>> = (0..num_players)
                .map(|_| create_agent(&agent_config))
                .collect();
            run_one_self_play_game(&mut agents)
        })
        .collect();

    let duration = start_time.elapsed();
    println!("\n--- Self-Play Complete ---");
    println!("Generated {} training samples in {:.2} seconds.", all_training_data.len(), duration.as_secs_f64());

    println!("Saving training data...");
    fs::create_dir_all("training_data")?;
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let data_path = format!("training_data/data_{}.json", timestamp);
    let data_file = fs::File::create(&data_path)?;
    serde_json::to_writer_pretty(data_file, &all_training_data)?;
    println!("Done. Data saved to '{}'", data_path);
    Ok(())
}

fn run_one_self_play_game(agents: &mut [Box<dyn AIAgent>]) -> Vec<TrainingData> {
    let num_players = agents.len();
    let mut game = GameState::new(num_players);
    let mut history: Vec<(Vec<f32>, Vec<f32>, usize)> = Vec::new();

    while !game.end_game_triggered {
        while !game.is_round_over() {
            let player_idx = game.current_player_idx;
            let agent = &mut agents[player_idx];
            let state_input_opt = agent.as_any().downcast_ref::<MctsNnAI>().and_then(|a| a.state_to_input(&game));

            if let Some(the_move) = agent.get_move(&game) {
                let mcts_agent = agent.as_any().downcast_ref::<MctsNnAI>().unwrap();
                if let (Some(state_input), Some(mcts_policy)) = (state_input_opt, mcts_agent.get_mcts_policy()) {
                    history.push((state_input, mcts_policy, player_idx));
                }
                game.apply_move(&the_move);
            } else {
                break;
            }
        }
        game.run_tiling_phase();
        if !game.end_game_triggered { game.refill_factories(); }
    }
    game.apply_end_game_scoring();

    let mut training_data = Vec::new();
    let winner_idx = game.players.iter().enumerate().max_by_key(|(_, p)| p.score).map(|(i, _)| i);

    for (state_input, mcts_policy, player_idx) in history {
        let outcome = if Some(player_idx) == winner_idx { 1.0 } else { -1.0 };
        training_data.push(TrainingData { state_input, mcts_policy, outcome });
    }
    training_data
}

fn run_simulations(cli: Cli) -> std::io::Result<()> {
    let num_games = cli.games;
    let agent_config = cli.players;
    println!("Running {} {}-player games in parallel...", num_games, agent_config.len());
    let start_time = Instant::now();

    let game_results: Vec<(GameState, GameLog)> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut current_matchup = agent_config.clone();
            let len = current_matchup.len();
            if len > 0 { current_matchup.rotate_left(i as usize % len); }
            let agents: Vec<Box<dyn AIAgent>> = current_matchup.iter().map(|name| create_agent(name)).collect();
            run_game(agents, current_matchup)
        })
        .collect();

    let duration = start_time.elapsed();
    let mut stats = GameStats::new();
    stats.simulation_time_seconds = duration.as_secs_f64();
    for name in &agent_config {
        stats.agent_wins.entry(name.clone()).or_insert(0);
    }
    let mut game_logs: Vec<GameLog> = Vec::with_capacity(num_games as usize);
    for (final_state, game_log) in game_results {
        stats.record_game(&final_state, &agent_config);
        game_logs.push(game_log);
    }

    stats.print_summary();
    println!("\nSaving results...");
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let output_dir = format!("stats/{}", timestamp);
    fs::create_dir_all(&output_dir)?;
    let stats_path = format!("{}/summary_stats.json", output_dir);
    let logs_path = format!("{}/game_logs.json", output_dir);
    let stats_file = fs::File::create(&stats_path)?;
    serde_json::to_writer_pretty(stats_file, &stats)?;
    let logs_file = fs::File::create(&logs_path)?;
    serde_json::to_writer_pretty(logs_file, &game_logs)?;
    println!("Done. Results saved in '{}' directory.", output_dir);
    Ok(())
}

fn run_game(mut agents: Vec<Box<dyn AIAgent>>, matchup: Vec<String>) -> (GameState, GameLog) {
    let mut game = GameState::new(agents.len());
    let mut round_history: Vec<GameRound> = Vec::new();
    let mut round_counter = 1;

    while !game.end_game_triggered {
        let tile_bag_at_start = TileBagSummary::from_vec(&game.tile_bag);
        let mut turns_this_round: Vec<GameTurn> = Vec::new();
        while !game.is_round_over() {
            let state_before_move = TurnState::from(&game);
            let agent = &mut agents[game.current_player_idx];
            if let Some(ai_move) = agent.get_move(&game) {
                let turn = GameTurn {
                    player_index: game.current_player_idx,
                    state_before_move,
                    chosen_move: ai_move.clone(),
                };
                turns_this_round.push(turn);
                game.apply_move(&ai_move);
            } else {
                break;
            }
        }
        round_history.push(GameRound {
            round_number: round_counter,
            tile_bag_at_start_of_round: tile_bag_at_start,
            turns: turns_this_round,
        });
        game.run_tiling_phase();
        if !game.end_game_triggered { game.refill_factories(); }
        round_counter += 1;
    }
    game.apply_end_game_scoring();
    let log = GameLog {
        matchup,
        history: round_history,
        final_scores: game.players.iter().map(|p| p.score).collect(),
    };
    (game, log)
}
