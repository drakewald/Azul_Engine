use azul_engine::ai::{
    simple_ai::SimpleAI, 
    heuristic_ai::HeuristicAI, 
    mcts_heuristic_ai::MctsHeuristicAI, // MODIFIED: Correct import
    mcts_nn_ai::MctsNnAI,             // MODIFIED: Correct import
    AIAgent
};
use azul_engine::{GameState, Move, TileBagSummary, TurnState};
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
    #[arg(short, long, num_args = 2.., value_delimiter = ' ', required = true)]
    players: Vec<String>,
    #[arg(short, long, default_value_t = 100)]
    games: u32,
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
            if score_cmp != std::cmp::Ordering::Equal {
                return score_cmp;
            }
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

// MODIFIED: Updated to use the new AI names and constructors
fn create_agent(name: &str) -> Box<dyn AIAgent> {
    let parts: Vec<&str> = name.split(':').collect();
    let agent_type = parts[0].to_lowercase();

    match agent_type.as_str() {
        "simpleai" => Box::new(SimpleAI),
        "heuristicai" => Box::new(HeuristicAI),
        "mctsheuristic" => {
            let iterations = if parts.len() > 1 {
                parts[1].parse::<u32>().unwrap_or(5000)
            } else {
                5000
            };
            Box::new(MctsHeuristicAI::new(iterations))
        }
        "mctsnn" => {
            let iterations = if parts.len() > 1 {
                parts[1].parse::<u32>().unwrap_or(800)
            } else {
                800
            };
            Box::new(MctsNnAI::new(iterations))
        }
        _ => panic!("Unknown AI type: {}", name),
    }
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let num_games = cli.games;
    let agent_config = cli.players;

    println!(
        "Running {} {}-player games in parallel on multiple cores...",
        num_games,
        agent_config.len()
    );

    let start_time = Instant::now();

    let game_results: Vec<(GameState, GameLog)> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let mut current_matchup = agent_config.clone();
            let len = current_matchup.len();
            if len > 0 {
                current_matchup.rotate_left(i as usize % len);
            }

            let agents: Vec<Box<dyn AIAgent>> =
                current_matchup.iter().map(|name| create_agent(name)).collect();

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
        if !game.end_game_triggered {
            game.refill_factories();
        }
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
