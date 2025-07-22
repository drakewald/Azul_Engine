use azul_engine::ai::{simple_ai::SimpleAI, heuristic_ai::HeuristicAI, mcts_ai::MctsAI, AIAgent};
use azul_engine::{GameState, Move};
use clap::Parser;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short, long, num_args = 2.., value_delimiter = ' ', required = true)]
    players: Vec<String>,
    #[arg(short, long, default_value_t = 100)]
    games: u32,
}

#[derive(Serialize)]
struct GameLog {
    matchup: Vec<String>,
    history: Vec<(usize, Move)>,
    final_scores: Vec<u32>,
}

#[derive(Serialize)]
struct GameStats {
    agent_wins: HashMap<String, u32>,
    total_games: u32,
    ties: u32,
}

impl GameStats {
    fn new(agent_names: &[String]) -> Self {
        let mut agent_wins = HashMap::new();
        for name in agent_names {
            agent_wins.entry(name.clone()).or_insert(0);
        }
        Self {
            agent_wins,
            total_games: 0,
            ties: 0,
        }
    }

    fn record_game(&mut self, final_state: &GameState, agent_names: &[String]) {
        self.total_games += 1;
        let winner = final_state
            .players
            .iter()
            .enumerate()
            .max_by_key(|(_, player)| player.score);

        if let Some((winner_idx, winner_player)) = winner {
            let is_tie = final_state
                .players
                .iter()
                .any(|p| p.score == winner_player.score && p != winner_player);

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
        println!("Ties: {}", self.ties);
        println!("Wins by Agent:");
        for (name, wins) in &self.agent_wins {
            let win_rate = (*wins as f64 / self.total_games as f64) * 100.0;
            println!("  - {}: {} ({:.2}%)", name, wins, win_rate);
        }
    }
}

fn create_agent(name: &str) -> Box<dyn AIAgent> {
    match name.to_lowercase().as_str() {
        "simpleai" => Box::new(SimpleAI),
        "heuristicai" => Box::new(HeuristicAI),
        "mctsai" => Box::new(MctsAI::new()),
        _ => panic!("Unknown AI type: {}", name),
    }
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let num_games = cli.games;
    let agent_config = cli.players;

    println!(
        "Running {} {}-player games...",
        num_games,
        agent_config.len()
    );

    let mut stats = GameStats::new(&agent_config);
    let mut game_logs: Vec<GameLog> = Vec::with_capacity(num_games as usize);

    for i in 0..num_games {
        let mut current_matchup = agent_config.clone();
        
        // --- THIS SECTION IS FIXED ---
        // First, get the length (immutable borrow).
        let len = current_matchup.len();
        // Then, perform the rotation (mutable borrow).
        current_matchup.rotate_left(i as usize % len);
        // --- END OF FIX ---
        
        let agents: Vec<Box<dyn AIAgent>> =
            current_matchup.iter().map(|name| create_agent(name)).collect();

        let (final_state, game_log) = run_game(agents, current_matchup);
        stats.record_game(&final_state, &agent_config);
        game_logs.push(game_log);
    }

    stats.print_summary();

    println!("\nSaving results...");
    fs::create_dir_all("stats")?;

    let stats_file = fs::File::create("stats/summary_stats.json")?;
    serde_json::to_writer_pretty(stats_file, &stats)?;

    let logs_file = fs::File::create("stats/game_logs.json")?;
    serde_json::to_writer_pretty(logs_file, &game_logs)?;
    
    println!("Done. Results saved in 'stats/' directory.");
    Ok(())
}

fn run_game(mut agents: Vec<Box<dyn AIAgent>>, matchup: Vec<String>) -> (GameState, GameLog) {
    let mut game = GameState::new(agents.len());
    let mut history: Vec<(usize, Move)> = Vec::new();

    while !game.end_game_triggered {
        while !game.is_round_over() {
            let agent = &mut agents[game.current_player_idx];
            if let Some(ai_move) = agent.get_move(&game) {
                history.push((game.current_player_idx, ai_move.clone()));
                game.apply_move(&ai_move);
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

    let log = GameLog {
        matchup,
        history,
        final_scores: game.players.iter().map(|p| p.score).collect(),
    };

    (game, log)
}