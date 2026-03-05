use azul_engine::elo::{EloState, STATE_FILE};
use serde::Deserialize;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

/// Minimal re-declaration — Serde ignores unknown fields (initial_state, history, etc.)
#[derive(Deserialize)]
struct GameLog {
    matchup: Vec<String>,
    final_scores: Vec<u32>,
}

fn main() -> anyhow::Result<()> {
    let mut state = EloState::load();

    let stats_dir = std::path::Path::new("stats");
    if !stats_dir.exists() {
        println!("No stats/ directory found.");
        state.print_leaderboard();
        return Ok(());
    }

    let mut log_files: Vec<PathBuf> = fs::read_dir(stats_dir)?
        .filter_map(Result::ok)
        .map(|e| e.path().join("game_logs.json.zst"))
        .filter(|p| p.exists())
        .collect();

    log_files.sort_by_key(|p| {
        fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });

    let new_files: Vec<PathBuf> = log_files
        .into_iter()
        .filter(|p| !state.processed_files.contains(p.to_string_lossy().as_ref()))
        .collect();

    if new_files.is_empty() {
        println!("No new game files to process.");
        state.print_leaderboard();
        return Ok(());
    }

    println!("Processing {} new file(s)...", new_files.len());
    let mut new_games = 0u32;

    for path in &new_files {
        let file = fs::File::open(path)?;
        let mut decoder = zstd::Decoder::new(file)?;
        let mut json_bytes = Vec::new();
        decoder.read_to_end(&mut json_bytes)?;
        let logs: Vec<GameLog> = serde_json::from_slice(&json_bytes)?;

        for log in &logs {
            state.update_game(&log.matchup, &log.final_scores);
            new_games += 1;
        }

        state.processed_files.insert(path.to_string_lossy().into_owned());
    }

    println!("Processed {} new games. Total: {}", new_games, state.total_games);
    state.save()?;
    println!("Saved to {}", STATE_FILE);
    state.print_leaderboard();
    Ok(())
}
