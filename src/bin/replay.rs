use azul_engine::{GameState, Move, MoveDestination, MoveSource, Tile};
use clap::Parser;
use serde::Deserialize;
use std::io::{self, BufRead, Write};

#[derive(Parser)]
#[command(about = "Step through recorded Azul games from a game_logs.json.zst file")]
struct Cli {
    /// Path to a game_logs.json.zst file
    path: String,
    /// Start at this game (1-based)
    #[arg(short, long, default_value_t = 1)]
    game: usize,
}

#[derive(Deserialize)]
struct GameTurn {
    player_index: usize,
    chosen_move: Move,
}

#[derive(Deserialize)]
struct GameRound {
    round_number: usize,
    #[serde(default)]
    factories_at_start: Vec<Vec<Tile>>,
    turns: Vec<GameTurn>,
}

#[derive(Deserialize)]
struct GameLog {
    matchup: Vec<String>,
    initial_state: GameState,
    history: Vec<GameRound>,
    final_scores: Vec<u32>,
}

struct Step {
    state: GameState,
    desc: String,
}

// Mirror of lib.rs WALL_LAYOUT (private there, duplicated here)
const WALL: [[Tile; 5]; 5] = [
    [Tile::Blue,   Tile::Yellow, Tile::Red,    Tile::Black, Tile::White],
    [Tile::White,  Tile::Blue,   Tile::Yellow, Tile::Red,   Tile::Black],
    [Tile::Black,  Tile::White,  Tile::Blue,   Tile::Yellow, Tile::Red],
    [Tile::Red,    Tile::Black,  Tile::White,  Tile::Blue,  Tile::Yellow],
    [Tile::Yellow, Tile::Red,    Tile::Black,  Tile::White, Tile::Blue],
];

fn tile_ansi(t: Tile) -> &'static str {
    match t {
        Tile::Blue   => "\x1b[94m",
        Tile::Yellow => "\x1b[93m",
        Tile::Red    => "\x1b[91m",
        Tile::Black  => "\x1b[90m",
        Tile::White  => "\x1b[97m",
    }
}

fn tile_char(t: Tile) -> char {
    match t {
        Tile::Blue   => 'B',
        Tile::Yellow => 'Y',
        Tile::Red    => 'R',
        Tile::Black  => 'K',
        Tile::White  => 'W',
    }
}

fn ct(t: Tile) -> String {
    format!("{}{}\x1b[0m", tile_ansi(t), tile_char(t))
}

fn build_steps(log: &GameLog) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut state = log.initial_state.clone();
    steps.push(Step { state: state.clone(), desc: "Game Start".to_string() });

    let num_rounds = log.history.len();
    for (r_idx, round) in log.history.iter().enumerate() {
        for (t_idx, turn) in round.turns.iter().enumerate() {
            let name = log.matchup.get(turn.player_index).map(|s| s.as_str()).unwrap_or("?");
            let src = match &turn.chosen_move.source {
                MoveSource::Factory(i) => format!("F{}", i + 1),
                MoveSource::Center => "Ctr".to_string(),
            };
            let dst = match &turn.chosen_move.destination {
                MoveDestination::PatternLine(i) => format!("L{}", i + 1),
                MoveDestination::Floor => "Floor".to_string(),
            };
            let desc = format!(
                "R{} T{}  P{} ({})  {} [{}] → {}",
                round.round_number, t_idx + 1,
                turn.player_index, name,
                src, tile_char(turn.chosen_move.tile), dst,
            );
            state.apply_move(&turn.chosen_move);
            steps.push(Step { state: state.clone(), desc });
        }

        if r_idx + 1 < num_rounds {
            state.run_tiling_phase();
            let next = &log.history[r_idx + 1];
            if !next.factories_at_start.is_empty() {
                state.factories = next.factories_at_start.clone();
            } else {
                // Old log without factories_at_start: simulate refill (factories will differ)
                state.refill_factories();
            }
            state.center.clear();
            state.first_player_marker_in_center = true;
            steps.push(Step {
                state: state.clone(),
                desc: format!("── Round {} → Round {} ──", round.round_number, next.round_number),
            });
        } else {
            state.run_tiling_phase();
            state.apply_end_game_scoring();
            steps.push(Step { state: state.clone(), desc: "── Game Over ──".to_string() });
        }
    }

    if log.history.is_empty() {
        state.apply_end_game_scoring();
        steps.push(Step { state: state.clone(), desc: "── Game Over ──".to_string() });
    }

    steps
}

fn render_factories(state: &GameState) {
    println!("\x1b[1mFactories:\x1b[0m");
    for (i, factory) in state.factories.iter().enumerate() {
        print!("  F{}: ", i + 1);
        if factory.is_empty() {
            print!("(empty)");
        } else {
            for &t in factory {
                print!("[{}]", ct(t));
            }
        }
        println!();
    }

    print!("\x1b[1mCenter:\x1b[0m  ");
    if state.first_player_marker_in_center {
        print!("\x1b[1m[1P]\x1b[0m ");
    }
    if state.center.is_empty() && !state.first_player_marker_in_center {
        print!("(empty)");
    } else {
        for &t in &state.center {
            print!("[{}]", ct(t));
        }
    }
    println!();
}

fn render_player(idx: usize, name: &str, player: &azul_engine::PlayerBoard, is_current: bool) {
    let marker = if is_current { "  \x1b[32m◄ current\x1b[0m" } else { "" };
    println!("\x1b[1mPlayer {} ({})  Score: {}{}\x1b[0m", idx, name, player.score, marker);
    println!("  Pattern Lines              Wall");

    for row in 0..5 {
        let cap = row + 1;
        let line = &player.pattern_lines[row];

        // Pattern line: right-align in 5-slot space, empty slots on left
        print!("  ");
        for _ in 0..(5 - cap) { print!("   "); }             // leading indent
        for _ in 0..(cap - line.len()) { print!("[_]"); }     // empty slots
        for &t in line { print!("[{}]", ct(t)); }             // tiles
        print!("  │  ");

        // Wall: filled = colored tile char, empty = dimmed expected color
        for col in 0..5 {
            match player.wall[row][col] {
                Some(t) => print!("[{}]", ct(t)),
                None => {
                    let exp = WALL[row][col];
                    print!("\x1b[2m[{}]\x1b[0m", tile_char(exp));
                }
            }
        }
        println!();
    }

    print!("  Floor: ");
    if player.has_first_player_marker { print!("\x1b[1m[1P]\x1b[0m "); }
    if player.floor_line.is_empty() && !player.has_first_player_marker {
        print!("(empty)");
    } else {
        for &t in &player.floor_line { print!("[{}]", ct(t)); }
    }
    println!("\n");
}

fn render(game_idx: usize, total: usize, step_idx: usize, steps: &[Step], log: &GameLog) {
    let step = &steps[step_idx];
    let state = &step.state;

    print!("\x1b[2J\x1b[H"); // clear screen, cursor home

    // Header
    println!(
        "\x1b[1m═══ Game {}/{} ═══ {} ═══\x1b[0m",
        game_idx + 1,
        total,
        log.matchup.join(" vs ")
    );
    println!(
        "Step {}/{}  │  {}",
        step_idx + 1,
        steps.len(),
        step.desc
    );
    println!();

    render_factories(state);
    println!();

    let is_game_over = step_idx == steps.len() - 1;
    for (i, player) in state.players.iter().enumerate() {
        let name = log.matchup.get(i).map(|s| s.as_str()).unwrap_or("?");
        let is_current = !is_game_over && i == state.current_player_idx;
        render_player(i, name, player, is_current);
    }

    if is_game_over {
        println!("\x1b[1mFinal Scores:\x1b[0m");
        let mut ranked: Vec<(usize, u32)> = log.final_scores.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        for (i, (player_idx, score)) in ranked.iter().enumerate() {
            let name = log.matchup.get(*player_idx).map(|s| s.as_str()).unwrap_or("?");
            let trophy = if i == 0 { " 🏆" } else { "" };
            println!("  #{}: P{} ({}) — {}{}", i + 1, player_idx, name, score, trophy);
        }
        println!();
    }

    println!("\x1b[2m[Enter/n] next  [p] prev  [g<N>] jump to game N  [q] quit\x1b[0m");
    print!("> ");
    let _ = io::stdout().flush();
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    print!("Loading {}... ", cli.path);
    let _ = io::stdout().flush();

    let file = std::fs::File::open(&cli.path)?;
    let decoder = zstd::Decoder::new(file)?;
    let logs: Vec<GameLog> = serde_json::from_reader(decoder)?;
    println!("{} game(s) loaded.", logs.len());

    if logs.is_empty() {
        println!("No games found in file.");
        return Ok(());
    }

    let mut game_idx = cli.game.saturating_sub(1).min(logs.len() - 1);
    let mut steps = build_steps(&logs[game_idx]);
    let mut step_idx = 0usize;

    let stdin = io::stdin();

    loop {
        render(game_idx, logs.len(), step_idx, &steps, &logs[game_idx]);

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() || line.is_empty() {
            break;
        }
        let cmd = line.trim();

        match cmd {
            "" | "n" => {
                if step_idx + 1 < steps.len() { step_idx += 1; }
            }
            "p" => {
                if step_idx > 0 { step_idx -= 1; }
            }
            "q" => break,
            s if s.starts_with('g') => {
                if let Ok(n) = s[1..].trim().parse::<usize>() {
                    if n >= 1 && n <= logs.len() {
                        game_idx = n - 1;
                        steps = build_steps(&logs[game_idx]);
                        step_idx = 0;
                    }
                }
            }
            _ => {}
        }
    }

    print!("\x1b[2J\x1b[H");
    Ok(())
}
