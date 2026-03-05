use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

const K_FACTOR: f64 = 32.0;
const INITIAL_ELO: f64 = 1000.0;
pub const STATE_FILE: &str = "elo/ratings.json";

#[derive(Serialize, Deserialize, Default)]
pub struct EloState {
    /// Current ELO rating per agent. BTreeMap keeps keys sorted for clean git diffs.
    pub ratings: BTreeMap<String, f64>,
    /// Total games played per agent.
    pub game_counts: BTreeMap<String, u32>,
    /// Paths of game log files whose games are already reflected in these ratings.
    pub processed_files: BTreeSet<String>,
    pub total_games: u32,
}

/// Strip the model path from mctsnn names so the agent is tracked as one evolving identity.
/// "mctsnn:300:training_models/azul_model_v7.safetensors" -> "mctsnn:300"
pub fn normalize(name: &str) -> String {
    let parts: Vec<&str> = name.splitn(3, ':').collect();
    if parts[0].eq_ignore_ascii_case("mctsnn") && parts.len() == 3 {
        format!("{}:{}", parts[0].to_lowercase(), parts[1])
    } else {
        name.to_lowercase()
    }
}

fn expected(ra: f64, rb: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf((rb - ra) / 400.0))
}

impl EloState {
    pub fn load() -> Self {
        fs::read_to_string(STATE_FILE)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        fs::create_dir_all("elo")?;
        fs::write(STATE_FILE, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Update ratings for one game given the normalized matchup names and final scores.
    pub fn update_game(&mut self, matchup: &[String], final_scores: &[u32]) {
        let names: Vec<String> = matchup.iter().map(|n| normalize(n)).collect();
        let n = names.len();
        if n < 2 || final_scores.len() != n {
            return;
        }

        for name in &names {
            self.ratings.entry(name.clone()).or_insert(INITIAL_ELO);
            *self.game_counts.entry(name.clone()).or_insert(0) += 1;
        }

        // Accumulate all pairwise deltas before applying to avoid order bias within a game.
        let mut deltas = vec![0.0f64; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let ri = self.ratings[&names[i]];
                let rj = self.ratings[&names[j]];
                let ei = expected(ri, rj);
                let (si, sj) = match final_scores[i].cmp(&final_scores[j]) {
                    std::cmp::Ordering::Greater => (1.0, 0.0),
                    std::cmp::Ordering::Less => (0.0, 1.0),
                    std::cmp::Ordering::Equal => (0.5, 0.5),
                };
                deltas[i] += K_FACTOR * (si - ei);
                deltas[j] += K_FACTOR * (sj - (1.0 - ei));
            }
        }
        for (i, name) in names.iter().enumerate() {
            *self.ratings.get_mut(name).unwrap() += deltas[i];
        }

        self.total_games += 1;
    }

    pub fn print_leaderboard(&self) {
        let mut rows: Vec<(&str, f64, u32)> = self
            .ratings
            .iter()
            .map(|(name, &elo)| (name.as_str(), elo, *self.game_counts.get(name).unwrap_or(&0)))
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("\n=== ELO Ratings ({} total games) ===\n", self.total_games);
        println!("{:<36} {:>8}  {:>7}", "Agent", "ELO", "Games");
        println!("{}", "─".repeat(56));
        for (name, elo, games) in rows {
            println!("{:<36} {:>8.1}  {:>7}", name, elo, games);
        }
    }
}
