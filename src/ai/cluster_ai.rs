use crate::{
    ai::{
        heuristic_utils::{build_candidates, floor_penalty, wall_placement_score, CandidateMove},
        AIAgent,
    },
    GameState, Move, MoveDestination, PlayerBoard, Tile,
};
use std::any::Any;

/// Fixed wall color layout — same as WALL_LAYOUT in lib.rs.
const WALL_COLOR: [[Tile; 5]; 5] = [
    [Tile::Blue,   Tile::Yellow, Tile::Red,   Tile::Black, Tile::White],  // row 0
    [Tile::White,  Tile::Blue,   Tile::Yellow, Tile::Red,  Tile::Black],  // row 1
    [Tile::Black,  Tile::White,  Tile::Blue,  Tile::Yellow, Tile::Red],   // row 2
    [Tile::Red,    Tile::Black,  Tile::White, Tile::Blue,  Tile::Yellow], // row 3
    [Tile::Yellow, Tile::Red,    Tile::Black, Tile::White, Tile::Blue],   // row 4
];

/// Wall column where `color` lands when pattern row `row` is completed.
fn wc(row: usize, color: Tile) -> usize {
    WALL_COLOR[row].iter().position(|&c| c == color).expect("color must appear in row")
}

/// Color needed at wall position (row, col).
fn color_for(row: usize, col: usize) -> Tile {
    WALL_COLOR[row][col]
}

/// The 5-spot color for `primary_col` — collecting it serves both column and color bonus.
fn anchor_color(primary_col: usize) -> Tile {
    WALL_COLOR[4][primary_col]
}

/// Total filled wall cells for one player.
fn count_wall_tiles(player: &PlayerBoard) -> usize {
    player.wall.iter().flat_map(|r| r.iter()).filter(|c| c.is_some()).count()
}

/// Orthogonal filled-neighbor count for (row, col) on the wall.
fn count_adj(wall: &[Vec<Option<Tile>>], row: usize, col: usize) -> usize {
    let mut adj = 0;
    if row > 0 && wall[row - 1][col].is_some() { adj += 1; }
    if row < 4 && wall[row + 1][col].is_some() { adj += 1; }
    if col > 0 && wall[row][col - 1].is_some() { adj += 1; }
    if col < 4 && wall[row][col + 1].is_some() { adj += 1; }
    adj
}

/// Count available tiles of `color` across all factories and center.
fn count_color_available(color: Tile, state: &GameState) -> usize {
    state.factories.iter().map(|f| f.iter().filter(|&&t| t == color).count()).sum::<usize>()
        + state.center.iter().filter(|&&t| t == color).count()
}

/// Estimate how many opponents will compete for `col` based on supply attractiveness.
fn estimate_contention(col: usize, totals: &[u32; 5], mins: &[u32; 5], n_opponents: usize) -> f32 {
    let attractiveness: Vec<f32> = (0..5).map(|c| {
        if mins[c] == 0 { return 0.0; }
        let supply = totals[c] as f32 / 25.0;
        let fragility = if mins[c] == 1 { 0.3 } else { 0.0 };
        let pos_bonus = match c { 2 => 0.15, 1 | 3 => 0.05, _ => 0.0 };
        supply - fragility + pos_bonus
    }).collect();

    let mut ranked: Vec<usize> = (0..5).collect();
    ranked.sort_by(|&a, &b| attractiveness[b].partial_cmp(&attractiveness[a]).unwrap());
    let rank = ranked.iter().position(|&c| c == col).unwrap_or(4);

    let base = match rank { 0 => 0.7, 1 => 0.4, 2 => 0.2, _ => 0.05 };
    base * n_opponents as f32
}

// ---------------------------------------------------------------------------

struct ColumnTargets {
    primary: usize,
    secondary: Option<usize>,
}

/// A stateful, column-focused heuristic agent that builds tight vertical clusters on
/// the wall, prioritises rows 0 and 1 every round, and forces game end by round 5.
pub struct ClusterAI {
    round: u32,
    prev_wall_count: usize,
    targets: ColumnTargets,
    // Scoring weights — adjust freely.
    cluster_weight: f32,
    column_weight: f32,
    row_priority_weight: f32,
    placement_weight: f32,
    end_game_weight: f32,
    deep_anchor_weight: f32,
    color_bonus_weight: f32,
    row_adj_penalty: f32,
    single_2spot_penalty: f32,
    deep_mispos_penalty: f32,
}

impl Default for ClusterAI {
    fn default() -> Self {
        Self {
            round: 0,
            prev_wall_count: 0,
            targets: ColumnTargets { primary: 2, secondary: None },
            cluster_weight: 2.0,
            column_weight: 1.8,
            row_priority_weight: 1.5,
            placement_weight: 1.0,
            end_game_weight: 1.0,
            deep_anchor_weight: 1.0,
            color_bonus_weight: 0.9,
            row_adj_penalty: 1.5,
            single_2spot_penalty: 4.0,
            deep_mispos_penalty: 1.0,
        }
    }
}

impl ClusterAI {
    // --- Lifecycle ---

    /// Runs once before the first move of the game. Chooses a primary column.
    fn on_game_start(&mut self, state: &GameState) {
        self.targets.primary = Self::select_primary_column(state);
        self.targets.secondary = None;
        self.round = 1;
        self.prev_wall_count = 0;
    }

    /// Runs at the start of each new round (detected by increased wall tile count).
    fn on_round_start(&mut self, state: &GameState) {
        let player = &state.players[state.current_player_idx];

        // Promote / open secondary column once primary is partially built.
        if self.targets.secondary.is_none() {
            let primary_filled = (0..5).filter(|&r| player.wall[r][self.targets.primary].is_some()).count();
            if primary_filled >= 1 {
                self.targets.secondary = self.pick_secondary(player, self.targets.primary);
            }
        }

        // Pivot primary if any required color has 0 supply this round.
        self.check_column_pivot(state);

        self.round += 1;
        self.prev_wall_count = count_wall_tiles(player);
    }

    // --- Column selection helpers ---

    fn select_primary_column(state: &GameState) -> usize {
        let n_opponents = state.players.len().saturating_sub(1);

        let mut totals = [0u32; 5];
        let mut mins = [0u32; 5];

        for col in 0..5 {
            let mut col_min = u32::MAX;
            let mut col_total = 0u32;
            for row in 0..5 {
                let avail = count_color_available(color_for(row, col), state) as u32;
                col_total += avail;
                col_min = col_min.min(avail);
            }
            totals[col] = col_total;
            mins[col] = if col_min == u32::MAX { 0 } else { col_min };
        }

        // Col 2 preferred when feasible and not over-contested.
        if mins[2] > 0 {
            let contention = estimate_contention(2, &totals, &mins, n_opponents);
            if contention < 1.5 {
                return 2; // MiddleColumnClear
            }
            // Too contested — try an adjacent column.
            let alt = if totals[1] >= totals[3] { 1 } else { 3 };
            if mins[alt] > 0 {
                return alt; // MiddleColumnContestedDeflect
            }
        } else {
            // Col 2 scarce — try adjacent columns.
            let alt = if mins[1] >= mins[3] { 1 } else { 3 };
            if mins[alt] > 0 {
                return alt; // MiddleColumnScarceDeflect
            }
        }

        // ForcedChoice: pick any feasible column with the most supply.
        (0..5)
            .filter(|&c| mins[c] > 0)
            .max_by_key(|&c| totals[c])
            .unwrap_or(2)
    }

    fn pick_secondary(&self, player: &PlayerBoard, primary: usize) -> Option<usize> {
        let candidates: Vec<usize> = match primary {
            0 => vec![1],
            4 => vec![3],
            _ => vec![primary + 1, primary - 1],
        };
        candidates.into_iter()
            .filter(|&c| (0..5).any(|r| player.wall[r][c].is_none())) // not already complete
            .min_by_key(|&c| (0..5).filter(|&r| player.wall[r][c].is_some()).count())
    }

    /// Pivot primary column if any required color has zero supply in the current round.
    fn check_column_pivot(&mut self, state: &GameState) {
        let player = &state.players[state.current_player_idx];
        let primary = self.targets.primary;

        let blocked = (0..5).any(|row| {
            player.wall[row][primary].is_none()
                && count_color_available(color_for(row, primary), state) == 0
        });

        if !blocked {
            return;
        }

        let candidates: Vec<usize> = match primary {
            0 => vec![1],
            4 => vec![3],
            _ => vec![primary - 1, primary + 1],
        };
        if let Some(&new_primary) = candidates.iter().find(|&&c| {
            (0..5).any(|r| player.wall[r][c].is_none())
        }) {
            self.targets.primary = new_primary;
            self.targets.secondary = None;
        }
    }

    // --- Move scoring ---

    fn score_move(&self, c: &CandidateMove, state: &GameState) -> f32 {
        let player = &state.players[state.current_player_idx];
        let cur_floor = player.floor_line.len();

        let (pattern_row, color) = match c.m.destination {
            MoveDestination::Floor => {
                return -floor_penalty(c.tiles_available, cur_floor) * 2.0;
            }
            MoveDestination::PatternLine(idx) => (idx, c.m.tile),
        };

        let wall_col = wc(pattern_row, color);

        // Waste penalty.
        let overflow = c.tiles_available.saturating_sub(c.pattern_line_space);
        let mut score = -floor_penalty(overflow, cur_floor) * 1.5;

        // Term 1: Cluster adjacency — touching existing wall tiles is always better.
        let adj = count_adj(&player.wall, pattern_row, wall_col);
        score += adj as f32 * self.cluster_weight;

        // Term 2: Column alignment.
        if wall_col == self.targets.primary {
            score += self.column_weight;
        } else if self.targets.secondary == Some(wall_col) {
            score += self.column_weight * 0.7;
        }

        // Term 3: Row priority — rows 0 and 1 are always urgent.
        match pattern_row {
            0 => score += self.row_priority_weight * 2.0,
            1 => score += self.row_priority_weight,
            _ => {}
        }

        // Term 4: Immediate placement score (only meaningful when completing the row).
        if c.completes_row {
            score += wall_placement_score(&player.wall, pattern_row, wall_col) * self.placement_weight;
        }

        // Term 5: End-game forcing — completing row 0 in late rounds is critical.
        if self.round >= 4 && pattern_row == 0 && c.completes_row {
            score += self.end_game_weight * 5.0;
        }

        // Term 6: Deep anchor — starting the 5-spot row early gives time to fill it.
        let is_first_tile = player.pattern_lines[pattern_row].is_empty();
        if pattern_row == 4 && is_first_tile && self.round <= 3 {
            score += self.deep_anchor_weight;
        }

        // Term 7: Anchor color bonus — this color serves column and color bonus together.
        if color == anchor_color(self.targets.primary) {
            score += self.color_bonus_weight;
        }

        // Penalty: placing a single tile in the 2-spot (row 1) when it won't complete it.
        if pattern_row == 1 && !c.completes_row && c.tiles_available == 1 {
            score -= self.single_2spot_penalty;
        }

        // Penalty: placing the first tile in a deep row (3-4) off the target columns.
        if pattern_row >= 3 && is_first_tile {
            let on_target = wall_col == self.targets.primary
                || self.targets.secondary == Some(wall_col);
            if !on_target {
                score -= self.deep_mispos_penalty * pattern_row as f32;
            }
        }

        // Penalty: isolated placement (no adjacency) outside target columns.
        if adj == 0 {
            let on_target = wall_col == self.targets.primary
                || self.targets.secondary == Some(wall_col)
                || pattern_row == 0; // row 0 always worth pursuing regardless
            if !on_target {
                score -= self.row_adj_penalty;
            }
        }

        score
    }

    fn select_move(&self, state: &GameState) -> Option<Move> {
        build_candidates(state)
            .into_iter()
            .max_by(|a, b| {
                self.score_move(a, state)
                    .partial_cmp(&self.score_move(b, state))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.m)
    }
}

impl AIAgent for ClusterAI {
    fn get_move(&mut self, state: &GameState) -> Option<Move> {
        if self.round == 0 {
            self.on_game_start(state);
        }
        let wall_count = count_wall_tiles(&state.players[state.current_player_idx]);
        if wall_count > self.prev_wall_count {
            self.on_round_start(state);
        }
        self.select_move(state)
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
