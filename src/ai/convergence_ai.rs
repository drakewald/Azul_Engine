use crate::{GameState, Move, MoveDestination, MoveSource, Tile, WALL_LAYOUT};
use crate::ai::{AIAgent, heuristic_utils::{build_candidates, floor_penalty, wall_col_for, wall_placement_score, CandidateMove}};
use std::any::Any;

pub struct ConvergenceAI;

impl Default for ConvergenceAI {
    fn default() -> Self {
        ConvergenceAI
    }
}

#[derive(Clone, Copy)]
enum WallBonus {
    Row(usize),
    Col(usize),
    Color(usize),
}

impl WallBonus {
    fn all() -> [WallBonus; 15] {
        [
            WallBonus::Row(0), WallBonus::Row(1), WallBonus::Row(2), WallBonus::Row(3), WallBonus::Row(4),
            WallBonus::Col(0), WallBonus::Col(1), WallBonus::Col(2), WallBonus::Col(3), WallBonus::Col(4),
            WallBonus::Color(0), WallBonus::Color(1), WallBonus::Color(2), WallBonus::Color(3), WallBonus::Color(4),
        ]
    }

    fn value(&self) -> f32 {
        match self {
            WallBonus::Row(_) => 2.0,
            WallBonus::Col(_) => 7.0,
            WallBonus::Color(_) => 10.0,
        }
    }
}

fn reachability(tiles_needed: u32) -> f32 {
    match tiles_needed {
        0 => 0.0,
        1 => 1.0,
        2 => 0.6,
        3 => 0.3,
        4 => 0.1,
        _ => 0.0,
    }
}

fn tiles_remaining_for_bonus(bonus: &WallBonus, wall: &[Vec<Option<Tile>>]) -> u32 {
    match bonus {
        WallBonus::Row(r) => wall[*r].iter().filter(|c| c.is_none()).count() as u32,
        WallBonus::Col(c) => (0..5).filter(|&r| wall[r][*c].is_none()).count() as u32,
        WallBonus::Color(k) => (0..5)
            .filter(|&r| {
                if let Some(col) = WALL_LAYOUT[r].iter().position(|&t| t as usize == *k) {
                    wall[r][col].is_none()
                } else {
                    false
                }
            })
            .count() as u32,
    }
}

fn contribution_to_bonus(bonus: &WallBonus, pattern_row: usize, color: Tile) -> f32 {
    match bonus {
        WallBonus::Row(r) => if pattern_row == *r { 1.0 } else { 0.0 },
        WallBonus::Col(c) => {
            if wall_col_for(pattern_row, color) == Some(*c) { 1.0 } else { 0.0 }
        }
        WallBonus::Color(k) => {
            if color as usize == *k { 1.0 } else { 0.0 }
        }
    }
}

fn opponent_gift_penalty(c: &CandidateMove, state: &GameState) -> f32 {
    // Only applies when taking from a factory (other tiles go to center)
    if let MoveSource::Factory(factory_idx) = c.m.source {
        let factory = &state.factories[factory_idx];
        let mut penalty = 0.0f32;

        // Count each non-chosen color that would go to center
        for &other_color in factory.iter().filter(|&&t| t != c.m.tile) {
            let gifted: usize = factory.iter().filter(|&&t| t == other_color).count();
            // Check if any opponent can use these tiles
            for (i, player) in state.players.iter().enumerate() {
                if i == state.current_player_idx { continue; }
                for (row_idx, line) in player.pattern_lines.iter().enumerate() {
                    let row_color_matches = line.first().map_or(true, |&t| t == other_color);
                    if !row_color_matches { continue; }
                    // Check wall slot not already filled
                    let wall_col = WALL_LAYOUT[row_idx].iter().position(|&t| t == other_color);
                    if let Some(wc) = wall_col {
                        if player.wall[row_idx][wc].is_some() { continue; }
                    } else {
                        continue;
                    }
                    let space = (row_idx + 1) - line.len();
                    if space <= gifted {
                        penalty += 0.4;
                    }
                }
            }
        }
        penalty
    } else {
        0.0
    }
}

impl ConvergenceAI {
    fn score_move(&self, c: &CandidateMove, state: &GameState) -> f32 {
        let player = &state.players[state.current_player_idx];
        let wall = &player.wall;
        let cur_floor = player.floor_line.len();

        let (pattern_row, color) = match c.m.destination {
            MoveDestination::Floor => {
                let fp = floor_penalty(c.tiles_available, cur_floor);
                return -fp * 1.5;
            }
            MoveDestination::PatternLine(idx) => (idx, c.m.tile),
        };

        let overflow = c.tiles_available.saturating_sub(c.pattern_line_space);
        let waste_penalty = floor_penalty(overflow, cur_floor) * 1.5;

        // Bonus trajectory score
        let bonus_score: f32 = WallBonus::all().iter().map(|bonus| {
            let remaining = tiles_remaining_for_bonus(bonus, wall);
            let contrib = contribution_to_bonus(bonus, pattern_row, color);
            reachability(remaining) * bonus.value() * contrib
        }).sum();

        // Immediate placement score (only if completing the row)
        let placement_value = if c.completes_row {
            wall_col_for(pattern_row, color)
                .map(|col| wall_placement_score(wall, pattern_row, col))
                .unwrap_or(1.0)
        } else {
            0.0
        };

        // Efficiency bonus: exact fit
        let efficiency = if c.tiles_available == c.pattern_line_space { 1.0 } else { 0.0 };

        let gift_pen = opponent_gift_penalty(c, state);

        1.0 * bonus_score
            + 1.2 * placement_value
            - waste_penalty
            - 0.4 * gift_pen
            + 0.3 * efficiency
    }
}

impl AIAgent for ConvergenceAI {
    fn get_move(&mut self, state: &GameState) -> Option<Move> {
        let candidates = build_candidates(state);
        candidates.into_iter()
            .max_by(|a, b| {
                self.score_move(a, state)
                    .partial_cmp(&self.score_move(b, state))
                    .unwrap()
            })
            .map(|c| c.m)
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
