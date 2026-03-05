use crate::{
    ai::AIAgent, GameState, Move, MoveDestination, MoveSource, PlayerBoard, Tile, WALL_LAYOUT,
};
use crate::ai::heuristic_utils::{
    count_at_source, floor_penalty, wall_placement_score,
};
use std::any::Any;

/// An AI that uses a series of prioritized, rule-based heuristics to select a move.
/// It plays strategically but does not look ahead more than one turn.
pub struct HeuristicAI;

impl AIAgent for HeuristicAI {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move> {
        let legal_moves = game_state.get_legal_moves();
        if legal_moves.is_empty() {
            return None;
        }

        // Priority 1: Big Grab
        if let Some(best_move) = find_big_grab_move(game_state, &legal_moves) {
            return Some(best_move);
        }

        // Priority 2 & 3: Take from center
        if game_state.first_player_marker_in_center {
            if let Some(best_move) = find_untouched_center_move(game_state, &legal_moves) {
                return Some(best_move);
            }
        }

        // Main Heuristic: Find the best general-purpose move
        find_best_general_move(game_state, &legal_moves)
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

// --- Heuristic Functions ---

fn find_big_grab_move(game_state: &GameState, legal_moves: &[Move]) -> Option<Move> {
    let current_player = &game_state.players[game_state.current_player_idx];
    let mut best_option: Option<Move> = None;
    let mut best_row_index = -1;

    for m in legal_moves.iter() {
        if let MoveDestination::PatternLine(idx) = m.destination {
            let tile_count = count_at_source(game_state, &m.source, m.tile);
            if tile_count >= 3 {
                let line = &current_player.pattern_lines[idx];
                let space_available = (idx + 1) - line.len();
                if tile_count == space_available {
                    // Change B: require placement score >= 2.0 (not an isolated tile)
                    if let Some(wall_col) = WALL_LAYOUT[idx].iter().position(|&t| t == m.tile) {
                        let score = wall_placement_score(&current_player.wall, idx, wall_col);
                        if score < 2.0 {
                            continue;
                        }
                    }
                    if (idx as i32) > best_row_index {
                        best_row_index = idx as i32;
                        best_option = Some(m.clone());
                    }
                }
            }
        }
    }
    best_option
}

fn find_untouched_center_move(game_state: &GameState, legal_moves: &[Move]) -> Option<Move> {
    let center_moves: Vec<_> = legal_moves.iter()
        .filter(|m| m.source == MoveSource::Center && matches!(m.destination, MoveDestination::PatternLine(_)))
        .collect();

    if center_moves.is_empty() { return None; }
    let current_player = &game_state.players[game_state.current_player_idx];

    for m in &center_moves {
        if let MoveDestination::PatternLine(idx) = m.destination {
            let tile_count = count_at_source(game_state, &m.source, m.tile);
            if tile_count >= 2 {
                let line = &current_player.pattern_lines[idx];
                if line.len() + tile_count == idx + 1 { return Some((*m).clone()); }
            }
        }
    }

    let best_column_placement = center_moves.iter()
        .filter_map(|m| if let MoveDestination::PatternLine(idx) = m.destination { Some((m, idx)) } else { None })
        .filter(|(m, idx)| count_at_source(game_state, &m.source, m.tile) >= 2 && *idx >= 2)
        .max_by_key(|(m, idx)| calculate_column_progress(current_player, *idx, m.tile));
    if let Some((best_move, _)) = best_column_placement { return Some((*best_move).clone()); }

    let best_single_tile_completion = center_moves.iter()
        .filter_map(|m| if let MoveDestination::PatternLine(idx) = m.destination { Some((m, idx)) } else { None })
        .filter(|(m, idx)| {
            count_at_source(game_state, &m.source, m.tile) == 1 &&
            current_player.pattern_lines[*idx].len() + 1 == *idx + 1
        })
        .max_by_key(|(_, idx)| *idx);

    if let Some((best_move, _)) = best_single_tile_completion {
        return Some((*best_move).clone());
    }

    None
}

fn find_best_general_move(game_state: &GameState, legal_moves: &[Move]) -> Option<Move> {
    let current_player = &game_state.players[game_state.current_player_idx];

    legal_moves.iter().max_by_key(|m| {
        let mut score: i32 = 0;
        let tile_count = count_at_source(game_state, &m.source, m.tile);

        match m.destination {
            MoveDestination::PatternLine(idx) => {
                let line = &current_player.pattern_lines[idx];
                let space_available = (idx + 1) - line.len();
                let tiles_placed = tile_count.min(space_available);

                // Change C: accurate floor penalty using the game schedule
                let overflow = (tile_count as i32 - space_available as i32).max(0) as usize;
                let cur_floor = current_player.floor_line.len();
                score -= (floor_penalty(overflow, cur_floor) * 10.0) as i32;

                score += (tiles_placed as i32) * 10;
                if tile_count >= space_available {
                    score += 15;
                }

                if let Some(wall_col) = WALL_LAYOUT[idx].iter().position(|&t| t == m.tile) {
                    // Change D: H+V score for completing moves, adjacency for non-completing
                    if tile_count >= space_available {
                        score += (wall_placement_score(&current_player.wall, idx, wall_col) * 5.0) as i32;
                    } else {
                        score += calculate_adjacency_score(current_player, idx, m.tile);
                    }

                    // Change E: 1-away end-game bonus signals
                    let row_filled = (0..5).filter(|&c| current_player.wall[idx][c].is_some()).count();
                    if row_filled == 4 { score += 8; }

                    let col_filled = (0..5).filter(|&r| current_player.wall[r][wall_col].is_some()).count();
                    if col_filled == 4 { score += 25; }

                    let color_on_wall = (0..5)
                        .filter(|&r| {
                            let c = WALL_LAYOUT[r].iter().position(|&t| t == m.tile).unwrap_or(5);
                            c < 5 && current_player.wall[r][c].is_some()
                        })
                        .count();
                    if color_on_wall == 4 { score += 35; }

                    if wall_col > 0 { score += calculate_column_progress_by_index(current_player, wall_col - 1) * 3; }
                    if wall_col < 4 { score += calculate_column_progress_by_index(current_player, wall_col + 1) * 3; }
                }
            }
            MoveDestination::Floor => {
                score = -((tile_count as i32) * 20) - 1;
            }
        }
        score
    }).cloned()
}

// --- Utility Functions ---

fn calculate_column_progress(player: &PlayerBoard, row_idx: usize, tile: Tile) -> i32 {
    if let Some(col_idx) = WALL_LAYOUT[row_idx].iter().position(|&t| t == tile) {
        return calculate_column_progress_by_index(player, col_idx);
    }
    0
}

fn calculate_column_progress_by_index(player: &PlayerBoard, col_idx: usize) -> i32 {
    (0..5).filter(|&r| player.wall[r][col_idx].is_some()).count() as i32
}

fn calculate_adjacency_score(player: &PlayerBoard, row_idx: usize, tile: Tile) -> i32 {
    if let Some(col_idx) = WALL_LAYOUT[row_idx].iter().position(|&t| t == tile) {
        let mut score = 0;
        if col_idx > 0 && player.wall[row_idx][col_idx - 1].is_some() { score += 1; }
        if col_idx < 4 && player.wall[row_idx][col_idx + 1].is_some() { score += 1; }
        if row_idx > 0 && player.wall[row_idx - 1][col_idx].is_some() { score += 1; }
        if row_idx < 4 && player.wall[row_idx + 1][col_idx].is_some() { score += 1; }
        return score;
    }
    0
}
