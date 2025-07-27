use crate::{
    ai::AIAgent, GameState, Move, MoveDestination, MoveSource, PlayerBoard, Tile, WALL_LAYOUT,
};
use std::any::Any;

/// An AI that uses a series of prioritized, rule-based heuristics to select a move.
/// It plays strategically but does not look ahead more than one turn.
pub struct HeuristicAI;

impl AIAgent for HeuristicAI {
    /// Selects a move by evaluating heuristics in a specific order of priority.
    /// The `&mut self` is included to conform to the AIAgent trait, but this AI is stateless.
    fn get_move(&mut self, game_state: &GameState) -> Option<Move> {
        // Efficiency: Legal moves are generated once and passed to the helper functions.
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

        // Priority 4: Special logic for the first move of the round
        if is_first_move_of_round(game_state) {
            if let Some(best_move) = find_first_move_priority(game_state, &legal_moves) {
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

// --- Heuristic Functions (Updated to accept `&[Move]`) ---

fn find_big_grab_move(game_state: &GameState, legal_moves: &[Move]) -> Option<Move> {
    let current_player = &game_state.players[game_state.current_player_idx];
    let mut best_option: Option<Move> = None;
    let mut best_row_index = -1;

    // Type Safety: Iterate through moves and only consider placements on a PatternLine.
    for m in legal_moves.iter() {
        if let MoveDestination::PatternLine(idx) = m.destination {
            let tile_count = count_tiles_at_source(game_state, &m.source, m.tile);
            if tile_count >= 3 {
                let line = &current_player.pattern_lines[idx];
                let space_available = (idx + 1) - line.len();
                if tile_count == space_available {
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
    // Type Safety: Filter for moves from the center targeting a PatternLine.
    let center_moves: Vec<_> = legal_moves.iter()
        .filter(|m| m.source == MoveSource::Center && matches!(m.destination, MoveDestination::PatternLine(_)))
        .collect();

    if center_moves.is_empty() { return None; }
    let current_player = &game_state.players[game_state.current_player_idx];

    for m in &center_moves {
        if let MoveDestination::PatternLine(idx) = m.destination {
            let tile_count = count_tiles_at_source(game_state, &m.source, m.tile);
            if tile_count >= 2 {
                let line = &current_player.pattern_lines[idx];
                if line.len() + tile_count == idx + 1 { return Some((*m).clone()); }
            }
        }
    }

    let best_column_placement = center_moves.iter()
        .filter_map(|m| if let MoveDestination::PatternLine(idx) = m.destination { Some((m, idx)) } else { None })
        .filter(|(m, idx)| count_tiles_at_source(game_state, &m.source, m.tile) >= 2 && *idx >= 2)
        .max_by_key(|(m, idx)| calculate_column_progress(current_player, *idx, m.tile));
    if let Some((best_move, _)) = best_column_placement { return Some((*best_move).clone()); }

    let best_single_tile_completion = center_moves.iter()
        .filter_map(|m| if let MoveDestination::PatternLine(idx) = m.destination { Some((m, idx)) } else { None })
        .filter(|(m, idx)| {
            count_tiles_at_source(game_state, &m.source, m.tile) == 1 &&
            current_player.pattern_lines[*idx].len() + 1 == *idx + 1
        })
        .max_by_key(|(_, idx)| *idx);

    if let Some((best_move, _)) = best_single_tile_completion {
        return Some((*best_move).clone());
    }

    None
}

fn find_first_move_priority(game_state: &GameState, legal_moves: &[Move]) -> Option<Move> {
    let current_player = &game_state.players[game_state.current_player_idx];

    // Type Safety: Filter specifically for moves to PatternLine index 1.
    legal_moves.iter()
        .filter(|m| m.destination == MoveDestination::PatternLine(1))
        .filter(|m| {
            let tile_count = count_tiles_at_source(game_state, &m.source, m.tile);
            let line = &current_player.pattern_lines[1];
            tile_count >= 2 - line.len()
        })
        .max_by_key(|m| calculate_adjacency_score(current_player, 1, m.tile))
        .cloned()
}

fn find_best_general_move(game_state: &GameState, legal_moves: &[Move]) -> Option<Move> {
    let current_player = &game_state.players[game_state.current_player_idx];

    legal_moves.iter().max_by_key(|m| {
        let mut score: i32 = 0;
        let tile_count = count_tiles_at_source(game_state, &m.source, m.tile);

        // Type Safety: Use a match statement to handle different destinations.
        match m.destination {
            MoveDestination::PatternLine(idx) => {
                let line = &current_player.pattern_lines[idx];
                let space_available = (idx + 1) - line.len();
                let tiles_placed = tile_count.min(space_available);
                let tiles_to_floor = (tile_count as i32 - space_available as i32).max(0);

                score -= tiles_to_floor * 20;
                score += (tiles_placed as i32) * 10;
                if tile_count >= space_available {
                    score += 15;
                }
                score += calculate_adjacency_score(current_player, idx, m.tile) * 5;

                if let Some(col_idx) = WALL_LAYOUT[idx].iter().position(|&t| t == m.tile) {
                    if col_idx > 0 { score += calculate_column_progress_by_index(current_player, col_idx - 1) * 3; }
                    if col_idx < 4 { score += calculate_column_progress_by_index(current_player, col_idx + 1) * 3; }
                }
            }
            MoveDestination::Floor => {
                // The `-1` ensures this is always slightly worse than any non-flooring move.
                score = -((tile_count as i32) * 20) - 1;
            }
        }
        score
    }).cloned()
}

// --- Utility Functions (Unchanged but used by the refactored code) ---

fn is_first_move_of_round(game_state: &GameState) -> bool {
    game_state.center.is_empty()
}

fn count_tiles_at_source(game_state: &GameState, source: &MoveSource, tile: Tile) -> usize {
    match source {
        MoveSource::Factory(idx) => game_state.factories[*idx].iter().filter(|&&t| t == tile).count(),
        MoveSource::Center => game_state.center.iter().filter(|&&t| t == tile).count(),
    }
}

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