use crate::{ai::AIAgent, GameState, Move, MoveSource, PlayerBoard, Tile, WALL_LAYOUT};

pub struct HeuristicAI;

impl AIAgent for HeuristicAI {
    fn get_move(&self, game_state: &GameState) -> Option<Move> {
        // Priority 1: Big Grab to complete a high-value row
        if let Some(best_move) = find_big_grab_move(game_state) {
            return Some(best_move);
        }

        // Priority 2 & 3: Take from center if it's untouched
        if game_state.first_player_marker_in_center {
            if let Some(best_move) = find_untouched_center_move(game_state) {
                return Some(best_move);
            }
        }
        
        // Priority 4: Special logic for the first move of the round
        if is_first_move_of_round(game_state) {
            if let Some(best_move) = find_first_move_priority(game_state) {
                return Some(best_move);
            }
        }

        // Main Heuristic: Find the best general-purpose move
        find_best_general_move(game_state)
    }
}

// Helper to check if it's the first move of a round
fn is_first_move_of_round(game_state: &GameState) -> bool {
    game_state.center.is_empty()
}

// --- Heuristic Functions in Order of Priority ---

fn find_big_grab_move(game_state: &GameState) -> Option<Move> {
    let legal_moves = game_state.get_legal_moves();
    let current_player = &game_state.players[game_state.current_player_idx];
    let mut best_option: Option<Move> = None;
    let mut best_row_index = -1;
    for m in legal_moves.iter().filter(|m| m.pattern_line_idx < 5) {
        let tile_count = count_tiles_at_source(game_state, &m.source, m.tile);
        if tile_count >= 3 {
            let line = &current_player.pattern_lines[m.pattern_line_idx];
            let space_available = (m.pattern_line_idx + 1) - line.len();
            if tile_count == space_available {
                if (m.pattern_line_idx as i32) > best_row_index {
                    best_row_index = m.pattern_line_idx as i32;
                    best_option = Some(m.clone());
                }
            }
        }
    }
    best_option
}

fn find_untouched_center_move(game_state: &GameState) -> Option<Move> {
    let center_moves: Vec<_> = game_state.get_legal_moves().into_iter()
        .filter(|m| m.source == MoveSource::Center && m.pattern_line_idx < 5).collect();
    if center_moves.is_empty() { return None; }
    let current_player = &game_state.players[game_state.current_player_idx];

    for m in &center_moves {
        let tile_count = count_tiles_at_source(game_state, &m.source, m.tile);
        if tile_count >= 2 {
            let line = &current_player.pattern_lines[m.pattern_line_idx];
            if line.len() + tile_count == m.pattern_line_idx + 1 { return Some(m.clone()); }
        }
    }
    
    let best_column_placement = center_moves.iter()
        .filter(|m| count_tiles_at_source(game_state, &m.source, m.tile) >= 2 && m.pattern_line_idx >= 2)
        .max_by_key(|m| calculate_column_progress(current_player, m.pattern_line_idx, m.tile));
    if let Some(best_move) = best_column_placement { return Some(best_move.clone()); }

    let best_single_tile_completion = center_moves.iter()
        .filter(|m| {
            count_tiles_at_source(game_state, &m.source, m.tile) == 1 &&
            current_player.pattern_lines[m.pattern_line_idx].len() + 1 == m.pattern_line_idx + 1
        })
        .max_by_key(|m| m.pattern_line_idx);

    if let Some(best_move) = best_single_tile_completion {
        return Some(best_move.clone());
    }

    None
}

fn find_first_move_priority(game_state: &GameState) -> Option<Move> {
    let legal_moves = game_state.get_legal_moves();
    let current_player = &game_state.players[game_state.current_player_idx];

    legal_moves.into_iter()
        .filter(|m| {
            if m.pattern_line_idx != 1 { return false; }
            let tile_count = count_tiles_at_source(game_state, &m.source, m.tile);
            let line = &current_player.pattern_lines[m.pattern_line_idx];
            tile_count >= (m.pattern_line_idx + 1) - line.len()
        })
        .max_by_key(|m| calculate_adjacency_score(current_player, m.pattern_line_idx, m.tile))
}

fn find_best_general_move(game_state: &GameState) -> Option<Move> {
    let legal_moves = game_state.get_legal_moves();
    if legal_moves.is_empty() { return None; }

    let current_player = &game_state.players[game_state.current_player_idx];

    legal_moves.into_iter().max_by_key(|m| {
        let mut score: i32 = 0;
        let tile_count = count_tiles_at_source(game_state, &m.source, m.tile);

        if m.pattern_line_idx < 5 {
            let line = &current_player.pattern_lines[m.pattern_line_idx];
            let space_available = (m.pattern_line_idx + 1) - line.len();
            
            let tiles_placed = tile_count.min(space_available);
            let tiles_to_floor = (tile_count as i32 - space_available as i32).max(0);

            score -= tiles_to_floor * 20;
            score += (tiles_placed as i32) * 10;
            if tile_count >= space_available {
                score += 15;
            }
            score += calculate_adjacency_score(current_player, m.pattern_line_idx, m.tile) * 5;

            if let Some(col_idx) = WALL_LAYOUT[m.pattern_line_idx].iter().position(|&t| t == m.tile) {
                if col_idx > 0 {
                    score += calculate_column_progress_by_index(current_player, col_idx - 1) * 3;
                }
                if col_idx < 4 {
                    score += calculate_column_progress_by_index(current_player, col_idx + 1) * 3;
                }
            }

        } else {
            score -= (tile_count as i32) * 20 - 1;
        }
        score
    })
}

// --- Utility Functions ---

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
    let mut score = 0;
    for r in 0..5 {
        if player.wall[r][col_idx].is_some() {
            score += 1;
        }
    }
    score
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