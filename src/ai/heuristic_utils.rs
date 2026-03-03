use crate::{GameState, Move, MoveDestination, MoveSource, Tile, WALL_LAYOUT};

const FLOOR_PENALTIES: [f32; 7] = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0];

pub struct CandidateMove {
    pub m: Move,
    pub tiles_available: usize,
    pub pattern_line_space: usize,
    pub completes_row: bool,
}

/// Wall column where (pattern_row, color) would be placed.
pub fn wall_col_for(pattern_row: usize, color: Tile) -> Option<usize> {
    WALL_LAYOUT[pattern_row].iter().position(|&t| t == color)
}

/// Count tiles of `tile` at `source`.
pub fn count_at_source(state: &GameState, source: &MoveSource, tile: Tile) -> usize {
    match source {
        MoveSource::Factory(i) => state.factories[*i].iter().filter(|&&t| t == tile).count(),
        MoveSource::Center => state.center.iter().filter(|&&t| t == tile).count(),
    }
}

/// Actual H+V run-length placement score for placing at (wall_row, wall_col).
pub fn wall_placement_score(wall: &[Vec<Option<Tile>>], wall_row: usize, wall_col: usize) -> f32 {
    let mut h = 1u32;
    for dc in [1i32, -1] {
        let mut c = wall_col as i32 + dc;
        while c >= 0 && c < 5 && wall[wall_row][c as usize].is_some() {
            h += 1;
            c += dc;
        }
    }
    let mut v = 1u32;
    for dr in [1i32, -1] {
        let mut r = wall_row as i32 + dr;
        while r >= 0 && r < 5 && wall[r as usize][wall_col].is_some() {
            v += 1;
            r += dr;
        }
    }
    if h > 1 && v > 1 {
        (h + v) as f32
    } else if h > 1 {
        h as f32
    } else {
        v as f32
    }
}

/// Floor penalty using the game's actual schedule, offset by tiles already on floor.
pub fn floor_penalty(overflow: usize, current_floor: usize) -> f32 {
    (0..overflow)
        .map(|i| FLOOR_PENALTIES[(current_floor + i).min(6)])
        .sum()
}

/// Enrich all legal moves with precomputed metadata.
pub fn build_candidates(state: &GameState) -> Vec<CandidateMove> {
    let player = &state.players[state.current_player_idx];
    state.get_legal_moves().into_iter().map(|m| {
        let tiles_available = count_at_source(state, &m.source, m.tile);
        let (pattern_line_space, completes_row) = match m.destination {
            MoveDestination::PatternLine(idx) => {
                let space = (idx + 1) - player.pattern_lines[idx].len();
                (space, tiles_available >= space)
            }
            MoveDestination::Floor => (0, false),
        };
        CandidateMove { m, tiles_available, pattern_line_space, completes_row }
    }).collect()
}
