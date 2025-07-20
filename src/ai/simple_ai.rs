use crate::{ai::AIAgent, GameState, Move, MoveSource};

pub struct SimpleAI;

impl AIAgent for SimpleAI {
    fn get_move(&self, game_state: &GameState) -> Option<Move> {
        let legal_moves = game_state.get_legal_moves();
        if legal_moves.is_empty() {
            return None;
        }

        let current_player = &game_state.players[game_state.current_player_idx];

        let best_move = legal_moves.into_iter().max_by_key(|m| {
            let mut score: i32 = 0; // Explicitly a signed integer
            
            let tile_count = match m.source {
                MoveSource::Factory(idx) => game_state.factories[idx].iter().filter(|&&t| t == m.tile).count(),
                MoveSource::Center => game_state.center.iter().filter(|&&t| t == m.tile).count(),
            };

            if m.pattern_line_idx < 5 {
                let line = &current_player.pattern_lines[m.pattern_line_idx];
                let capacity = m.pattern_line_idx + 1;
                let space_available = capacity - line.len();

                // Cast both usize values to i32 before subtracting
                let tiles_to_floor = tile_count as i32 - space_available as i32;

                score -= tiles_to_floor * 10;

                if tile_count >= space_available {
                    score += 5;
                }
            } else {
                // Cast tile_count to i32 before multiplying
                score -= (tile_count as i32) * 10;
            }
            score
        });

        best_move
    }
}
