use crate::{GameState, Move, MoveDestination, Tile};
use crate::ai::{AIAgent, heuristic_utils::{build_candidates, floor_penalty, wall_col_for, wall_placement_score, CandidateMove}};
use std::any::Any;

pub struct GreedyAI;

impl AIAgent for GreedyAI {
    fn get_move(&mut self, state: &GameState) -> Option<Move> {
        let candidates = build_candidates(state);
        let wall = &state.players[state.current_player_idx].wall;
        let cur_floor = state.players[state.current_player_idx].floor_line.len();
        candidates.into_iter()
            .max_by(|a, b| {
                score_greedy(a, wall, cur_floor)
                    .partial_cmp(&score_greedy(b, wall, cur_floor))
                    .unwrap()
            })
            .map(|c| c.m)
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

fn score_greedy(c: &CandidateMove, wall: &[Vec<Option<Tile>>], cur_floor: usize) -> f32 {
    match c.m.destination {
        MoveDestination::Floor => {
            -floor_penalty(c.tiles_available, cur_floor) - 0.1
        }
        MoveDestination::PatternLine(idx) => {
            let overflow = c.tiles_available.saturating_sub(c.pattern_line_space);
            let mut score = if c.completes_row {
                wall_col_for(idx, c.m.tile)
                    .map(|col| wall_placement_score(wall, idx, col))
                    .unwrap_or(1.0)
            } else {
                c.tiles_available.min(c.pattern_line_space) as f32 * 0.1
            };
            score -= floor_penalty(overflow, cur_floor);
            score
        }
    }
}
