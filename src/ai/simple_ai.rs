use crate::{ai::AIAgent, GameState, Move, MoveDestination};
use std::any::Any;

pub struct SimpleAI;

impl AIAgent for SimpleAI {
    fn get_move(&mut self, game_state: &GameState) -> Option<Move> {
        let legal_moves = game_state.get_legal_moves();
        if legal_moves.is_empty() {
            return None;
        }

        // The simple AI just finds the first move that doesn't go directly to the floor.
        legal_moves.into_iter().find(|m| {
            matches!(m.destination, MoveDestination::PatternLine(_))
        }).or_else(|| {
            // If all moves go to the floor, just take the first one available.
            game_state.get_legal_moves().into_iter().next()
        })
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}