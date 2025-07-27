use crate::{ai::AIAgent, GameState, Move};
use std::any::Any; // Add this import

// The HumanAgent is a placeholder for web UI interaction.
pub struct HumanAgent;

impl AIAgent for HumanAgent {
    // In a headless simulation, a human can't make a move, so it does nothing.
    fn get_move(&mut self, _game_state: &GameState) -> Option<Move> {
        None
    }

    // NEW: Added the required `as_any` method implementation.
    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
