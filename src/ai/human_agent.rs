use crate::{ai::AIAgent, GameState, Move};

pub struct HumanAgent;

impl AIAgent for HumanAgent {
    // This will not be called for a human player in the web interface.
    fn get_move(&mut self, _game_state: &GameState) -> Option<Move> {
        None
    }
}