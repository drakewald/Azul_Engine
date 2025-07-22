use azul_engine::{GameState, Move, MoveDestination};
use std::io;

fn main() {
    println!("Starting Azul Game!");
    // In a full app, you might ask the user for the number of players.
    let num_players = 2;
    let mut game = GameState::new(num_players);
    let mut round_counter = 1;

    // --- Main Game Loop ---
    loop {
        println!("\n\n--- Starting Round {} ---", round_counter);

        // --- Drafting Phase ---
        // This loop continues as long as there are tiles to be taken.
        while !game.is_round_over() {
            let player_idx = game.current_player_idx;
            println!("\nPlayer {}'s turn.", player_idx + 1);
            println!("{}", game.players[player_idx]);

            let legal_moves = game.get_legal_moves();
            if legal_moves.is_empty() {
                // This can happen if one player takes the last tiles, ending the
                // drafting phase before other players have had their turn.
                break;
            }

            println!("Legal moves:");
            for (i, m) in legal_moves.iter().enumerate() {
                // Correctly display the move's destination.
                let dest_str = match m.destination {
                    MoveDestination::PatternLine(idx) => format!("pattern line {}", idx + 1),
                    MoveDestination::Floor => "the floor".to_string(),
                };
                println!(
                    "  {}: Take {:?} from {:?}, place on {}",
                    i + 1, m.tile, m.source, dest_str
                );
            }

            let chosen_move = get_player_move(&legal_moves);
            game.apply_move(&chosen_move);
        }

        // --- Tiling Phase ---
        println!("\n--- Tiling Phase ---");
        // The run_tiling_phase method now handles all end-of-round logic.
        game.run_tiling_phase();

        println!("--- End of Round {} Scores ---", round_counter);
        for (i, player) in game.players.iter().enumerate() {
            println!("Player {} score: {}", i + 1, player.score);
        }
        
        // Check if the game's end condition was triggered during tiling.
        if game.end_game_triggered {
            println!("\nFinal round completed!");
            break; 
        }

        // --- Round Cleanup ---
        println!("\nPlayer {} will start the next round.", game.current_player_idx + 1);
        game.refill_factories();
        round_counter += 1;
    }

    // --- End of Game Scoring ---
    println!("\n--- Final Scoring ---");
    game.apply_end_game_scoring();
    
    for (i, player) in game.players.iter().enumerate() {
        println!("Player {} final score: {}", i + 1, player.score);
    }
}

/// Prompts the user to select a move from the provided list.
fn get_player_move(legal_moves: &[Move]) -> Move {
    loop {
        println!("Please enter the number of your move:");
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");

        match input.trim().parse::<usize>() {
            Ok(num) if num > 0 && num <= legal_moves.len() => {
                // The chosen move is cloned from the list of legal moves.
                return legal_moves[num - 1].clone();
            }
            _ => {
                println!("Invalid input. Please enter a number between 1 and {}.", legal_moves.len());
            }
        }
    }
}