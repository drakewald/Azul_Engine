use azul_engine::{GameState, Move, MoveSource};
use std::io;

fn main() {
    println!("Starting Azul Game!");
    let num_players = 2;
    let mut game = GameState::new(num_players);

    let mut round_counter = 1;

    // --- MAIN GAME LOOP ---
    loop {
        println!("\n--- Starting Round {} ---", round_counter);

        // --- DRAFTING PHASE ---
        while let Some(legal_moves) = Some(game.get_legal_moves()).filter(|m| !m.is_empty()) {
            let player_idx = game.current_player_idx;
            println!("\nPlayer {}'s turn.", player_idx + 1);
            
            println!("{}", game.players[player_idx]);

            println!("Legal moves:");
            for (i, m) in legal_moves.iter().enumerate() {
                let tile_count = match m.source {
                    MoveSource::Factory(idx) => game.factories[idx].iter().filter(|&&t| t == m.tile).count(),
                    MoveSource::Center => game.center.iter().filter(|&&t| t == m.tile).count(),
                };
                println!(
                    "  {}: Take {} {:?} tile(s) from {:?}, place on row/floor {}",
                    i + 1, tile_count, m.tile, m.source, m.pattern_line_idx + 1
                );
            }

            let chosen_move: Move;
            loop {
                println!("Please enter the number of your move:");
                let mut input = String::new();
                io::stdin().read_line(&mut input).expect("Failed to read line");

                match input.trim().parse::<usize>() {
                    Ok(num) if num > 0 && num <= legal_moves.len() => {
                        chosen_move = legal_moves[num - 1].clone();
                        break;
                    }
                    _ => {
                        println!("Invalid input. Please enter a number between 1 and {}.", legal_moves.len());
                    }
                }
            }
            game.apply_move(&chosen_move);
        }

        // STEP 1: Find who starts next *before* flags are cleared.
        let next_starting_player_idx = game.players.iter().position(|p| p.has_first_player_marker)
            .unwrap_or(game.current_player_idx);

        println!("\n--- Tiling Phase ---");
        // STEP 2: Run the tiling phase, which clears the `has_first_player_marker` flags.
        for i in 0..num_players {
            game.players[i].run_tiling_phase(&mut game.discard_pile);
        }

        println!("--- End of Round {} Scores ---", round_counter);
        for (i, player) in game.players.iter().enumerate() {
            println!("Player {} score: {}", i + 1, player.score);
        }

        if game.end_game_triggered {
            println!("\nFinal round completed!");
            break; 
        }
        
        // STEP 3: Set the starting player for the next round using the index we saved in Step 1.
        game.current_player_idx = next_starting_player_idx;
        println!("\nPlayer {} will start the next round.", game.current_player_idx + 1);

        game.refill_factories();
        round_counter += 1;
    }

    // --- END OF GAME ---
    println!("\n--- Final Scoring ---");
    game.apply_end_game_scoring();
    
    for (i, player) in game.players.iter().enumerate() {
        println!("Player {} final score: {}", i + 1, player.score);
    }
}