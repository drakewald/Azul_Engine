use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fmt;
use wasm_bindgen::prelude::*;

// --- Module Declarations ---
mod ai;

// --- Imports ---
use ai::{AIAgent, simple_ai::SimpleAI, heuristic_ai::HeuristicAI, mcts_ai::MctsAI};

// --- Core Game Structs (Pure Rust, no Wasm annotations) ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tile {
    Blue,
    Yellow,
    Red,
    Black,
    White,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerBoard {
    pub score: u32,
    pub pattern_lines: Vec<Vec<Tile>>, 
    pub wall: Vec<Vec<Option<Tile>>>,
    pub floor_line: Vec<Tile>,
    pub has_first_player_marker: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub players: Vec<PlayerBoard>,
    pub factories: Vec<Vec<Tile>>,
    pub center: Vec<Tile>,
    pub tile_bag: Vec<Tile>,
    pub discard_pile: Vec<Tile>,
    pub current_player_idx: usize,
    pub first_player_marker_in_center: bool,
    pub end_game_triggered: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveSource {
    Factory(usize),
    Center,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Move {
    pub source: MoveSource,
    pub tile: Tile,
    pub pattern_line_idx: usize,
}

// NEW: PlayerType enum to distinguish different controllers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    SimpleAI,
    HeuristicAI,
    MctsAI,
}


const WALL_LAYOUT: [[Tile; 5]; 5] = [
    [Tile::Blue, Tile::Yellow, Tile::Red, Tile::Black, Tile::White],
    [Tile::White, Tile::Blue, Tile::Yellow, Tile::Red, Tile::Black],
    [Tile::Black, Tile::White, Tile::Blue, Tile::Yellow, Tile::Red],
    [Tile::Red, Tile::Black, Tile::White, Tile::Blue, Tile::Yellow],
    [Tile::Yellow, Tile::Red, Tile::Black, Tile::White, Tile::Blue],
];

// --- Native Rust Implementation (for main.rs) ---

impl GameState {
   pub fn new(num_players: usize) -> Self {
        let players = (0..num_players).map(|_| PlayerBoard::new()).collect();
        let all_colors = [Tile::Blue, Tile::Yellow, Tile::Red, Tile::Black, Tile::White];
        let mut tile_bag: Vec<Tile> = all_colors
            .iter()
            .flat_map(|&tile| std::iter::repeat(tile).take(20))
            .collect();
        tile_bag.shuffle(&mut thread_rng());
        let num_factories = match num_players {
            2 => 5, 3 => 7, 4 => 9,
            _ => panic!("Invalid number of players. Must be 2, 3, or 4."),
        };
        let mut game_state = Self {
            players,
            factories: vec![vec![]; num_factories],
            center: Vec::new(),
            tile_bag,
            discard_pile: Vec::new(),
            current_player_idx: 0,
            first_player_marker_in_center: true,
            end_game_triggered: false,
        };
        game_state.refill_factories();
        game_state
    }

    pub fn refill_factories(&mut self) {
        for factory in self.factories.iter_mut() {
            factory.clear(); 
            for _ in 0..4 {
                if self.tile_bag.is_empty() {
                    if self.discard_pile.is_empty() { break; }
                    std::mem::swap(&mut self.tile_bag, &mut self.discard_pile);
                    self.tile_bag.shuffle(&mut thread_rng());
                }
                if let Some(tile) = self.tile_bag.pop() {
                    factory.push(tile);
                }
            }
        }
        self.center.clear();
        self.first_player_marker_in_center = true;
    }

    pub fn get_legal_moves(&self) -> Vec<Move> {
        let mut legal_moves = Vec::new();
        let current_player_board = &self.players[self.current_player_idx];
        let mut generate_moves_for_source = |source: MoveSource, tiles: &[Tile]| {
            let unique_tiles: HashSet<_> = tiles.iter().cloned().collect();
            for tile in unique_tiles {
                for i in 0..5 {
                    if current_player_board.is_placement_valid(i, tile) {
                        legal_moves.push(Move {
                            source: source.clone(),
                            tile,
                            pattern_line_idx: i,
                        });
                    }
                }
                legal_moves.push(Move {
                    source: source.clone(),
                    tile,
                    pattern_line_idx: 5,
                });
            }
        };
        for (i, factory) in self.factories.iter().enumerate() {
            generate_moves_for_source(MoveSource::Factory(i), factory);
        }
        generate_moves_for_source(MoveSource::Center, &self.center);
        legal_moves
    }

    pub fn apply_move(&mut self, player_move: &Move) {
        let player = &mut self.players[self.current_player_idx];
        let source_tiles = match player_move.source {
            MoveSource::Factory(idx) => std::mem::take(&mut self.factories[idx]),
            MoveSource::Center => std::mem::take(&mut self.center),
        };
        let (mut taken_tiles, remaining): (Vec<Tile>, Vec<Tile>) = 
            source_tiles.into_iter().partition(|&t| t == player_move.tile);
        if let MoveSource::Factory(_) = player_move.source {
            self.center.extend(remaining);
        } else {
            self.center = remaining;
            if self.first_player_marker_in_center {
                self.first_player_marker_in_center = false;
                player.has_first_player_marker = true;
            }
        }
        player.place_tiles(&mut taken_tiles, player_move.pattern_line_idx);
        if player_move.pattern_line_idx < 5 {
            if !self.end_game_triggered && player.will_complete_horizontal_row(player_move.pattern_line_idx) {
                self.end_game_triggered = true;
            }
        }
        self.current_player_idx = (self.current_player_idx + 1) % self.players.len();
    }

    pub fn is_game_over(&self) -> bool {
        self.players.iter().any(|p| p.has_complete_row())
    }

    pub fn apply_end_game_scoring(&mut self) {
        for player in self.players.iter_mut() {
            let bonus_score = player.calculate_end_game_bonuses();
            player.score += bonus_score;
        }
    }
}

impl PlayerBoard {
    pub fn new() -> Self {
        Self {
            score: 0,
            pattern_lines: vec![
                Vec::with_capacity(1), Vec::with_capacity(2),
                Vec::with_capacity(3), Vec::with_capacity(4),
                Vec::with_capacity(5),
            ],
            wall: vec![vec![None; 5]; 5],
            floor_line: Vec::new(),
            has_first_player_marker: false,
        }
    }

    pub fn has_complete_row(&self) -> bool {
        self.wall.iter().any(|row| row.iter().all(|tile| tile.is_some()))
    }

   fn will_complete_horizontal_row(&self, pattern_line_idx: usize) -> bool {
        if self.pattern_lines[pattern_line_idx].len() != pattern_line_idx + 1 {
            return false;
        }
        let wall_row_to_check = &self.wall[pattern_line_idx];
        let tiles_on_wall_row = wall_row_to_check.iter().filter(|tile| tile.is_some()).count();
        tiles_on_wall_row == 4
    }

    pub fn place_tiles(&mut self, tiles_to_place: &mut Vec<Tile>, pattern_line_idx: usize) {
        if pattern_line_idx == 5 {
            self.floor_line.append(tiles_to_place);
            return;
        }
        let capacity = pattern_line_idx + 1;
        let pattern_line = &mut self.pattern_lines[pattern_line_idx];
        while !tiles_to_place.is_empty() && pattern_line.len() < capacity {
            pattern_line.push(tiles_to_place.pop().unwrap());
        }
        self.floor_line.append(tiles_to_place);
    }

    pub fn is_placement_valid(&self, pattern_line_idx: usize, tile_color: Tile) -> bool {
        let line = &self.pattern_lines[pattern_line_idx];
        let capacity = pattern_line_idx + 1;
        if line.len() >= capacity { return false; }
        if !line.is_empty() && line[0] != tile_color { return false; }

        if let Some(col_idx) = WALL_LAYOUT[pattern_line_idx].iter().position(|&t| t == tile_color) {
            if self.wall[pattern_line_idx][col_idx].is_some() {
                return false;
            }
        } else {
            return false;
        }
        true
    }

    pub fn run_tiling_phase(&mut self, discard_pile: &mut Vec<Tile>) -> bool {
        for row_idx in 0..5 {
            let capacity = row_idx + 1;
            if self.pattern_lines[row_idx].len() == capacity {
                let tile_color = self.pattern_lines[row_idx][0];
                if let Some(col_idx) = WALL_LAYOUT[row_idx].iter().position(|&t| t == tile_color) {
                    if self.wall[row_idx][col_idx].is_none() {
                        self.wall[row_idx][col_idx] = Some(tile_color);
                        let placement_score = self.calculate_placement_score(row_idx, col_idx);
                        self.score += placement_score;
                        let pattern_line = &mut self.pattern_lines[row_idx];
                        discard_pile.push(pattern_line.pop().unwrap());
                        discard_pile.append(pattern_line);
                    }
                }
            }
        }
        let floor_penalty_values = [1, 1, 2, 2, 2, 3, 3];
        let num_floor_tiles = self.floor_line.len() + if self.has_first_player_marker { 1 } else { 0 };
        let mut penalty: u32 = 0;
        if num_floor_tiles > 0 {
            penalty = floor_penalty_values[..num_floor_tiles.min(7)].iter().sum();
        }
        discard_pile.append(&mut self.floor_line);
        self.has_first_player_marker = false;
        self.score = self.score.saturating_sub(penalty);
        self.has_complete_row()
    }

    fn calculate_placement_score(&self, row: usize, col: usize) -> u32 {
        let mut horizontal_score = 1;
        for i in (0..col).rev() { if self.wall[row][i].is_some() { horizontal_score += 1; } else { break; } }
        for i in (col + 1)..5 { if self.wall[row][i].is_some() { horizontal_score += 1; } else { break; } }
        let mut vertical_score = 1;
        for i in (0..row).rev() { if self.wall[i][col].is_some() { vertical_score += 1; } else { break; } }
        for i in (row + 1)..5 { if self.wall[i][col].is_some() { vertical_score += 1; } else { break; } }
        if horizontal_score > 1 && vertical_score > 1 {
            horizontal_score + vertical_score
        } else {
            horizontal_score.max(vertical_score)
        }
    }

    pub fn calculate_end_game_bonuses(&self) -> u32 {
        let mut bonus_score = 0;
        for row in 0..5 { if self.wall[row].iter().all(|tile| tile.is_some()) { bonus_score += 2; } }
        for col in 0..5 { if (0..5).all(|row| self.wall[row][col].is_some()) { bonus_score += 7; } }
        for color_to_check in [Tile::Blue, Tile::Yellow, Tile::Red, Tile::Black, Tile::White] {
            let count = self.wall.iter().flatten().filter(|&&tile| tile == Some(color_to_check)).count();
            if count == 5 { bonus_score += 10; }
        }
        bonus_score
    }
}

impl fmt::Display for PlayerBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Score: {}", self.score)?;
        writeln!(f, "------------------------------------")?;
        writeln!(f, "Pattern Lines         Wall")?;
        for i in 0..5 {
            let capacity = i + 1;
            let line = &self.pattern_lines[i];
            for _ in 0..(5 - capacity) { write!(f, "  ")?; }
            for _ in 0..(capacity - line.len()) { write!(f, "[_] ")?; }
            for tile in line { write!(f, "[{}] ", tile_to_char(*tile))?; }
            write!(f, "   |   ")?;
            for tile_option in &self.wall[i] {
                match tile_option {
                    Some(tile) => write!(f, "[{}] ", tile_to_char(*tile))?,
                    None => write!(f, "[.] ")?,
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "------------------------------------")?;
        write!(f, "Floor Line: ")?;
        if self.has_first_player_marker { write!(f, "[1] ")?; }
        for tile in &self.floor_line { write!(f, "[{}] ", tile_to_char(*tile))?; }
        writeln!(f)
    }
}

fn tile_to_char(tile: Tile) -> char {
    match tile {
        Tile::Blue => 'B', Tile::Yellow => 'Y',
        Tile::Red => 'R', Tile::Black => 'K',
        Tile::White => 'W',
    }
}

// --- WebAssembly Wrapper (The only part JavaScript will see) ---

#[wasm_bindgen]
pub struct WasmGame {
    state: GameState,
    player_configs: Vec<PlayerType>,
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new(player_config_js: JsValue) -> WasmGame {
        let player_types_as_u8: Vec<u8> = serde_wasm_bindgen::from_value(player_config_js).unwrap();
        let num_players = player_types_as_u8.len();

        let player_configs = player_types_as_u8.into_iter().map(|n| {
            match n {
                0 => PlayerType::Human,
                1 => PlayerType::SimpleAI,
                2 => PlayerType::HeuristicAI, 
                3 => PlayerType::MctsAI,
                _ => panic!("Invalid player type"),
            }
        }).collect();

        WasmGame {
            state: GameState::new(num_players),
            player_configs,
        }
    }

    #[wasm_bindgen(js_name = getState)]
    pub fn get_state(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.state).unwrap()
    }

    #[wasm_bindgen(js_name = getLegalMoves)]
    pub fn get_legal_moves(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.state.get_legal_moves()).unwrap()
    }

    #[wasm_bindgen(js_name = applyMove)]
    pub fn apply_move(&mut self, move_js: JsValue) {
        let player_move: Move = serde_wasm_bindgen::from_value(move_js).unwrap();
        self.state.apply_move(&player_move);
    }

    #[wasm_bindgen(js_name = runFullTilingPhase)]
    pub fn run_full_tiling_phase(&mut self) {
        let next_starter_idx = self.state.players.iter().position(|p| p.has_first_player_marker)
            .unwrap_or(self.state.current_player_idx);

        let mut discard_pile_ref = std::mem::take(&mut self.state.discard_pile);
        for player in self.state.players.iter_mut() {
            player.run_tiling_phase(&mut discard_pile_ref);
        }
        self.state.discard_pile = discard_pile_ref;
        
        self.state.current_player_idx = next_starter_idx;

        if !self.state.end_game_triggered {
            self.state.refill_factories();
        }
    }

    #[wasm_bindgen(js_name = applyEndGameScoring)]
    pub fn apply_end_game_scoring(&mut self) {
        self.state.apply_end_game_scoring();
    }

    #[wasm_bindgen(js_name = isGameOver)]
    pub fn is_game_over(&self) -> bool {
        self.state.is_game_over()
    }

    #[wasm_bindgen(js_name = getWallLayout)]
    pub fn get_wall_layout(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&WALL_LAYOUT).unwrap()
    }

    #[wasm_bindgen(js_name = runAiTurn)]
    pub fn run_ai_turn(&mut self) {
        let current_player_type = self.player_configs[self.state.current_player_idx];
        
        // Update this match statement
        let agent: Box<dyn AIAgent> = match current_player_type {
            PlayerType::SimpleAI => Box::new(SimpleAI),
            PlayerType::HeuristicAI => Box::new(HeuristicAI), 
            PlayerType::MctsAI => Box::new(MctsAI::new(self.state.clone())),
            PlayerType::Human => return,
        };

        if let Some(ai_move) = agent.get_move(&self.state) {
            self.state.apply_move(&ai_move);
        }
    }
}
