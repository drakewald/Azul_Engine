use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use wasm_bindgen::prelude::*;
use std::fmt;

pub mod ai;
use ai::{
    human_agent::HumanAgent,
    heuristic_ai::HeuristicAI,
    mcts_heuristic_ai::MctsHeuristicAI,
    simple_ai::SimpleAI,
    AIAgent
};
// Conditionally import the MctsNnAI only when the "native" feature is enabled.
#[cfg(feature = "native")]
use ai::mcts_nn_ai::MctsNnAI;


// --- Structs for Game Logic ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tile {
    Blue,
    Yellow,
    Red,
    Black,
    White,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TileBagSummary {
    pub blue: usize,
    pub yellow: usize,
    pub red: usize,
    pub black: usize,
    pub white: usize,
}

impl TileBagSummary {
    pub fn from_vec(tiles: &[Tile]) -> Self {
        let mut summary = Self { blue: 0, yellow: 0, red: 0, black: 0, white: 0 };
        for &tile in tiles {
            match tile {
                Tile::Blue => summary.blue += 1,
                Tile::Yellow => summary.yellow += 1,
                Tile::Red => summary.red += 1,
                Tile::Black => summary.black += 1,
                Tile::White => summary.white += 1,
            }
        }
        summary
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Serialize)]
pub struct TurnState {
    pub players: Vec<PlayerBoard>,
    pub factories: Vec<Vec<Tile>>,
    pub center: Vec<Tile>,
    pub current_player_idx: usize,
    pub first_player_marker_in_center: bool,
    pub end_game_triggered: bool,
}

impl From<&GameState> for TurnState {
    fn from(game_state: &GameState) -> Self {
        Self {
            players: game_state.players.clone(),
            factories: game_state.factories.clone(),
            center: game_state.center.clone(),
            current_player_idx: game_state.current_player_idx,
            first_player_marker_in_center: game_state.first_player_marker_in_center,
            end_game_triggered: game_state.end_game_triggered,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum MoveSource {
    Factory(usize),
    Center,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum MoveDestination {
    PatternLine(usize),
    Floor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct Move {
    pub source: MoveSource,
    pub tile: Tile,
    pub destination: MoveDestination,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingData {
    pub state_input: Vec<f32>,
    pub mcts_policy: Vec<f32>,
    pub outcome: f32,
}

// --- Constants ---
const NUM_ROWS: usize = 5;
const NUM_COLS: usize = 5;
const TILES_PER_COLOR: usize = 20;
const FLOOR_PENALTY_VALUES: [u32; 7] = [1, 1, 2, 2, 2, 3, 3];
const WALL_LAYOUT: [[Tile; NUM_COLS]; NUM_ROWS] = [
    [Tile::Blue, Tile::Yellow, Tile::Red, Tile::Black, Tile::White],
    [Tile::White, Tile::Blue, Tile::Yellow, Tile::Red, Tile::Black],
    [Tile::Black, Tile::White, Tile::Blue, Tile::Yellow, Tile::Red],
    [Tile::Red, Tile::Black, Tile::White, Tile::Blue, Tile::Yellow],
    [Tile::Yellow, Tile::Red, Tile::Black, Tile::White, Tile::Blue],
];

// --- Game Logic Implementation ---

impl GameState {
    pub fn new(num_players: usize) -> Self {
        let players = (0..num_players).map(|_| PlayerBoard::new()).collect();
        let all_colors = [Tile::Blue, Tile::Yellow, Tile::Red, Tile::Black, Tile::White];
        let mut tile_bag: Vec<Tile> = all_colors
            .iter()
            .flat_map(|&tile| std::iter::repeat(tile).take(TILES_PER_COLOR))
            .collect();
        tile_bag.shuffle(&mut thread_rng());

        let num_factories = match num_players {
            2 => 5,
            3 => 7,
            4 => 9,
            _ => panic!("Invalid number of players."),
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
        let mut rng = thread_rng();
        for factory in self.factories.iter_mut() {
            factory.clear();
            for _ in 0..4 {
                if self.tile_bag.is_empty() {
                    if self.discard_pile.is_empty() { break; }
                    std::mem::swap(&mut self.tile_bag, &mut self.discard_pile);
                    self.tile_bag.shuffle(&mut rng);
                }
                if !self.tile_bag.is_empty() {
                    let random_index = rng.gen_range(0..self.tile_bag.len());
                    let tile = self.tile_bag.remove(random_index);
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
            for &tile in unique_tiles.iter() {
                for i in 0..NUM_ROWS {
                    if current_player_board.is_placement_valid(i, tile) {
                        legal_moves.push(Move {
                            source: source.clone(),
                            tile,
                            destination: MoveDestination::PatternLine(i),
                        });
                    }
                }
                legal_moves.push(Move {
                    source: source.clone(),
                    tile,
                    destination: MoveDestination::Floor,
                });
            }
        };

        for (i, factory) in self.factories.iter().enumerate() {
            if !factory.is_empty() {
                generate_moves_for_source(MoveSource::Factory(i), factory);
            }
        }
        if !self.center.is_empty() {
            generate_moves_for_source(MoveSource::Center, &self.center);
        }
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
        player.place_tiles(&mut taken_tiles, &player_move.destination);
        if let MoveDestination::PatternLine(idx) = player_move.destination {
            if !self.end_game_triggered && player.will_complete_horizontal_row(idx) {
                self.end_game_triggered = true;
            }
        }
        self.current_player_idx = (self.current_player_idx + 1) % self.players.len();
    }

    pub fn is_round_over(&self) -> bool {
        self.factories.iter().all(|f| f.is_empty()) && self.center.is_empty()
    }

    pub fn run_tiling_phase(&mut self) {
        let next_starter_idx = self.players.iter().position(|p| p.has_first_player_marker)
            .unwrap_or(self.current_player_idx);
        let mut discard_pile_ref = std::mem::take(&mut self.discard_pile);
        for player in self.players.iter_mut() {
            if player.run_tiling_phase(&mut discard_pile_ref) {
                self.end_game_triggered = true;
            }
        }
        self.discard_pile = discard_pile_ref;
        self.current_player_idx = next_starter_idx;
    }

    pub fn apply_end_game_scoring(&mut self) {
        for player in self.players.iter_mut() {
            player.score += player.calculate_end_game_bonuses();
        }
    }
}

impl PlayerBoard {
    pub fn new() -> Self {
        Self {
            score: 0,
            pattern_lines: vec![
                Vec::with_capacity(1), Vec::with_capacity(2), Vec::with_capacity(3),
                Vec::with_capacity(4), Vec::with_capacity(5),
            ],
            wall: vec![vec![None; NUM_COLS]; NUM_ROWS],
            floor_line: Vec::new(),
            has_first_player_marker: false,
        }
    }
    
    pub fn count_complete_rows(&self) -> usize {
        self.wall.iter().filter(|row| row.iter().all(|tile| tile.is_some())).count()
    }

    fn will_complete_horizontal_row(&self, pattern_line_idx: usize) -> bool {
        if self.pattern_lines[pattern_line_idx].len() != pattern_line_idx + 1 { return false; }
        self.wall[pattern_line_idx].iter().filter(|tile| tile.is_some()).count() == 4
    }

    pub fn place_tiles(&mut self, tiles_to_place: &mut Vec<Tile>, destination: &MoveDestination) {
        match destination {
            MoveDestination::Floor => self.floor_line.append(tiles_to_place),
            MoveDestination::PatternLine(idx) => {
                let pattern_line = &mut self.pattern_lines[*idx];
                let capacity = *idx + 1;
                while !tiles_to_place.is_empty() && pattern_line.len() < capacity {
                    pattern_line.push(tiles_to_place.pop().unwrap());
                }
                self.floor_line.append(tiles_to_place);
            }
        }
    }

    pub fn is_placement_valid(&self, pattern_line_idx: usize, tile_color: Tile) -> bool {
        let line = &self.pattern_lines[pattern_line_idx];
        if line.len() >= pattern_line_idx + 1 { return false; }
        if !line.is_empty() && line[0] != tile_color { return false; }
        if let Some(col_idx) = WALL_LAYOUT[pattern_line_idx].iter().position(|&t| t == tile_color) {
            if self.wall[pattern_line_idx][col_idx].is_some() { return false; }
        }
        true
    }

    pub fn run_tiling_phase(&mut self, discard_pile: &mut Vec<Tile>) -> bool {
        let mut completed_a_row = false;
        let mut new_score: u32 = 0;
        let mut tiles_to_discard: Vec<Vec<Tile>> = vec![vec![]; NUM_ROWS];

        for row_idx in 0..NUM_ROWS {
            if self.pattern_lines[row_idx].len() == row_idx + 1 {
                let tile_color = self.pattern_lines[row_idx][0];
                if let Some(col_idx) = WALL_LAYOUT[row_idx].iter().position(|&t| t == tile_color) {
                    if self.wall[row_idx][col_idx].is_none() {
                        new_score += self.calculate_placement_score(row_idx, col_idx);
                        self.wall[row_idx][col_idx] = Some(tile_color);
                        tiles_to_discard[row_idx] = std::mem::take(&mut self.pattern_lines[row_idx]);
                        if !completed_a_row && self.wall[row_idx].iter().all(Option::is_some) {
                            completed_a_row = true;
                        }
                    }
                }
            }
        }
        self.score += new_score;
        for mut line in tiles_to_discard { discard_pile.append(&mut line); }

        let mut floor_items_count = self.floor_line.len();
        if self.has_first_player_marker { floor_items_count += 1; }
        if floor_items_count > 0 {
            let penalty: u32 = FLOOR_PENALTY_VALUES[..floor_items_count.min(7)].iter().sum();
            self.score = self.score.saturating_sub(penalty);
        }
        discard_pile.append(&mut self.floor_line);
        self.has_first_player_marker = false;
        completed_a_row
    }

    fn calculate_placement_score(&self, row: usize, col: usize) -> u32 {
        let mut horizontal_score = 1;
        for i in (0..col).rev() { if self.wall[row][i].is_some() { horizontal_score += 1; } else { break; } }
        for i in (col + 1)..NUM_COLS { if self.wall[row][i].is_some() { horizontal_score += 1; } else { break; } }
        let mut vertical_score = 1;
        for i in (0..row).rev() { if self.wall[i][col].is_some() { vertical_score += 1; } else { break; } }
        for i in (row + 1)..NUM_ROWS { if self.wall[i][col].is_some() { vertical_score += 1; } else { break; } }
        if horizontal_score > 1 && vertical_score > 1 { horizontal_score + vertical_score } else { horizontal_score.max(vertical_score) }
    }

    pub fn calculate_end_game_bonuses(&self) -> u32 {
        let mut bonus_score = 0;
        for row in 0..NUM_ROWS { if self.wall[row].iter().all(Option::is_some) { bonus_score += 2; } }
        for col in 0..NUM_COLS { if (0..NUM_ROWS).all(|row| self.wall[row][col].is_some()) { bonus_score += 7; } }
        for color_to_check in [Tile::Blue, Tile::Yellow, Tile::Red, Tile::Black, Tile::White] {
            if self.wall.iter().flatten().filter(|&&tile| tile == Some(color_to_check)).count() == 5 {
                bonus_score += 10;
            }
        }
        bonus_score
    }
}

fn tile_to_char(tile: Tile) -> char {
    match tile {
        Tile::Blue => 'B',
        Tile::Yellow => 'Y',
        Tile::Red => 'R',
        Tile::Black => 'K',
        Tile::White => 'W',
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


// --- WebAssembly Wrapper ---

#[derive(Serialize, Deserialize)]
struct WasmGameConfig {
    player_types: Vec<u8>,
    model_bytes: Option<Vec<u8>>,
}

#[wasm_bindgen]
pub struct WasmGame {
    state: GameState,
    agents: Vec<Box<dyn AIAgent>>,
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new(config_js: JsValue) -> Result<WasmGame, JsValue> {
        let config: WasmGameConfig = serde_wasm_bindgen::from_value(config_js)
            .map_err(|e| JsValue::from_str(&format!("Config error: {}", e)))?;
        let num_players = config.player_types.len();
        if !(2..=4).contains(&num_players) { return Err(JsValue::from_str("Invalid player count.")); }

        let initial_state = GameState::new(num_players);
        
        let agents: Vec<Box<dyn AIAgent>> = config.player_types.into_iter().map(|n| -> Box<dyn AIAgent> {
            match n {
                0 => Box::new(HumanAgent),
                1 => Box::new(SimpleAI),
                2 => Box::new(HeuristicAI),
                3 => Box::new(MctsHeuristicAI::new(500)),
                4 => {
                    // This code will only be included when compiling for Wasm.
                    #[cfg(target_arch = "wasm32")]
                    {
                        web_sys::console::warn_1(&"MctsNnAI is not available in WebAssembly. Falling back to SimpleAI.".into());
                    }
                    Box::new(SimpleAI)
                },
                _ => Box::new(HumanAgent),
            }
        }).collect();

        Ok(WasmGame { state: initial_state, agents })
    }

    #[wasm_bindgen(js_name = getState)]
    pub fn get_state(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.state).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = getLegalMoves)]
    pub fn get_legal_moves(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.state.get_legal_moves()).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = applyMove)]
    pub fn apply_move(&mut self, move_js: JsValue) -> Result<(), JsValue> {
        let player_move: Move = serde_wasm_bindgen::from_value(move_js).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.state.apply_move(&player_move);
        Ok(())
    }

    #[wasm_bindgen(js_name = handleRoundEnd)]
    pub fn handle_round_end(&mut self) {
        if self.state.is_round_over() {
            self.state.run_tiling_phase();
            if !self.state.end_game_triggered {
                self.state.refill_factories();
            }
        }
    }

    #[wasm_bindgen(js_name = applyEndGameScoring)]
    pub fn apply_end_game_scoring(&mut self) {
        self.state.apply_end_game_scoring();
    }

    #[wasm_bindgen(js_name = isGameOver)]
    pub fn is_game_over(&self) -> bool {
        self.state.end_game_triggered && self.state.is_round_over()
    }

    #[wasm_bindgen(js_name = getWallLayout)]
    pub fn get_wall_layout(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&WALL_LAYOUT).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = runAiTurn)]
    pub fn run_ai_turn(&mut self) -> Result<(), JsValue> {
        let agent = &mut self.agents[self.state.current_player_idx];
        if let Some(ai_move) = agent.get_move(&self.state) {
            self.state.apply_move(&ai_move);
        }
        Ok(())
    }
}
