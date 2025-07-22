import init, { WasmGame } from './pkg/azul_engine.js';

// --- DOM Elements ---
const setupScreen = document.getElementById('setup-screen');
const gameScreen = document.getElementById('game-screen');
const gameContainer = document.getElementById('game-container');
const playersContainer = document.getElementById('players-container');
const numPlayersSelect = document.getElementById('num-players');
const playerOptionsContainer = document.getElementById('player-options-container');
const startGameBtn = document.getElementById('start-game-btn');

// --- Game State Variables ---
let game;
let WALL_LAYOUT;
let selectedTake = null;
let playerConfigs = [];

// --- Render Function ---
async function render() {
  if (!game) return;

  try {
    const state = await game.getState();
    if (!state) return;

    // --- Banner for Final Round ---
    const existingBanner = document.getElementById('end-game-banner');
    if (existingBanner) existingBanner.remove();
    if (state.end_game_triggered && !game.isGameOver()) {
        const banner = document.createElement('div');
        banner.id = 'end-game-banner';
        banner.innerText = 'FINAL ROUND';
        document.body.prepend(banner);
    }
    
    // --- Clear and Render Game Board ---
    gameContainer.innerHTML = '<h2>Factories</h2>';
    playersContainer.innerHTML = '';
    
    // --- Factories ---
    state.factories.forEach((factory, factoryIndex) => {
      const factoryDiv = document.createElement('div');
      factoryDiv.className = 'factory';
      if (factory.length === 0) {
        factoryDiv.classList.add('empty');
      }
      factoryDiv.innerHTML = `<strong>F${factoryIndex + 1}:</strong> `;
      factory.forEach(tile => {
        const tileDiv = document.createElement('span');
        tileDiv.className = `tile ${tile}`;
        tileDiv.innerText = tile.charAt(0);
        tileDiv.addEventListener('click', () => tileClicked(tile, { Factory: factoryIndex }));
        factoryDiv.appendChild(tileDiv);
      });
      gameContainer.appendChild(factoryDiv);
    });

    // --- Center ---
    const centerDiv = document.createElement('div');
    centerDiv.className = 'center';
    centerDiv.innerHTML = '<h2>Center</h2>';
    const centerTileArea = document.createElement('div');
    centerTileArea.className = 'tile-area';

    // --- ADDED: Logic to show the first player marker in the center ---
    if (state.first_player_marker_in_center) {
        const firstPlayerMarker = document.createElement('span');
        firstPlayerMarker.className = 'tile placeholder';
        firstPlayerMarker.innerText = '1';
        firstPlayerMarker.style.cursor = 'default'; // Make sure it's not clickable
        centerTileArea.appendChild(firstPlayerMarker);
    }
    // --- END OF ADDED LOGIC ---

    state.center.forEach(tile => {
        const tileDiv = document.createElement('span');
        tileDiv.className = `tile ${tile}`;
        tileDiv.innerText = tile.charAt(0);
        tileDiv.addEventListener('click', () => tileClicked(tile, 'Center'));
        centerTileArea.appendChild(tileDiv);
    });
    centerDiv.appendChild(centerTileArea);
    gameContainer.appendChild(centerDiv);
    
    // --- Player Boards ---
    state.players.forEach((player, playerIndex) => {
        const playerDiv = document.createElement('div');
        playerDiv.className = 'player-board';
        if (playerIndex === state.current_player_idx && !game.isGameOver()) {
            playerDiv.style.borderColor = 'gold';
            playerDiv.style.borderWidth = '2px';
        }
        playerDiv.innerHTML = `<h3>Player ${playerIndex + 1} (Score: ${player.score})</h3>`;

        const boardGrid = document.createElement('div');
        boardGrid.className = 'board-grid';

        const patternLinesDiv = document.createElement('div');
        patternLinesDiv.className = 'pattern-lines';
        player.pattern_lines.forEach((line, i) => {
            const lineDiv = document.createElement('div');
            lineDiv.className = 'pattern-line';
            lineDiv.id = `p${playerIndex}-row${i}`;
            const capacity = i + 1;
            for (let j = 0; j < capacity - line.length; j++) {
                const placeholderSpan = document.createElement('span');
                placeholderSpan.className = 'tile placeholder';
                lineDiv.appendChild(placeholderSpan);
            }
            line.forEach(tile => {
                const tileSpan = document.createElement('span');
                tileSpan.className = `tile ${tile}`;
                tileSpan.innerText = tile.charAt(0);
                lineDiv.appendChild(tileSpan);
            });
            patternLinesDiv.appendChild(lineDiv);
        });
        
        const wallDiv = document.createElement('div');
        wallDiv.className = 'wall-grid';
        player.wall.forEach((row, rowIndex) => {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'wall-row';
            row.forEach((tile, colIndex) => {
                const tileSpan = document.createElement('span');
                tileSpan.className = 'tile';
                if (tile) {
                    tileSpan.classList.add(tile);
                    tileSpan.innerText = tile.charAt(0);
                } else {
                    const ghostColor = WALL_LAYOUT[rowIndex][colIndex];
                    tileSpan.classList.add(ghostColor, 'ghost');
                }
                rowDiv.appendChild(tileSpan);
            });
            wallDiv.appendChild(rowDiv);
        });

        const floorDiv = document.createElement('div');
        floorDiv.className = 'floor-line';
        floorDiv.innerHTML = '<strong>Floor:</strong> ';
        if (player.has_first_player_marker) {
            const firstPlayerMarker = document.createElement('span');
            firstPlayerMarker.className = 'tile placeholder';
            firstPlayerMarker.innerText = '1';
            floorDiv.appendChild(firstPlayerMarker);
        }
        player.floor_line.forEach(tile => {
            const tileSpan = document.createElement('span');
            tileSpan.className = `tile ${tile}`;
            tileSpan.innerText = tile.charAt(0);
            floorDiv.appendChild(tileSpan);
        });
        floorDiv.id = `p${playerIndex}-floor`;

        boardGrid.appendChild(patternLinesDiv);
        boardGrid.appendChild(wallDiv);
        playerDiv.appendChild(boardGrid);
        playerDiv.appendChild(floorDiv);
        
        playersContainer.appendChild(playerDiv);
    });
    
    if (selectedTake) {
        highlightLegalPlacements();
    }

  } catch (error) {
      console.error("Failed to render game state:", error);
      alert(`Error rendering game: ${error}`);
  }
}

// --- Game Logic Functions ---
function tileClicked(tileColor, source) {
    if (game.isGameOver() || playerConfigs[game.getState().current_player_idx] !== 'Human') return;

    if (selectedTake && JSON.stringify(selectedTake.source) === JSON.stringify(source) && selectedTake.tile === tileColor) {
        selectedTake = null;
    } else {
        selectedTake = { tile: tileColor, source: source };
    }
    render();
}

async function placementClicked(rowIndex) {
    if (!selectedTake) return;

    const destination = (rowIndex < 5) ? { PatternLine: rowIndex } : { Floor: null };
    const move = { ...selectedTake, destination };

    try {
        await game.applyMove(move);
        selectedTake = null;
        await handleEndOfTurn();
    } catch (error) {
        console.error("Error applying move:", error);
        alert(`An error occurred: ${error}`);
        selectedTake = null;
        render();
    }
}

async function handleEndOfTurn() {
    try {
        const state = await game.getState();
        const isDraftingOver = state.factories.every(f => f.length === 0) && state.center.length === 0;

        if (isDraftingOver) {
            await game.handleRoundEnd();
        }

        if (game.isGameOver()) {
            await game.applyEndGameScoring();
            render();
            const finalState = await game.getState();
            const winner = findWinner(finalState);
            setTimeout(() => alert(`Game Over! Player ${winner.index + 1} wins with ${winner.score} points!`), 100);
            return;
        }

        render();
        checkForAIMove();
    } catch (error) {
        console.error("Error at end of turn:", error);
    }
}

function checkForAIMove() {
    try {
        const state = game.getState();
        const currentPlayerType = playerConfigs[state.current_player_idx];

        if (currentPlayerType !== 'Human' && !game.isGameOver()) {
            document.body.style.pointerEvents = 'none';
            setTimeout(async () => {
                try {
                    console.log(`--- Running ${currentPlayerType} for Player ${state.current_player_idx + 1} ---`);
                    await game.runAiTurn();
                    await handleEndOfTurn();
                } catch(aiError) {
                    console.error("AI Error:", aiError);
                    alert(`AI failed to make a move: ${aiError}`);
                } finally {
                    document.body.style.pointerEvents = 'auto';
                }
            }, 500);
        }
    } catch (error) {
        console.error("Could not check for AI move:", error);
    }
}

// --- Setup Functions ---
function updatePlayerOptions(numPlayers) {
  playerOptionsContainer.innerHTML = '';
  for (let i = 0; i < numPlayers; i++) {
    const div = document.createElement('div');
    div.className = 'player-option';
    div.innerHTML = `
      <label for="player-type-${i}">Player ${i + 1}:</label>
      <select id="player-type-${i}">
        <option value="Human" ${i === 0 ? 'selected' : ''}>Human</option>
        <option value="SimpleAI">Simple AI</option>
        <option value="HeuristicAI">Heuristic AI</option>
        <option value="MctsAI" ${i !== 0 ? 'selected' : ''}>MCTS AI</option>
      </select>
    `;
    playerOptionsContainer.appendChild(div);
  }
}

async function startGame() {
  const numPlayers = parseInt(numPlayersSelect.value, 10);
  
  playerConfigs = [];
  const playerTypesForWasm = [];

  for (let i = 0; i < numPlayers; i++) {
    const selectElement = document.getElementById(`player-type-${i}`);
    
    if (!selectElement) {
        console.error(`Could not find player type selector for player ${i + 1}`);
        alert("A UI error occurred. Please refresh the page.");
        return;
    }
    
    const playerType = selectElement.value;
    playerConfigs.push(playerType);
    
    if (playerType === 'Human') playerTypesForWasm.push(0);
    if (playerType === 'SimpleAI') playerTypesForWasm.push(1);
    if (playerType === 'HeuristicAI') playerTypesForWasm.push(2);
    if (playerType === 'MctsAI') playerTypesForWasm.push(3);
  }

  try {
    game = new WasmGame(playerTypesForWasm);
    WALL_LAYOUT = await game.getWallLayout();

    setupScreen.style.display = 'none';
    gameScreen.style.display = 'flex';

    render();
    checkForAIMove();
  } catch (error) {
    console.error("Failed to start game:", error);
    alert(`Could not start the game: ${error}`);
  }
}

// --- Helper Functions ---
function findWinner(state) {
  let bestPlayer = { index: 0, score: -1, rows: 0 };
  state.players.forEach((player, index) => {
    const completedRows = player.wall.filter(row => row.every(t => t !== null)).length;
    if (player.score > bestPlayer.score) {
      bestPlayer = { index, score: player.score, rows: completedRows };
    } else if (player.score === bestPlayer.score) {
      if (completedRows > bestPlayer.rows) {
        bestPlayer = { index, score: player.score, rows: completedRows };
      }
    }
  });
  return bestPlayer;
}

async function highlightLegalPlacements() {
    try {
        const allMoves = await game.getLegalMoves();
        const state = await game.getState();
        const validPlacements = allMoves.filter(move =>
            JSON.stringify(move.source) === JSON.stringify(selectedTake.source) &&
            move.tile === selectedTake.tile
        );

        validPlacements.forEach(move => {
            let element;
            if (move.destination.PatternLine !== undefined) {
                element = document.getElementById(`p${state.current_player_idx}-row${move.destination.PatternLine}`);
            } else {
                element = document.getElementById(`p${state.current_player_idx}-floor`);
            }
            
            if (element) {
                element.classList.add('highlight');
                element.addEventListener('click', () => {
                    if (move.destination.PatternLine !== undefined) {
                        placementClicked(move.destination.PatternLine);
                    } else {
                        placementClicked(5);
                    }
                }, { once: true });
            }
        });
    } catch (error) {
        console.error("Error highlighting moves:", error);
    }
}


// --- Main Execution ---
async function main() {
    await init();
    numPlayersSelect.addEventListener('change', (e) => updatePlayerOptions(e.target.value));
    startGameBtn.addEventListener('click', startGame);
    updatePlayerOptions(numPlayersSelect.value);
}

main();