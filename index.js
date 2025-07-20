import init, { WasmGame } from './pkg/azul_engine.js';

const gameContainer = document.getElementById('game-container');
const playersContainer = document.getElementById('players-container');
let game;
let WALL_LAYOUT;
let selectedTake = null;

function render() {
  if (!game) return;
  const state = game.getState();
  if (!state) return;

  // --- BANNER LOGIC ---
  const existingBanner = document.getElementById('end-game-banner');
  if (existingBanner) existingBanner.remove();

  // The banner now ONLY depends on the trigger flag.
  if (state.end_game_triggered) {
    const banner = document.createElement('div');
    banner.id = 'end-game-banner';
    banner.innerText = 'FINAL ROUND';
    document.body.prepend(banner);
  }

  gameContainer.innerHTML = '';
  playersContainer.innerHTML = '';
  const currentPlayerIdx = state.current_player_idx;

  // ... (Rest of the rendering logic is the same)
  const factoriesDiv = document.createElement('div');
  factoriesDiv.innerHTML = '<h2>Factories</h2>';
  state.factories.forEach((factory, factoryIndex) => {
    const factoryDiv = document.createElement('div');
    factoryDiv.className = 'factory';
    factoryDiv.innerHTML = `<strong>F${factoryIndex + 1}:</strong> `;
    factory.forEach(tile => {
      const tileDiv = document.createElement('span');
      tileDiv.className = `tile ${tile}`;
      tileDiv.innerText = tile.charAt(0);
      tileDiv.addEventListener('click', () => tileClicked(tile, { Factory: factoryIndex }));
      factoryDiv.appendChild(tileDiv);
    });
    factoriesDiv.appendChild(factoryDiv);
  });
  gameContainer.appendChild(factoriesDiv);
  const centerDiv = document.createElement('div');
  centerDiv.className = 'center';
  centerDiv.innerHTML = `<strong>Center:</strong> `;
  state.center.forEach(tile => {
    const tileDiv = document.createElement('span');
    tileDiv.className = `tile ${tile}`;
    tileDiv.innerText = tile.charAt(0);
    tileDiv.addEventListener('click', () => tileClicked(tile, 'Center'));
    centerDiv.appendChild(tileDiv);
  });
  gameContainer.appendChild(centerDiv);
  state.players.forEach((player, playerIndex) => {
    const playerDiv = document.createElement('div');
    playerDiv.className = 'player-board';
    playerDiv.innerHTML = `<h2>Player ${playerIndex + 1} (Score: ${player.score})</h2>`;
    if (playerIndex === currentPlayerIdx && !isGameOver()) {
      playerDiv.style.borderColor = 'gold';
    }
    const boardGrid = document.createElement('div');
    boardGrid.className = 'board-grid';
    const patternLinesDiv = document.createElement('div');
    patternLinesDiv.className = 'pattern-lines';
    player.pattern_lines.forEach((line, i) => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'pattern-line';
        lineDiv.id = `p${playerIndex}-row${i}`;
        const capacity = i + 1;
        const numPlaceholders = capacity - line.length;
        for (let j = 0; j < numPlaceholders; j++) {
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
    boardGrid.appendChild(patternLinesDiv);
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
          tileSpan.innerText = ghostColor.charAt(0);
        }
        rowDiv.appendChild(tileSpan);
      });
      wallDiv.appendChild(rowDiv);
    });
    boardGrid.appendChild(wallDiv);
    playerDiv.appendChild(boardGrid);
    const floorDiv = document.createElement('div');
    floorDiv.className = 'floor-line';
    floorDiv.innerHTML = '<strong>Floor:</strong> ';
    if (player.has_first_player_marker) {
      floorDiv.innerHTML += '<span class="tile placeholder">1</span> ';
    }
    player.floor_line.forEach(tile => {
      floorDiv.innerHTML += `<span class="tile ${tile}">${tile.charAt(0)}</span>`;
    });
    floorDiv.id = `p${playerIndex}-floor`;
    playerDiv.appendChild(floorDiv);
    playersContainer.appendChild(playerDiv);
  });
  if (selectedTake) {
    const allMoves = game.getLegalMoves();
    const validPlacements = allMoves.filter(move =>
      JSON.stringify(move.source) === JSON.stringify(selectedTake.source) &&
      move.tile === selectedTake.tile
    );
    const validRows = validPlacements.map(move => move.pattern_line_idx);
    validRows.forEach(rowIndex => {
      const rowId = rowIndex < 5 ? `p${currentPlayerIdx}-row${rowIndex}` : `p${currentPlayerIdx}-floor`;
      const element = document.getElementById(rowId);
      if (element) {
        element.classList.add('highlight');
        element.addEventListener('click', () => placementClicked(rowIndex));
      }
    });
  }
}

function tileClicked(tileColor, source) {
  if (isGameOver()) return; // Prevent moves after the game is truly over.

  if (selectedTake && JSON.stringify(selectedTake.source) === JSON.stringify(source) && selectedTake.tile === tileColor) {
    selectedTake = null;
  } else {
    selectedTake = { tile: tileColor, source: source };
  }
  render();
}

function placementClicked(pattern_line_idx) {
  const move = { ...selectedTake, pattern_line_idx };
  game.applyMove(move);
  selectedTake = null;

  const state = game.getState();
  const isDraftingOver = state.factories.every(f => f.length === 0) && state.center.length === 0;

  if (!isDraftingOver) {
    render();
    return;
  }

  // --- End of Round Logic ---
  game.runFullTilingPhase();
  
  // The game ends ONLY after the tiling phase is complete and a row is filled.
  if (isGameOver()) {
    console.log("--- Game Over! Calculating Final Scores ---");
    game.applyEndGameScoring();
    render(); // Render the final board state
    
    const finalState = game.getState();
    const winner = findWinner(finalState);
    alert(`Game Over! The winner is Player ${winner.index + 1} with ${winner.score} points!`);
    return;
  }
  
  render(); // Render for the next round
}

// This helper function now calls our Wasm function.
function isGameOver() {
    if (!game) return false;
    return game.isGameOver();
}

// This can remain a JS helper, as it's pure presentation logic.
function findWinner(state) {
    let bestPlayer = { index: 0, score: -1, rows: 0 }; // Start score at -1
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

async function main() {
  await init();
  game = new WasmGame(2);
  WALL_LAYOUT = game.getWallLayout();
  render();
}

main();