import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

def display_board(game_state: GameState):
    """
    Prints the Sudoku board with distinctions:
    - X  : occupied by player 1
    - O  : occupied by player 2
    - .  : empty but not allowed
    - 1  : empty & allowed for player 1
    - 2  : empty & allowed for player 2
    - 3  : empty & allowed for both
    """
    N = game_state.board.N
    board = game_state.board
    for r in range(N):
        row_display = []
        for c in range(N):
            cell = board.get((r, c))
            if cell != board.empty:
                # Determine which player owns the cell
                if (r, c) in game_state.occupied_squares1:
                    row_display.append(f"X")  # Player 1
                elif (r, c) in game_state.occupied_squares2:
                    row_display.append(f"O")  # Player 2
                else:
                    row_display.append(f"{cell}")  # Already filled (fallback)
            else:
                # Empty cell
                a1 = (r, c) in game_state.allowed_squares1
                a2 = (r, c) in game_state.allowed_squares2
                if a1 and a2:
                    row_display.append("3")
                elif a1:
                    row_display.append("1")
                elif a2:
                    row_display.append("2")
                else:
                    row_display.append(".")
        print(" ".join(row_display))
    print("\n")

# --- Compute incremental points according to completed regions ---
def count_completed_regions(board: SudokuBoard, square: tuple, region_h: int, region_w: int) -> int:
    r, c = square
    completed = 0

    # Check row
    if all(board.get((r, j)) != board.empty for j in range(board.N)):
        completed += 1
    # Check column
    if all(board.get((i, c)) != board.empty for i in range(board.N)):
        completed += 1
    # Check block/region
    region_r = (r // region_h) * region_h
    region_c = (c // region_w) * region_w
    block_cells = [(region_r + i, region_c + j) for i in range(region_h) for j in range(region_w)]
    if all(board.get(cell) != board.empty for cell in block_cells):
        completed += 1

    return completed
