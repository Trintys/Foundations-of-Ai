import random, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

def display_board(game_state: GameState):
    """
    Prints the Sudoku board:
    - Values are shown in each cell
    - Player 1 moves have '-' suffix
    - Player 2 moves have '+' suffix
    - Empty cells are printed as '.'
    """
    N = game_state.board.N
    board = game_state.board
    for r in range(N):
        row_display = []
        for c in range(N):
            cell = board.get((r, c))
            if cell != board.empty:
                if (r, c) in game_state.occupied_squares1:
                    row_display.append(f"{cell}-")  # Player 1
                elif (r, c) in game_state.occupied_squares2:
                    row_display.append(f"{cell}+")  # Player 2
                else:
                    row_display.append(f"{cell}")   # Pre-filled or fallback
            else:
                row_display.append(".")  # Empty cell
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

    block_cells = []
    for i in range(region_h):
        for j in range(region_w):
            row_idx = region_r + i
            col_idx = region_c + j
            if row_idx < board.N and col_idx < board.N:  # <-- clip to board size
                block_cells.append((row_idx, col_idx))

    if all(board.get(cell) != board.empty for cell in block_cells):
        completed += 1

    return completed


def board_is_full(board):
    N = board.N
    for i in range(N):
        for j in range(N):
            if board.get((i, j)) == SudokuBoard.empty:
                return False
    return True

def is_really_legal(game_state: GameState, move: Move, player_num: int) -> bool:
    r, c = move.square

    # Already occupied?
    if game_state.board.get((r, c)) != 0:
        return False

    # Allowed square?
    allowed = (
        game_state.allowed_squares1 if player_num == 1
        else game_state.allowed_squares2
    )
    if move.square not in allowed:
        return False

    # Taboo?
    if TabooMove(move.square, move.value) in getattr(game_state, 'taboo_moves', []):
        return False

    return True

def seed_board_with_N_values(game_state: GameState, agent1, agent2, N: int):
    """
    Adds exactly N legal values to the board before the episode starts.
    Values are 1..N, each used once.
    """
    values = list(range(1, N + 1))
    random.shuffle(values)

    for v in values:
        player_num = game_state.current_player
        agent = agent1 if player_num == 1 else agent2

        # Generate legal moves for THIS value only
        raw_moves = agent.generate_legal_moves(game_state)
        legal_moves = [
            mv for mv in raw_moves
            if mv.value == v and is_really_legal(game_state, mv, player_num)
        ]

        if not legal_moves:
            # Skip this value if no legal placement exists
            continue

        move = random.choice(legal_moves)
        game_state = agent._apply_move(game_state, move)

        # Update occupied squares
        if player_num == 1:
            game_state.occupied_squares1.append(move.square)
        else:
            game_state.occupied_squares2.append(move.square)

        # Alternate player
        game_state.current_player = 3 - game_state.current_player

    return game_state
