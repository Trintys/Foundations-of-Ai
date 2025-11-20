#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import copy

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        """
        Initialize the Sudoku AI instance.
        """
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        """
        Compute the best move for the current game state.

        @Param game_state: A GameState object representing the current board state.
        @Return: None
        """

        N = game_state.board.N

        def is_valid_move(board: SudokuBoard, square: tuple, num: int) -> bool:
            """
            Check if placing 'num' in 'square' is a valid move.

            @Param board: A SudokuBoard object representing the current game board.
            @Param square: A tuple (row, col) indicating the cell position.
            @Param num: An integer value to be placed in the square.
            @Return: True if the move is valid, False otherwise.
            """
            N = board.board_width()  # The total size of the board (N x N)
            row, col = square

            # Check if num is already in the same row
            for c in range(N):
                if board.get((row, c)) == num:
                    return False

            # Check if num is already in the same column
            for r in range(N):
                if board.get((r, col)) == num:
                    return False

            # Check if num is in the same region
            region_start_row = (row // board.region_height()) * board.region_height()
            region_start_col = (col // board.region_width()) * board.region_width()
            for r in range(region_start_row, region_start_row + board.region_height()):
                for c in range(region_start_col, region_start_col + board.region_width()):
                    if board.get((r, c)) == num:
                        return False

            return True

        def get_neighbors(state: GameState, move: Move):
            """
            Generate all valid neighbors of a square.

            @Param state: A GameState object representing the current board state.
            @Param move: A Move object representing the cell and value to place.
            @Return: A list of neighboring empty squares.
            """
            row, col = move.square
            N = state.board.N

            def neighbors(square):
                """
                Generate all neighboring cells of a square.

                @Param square: A tuple (row, col) representing the current cell position.
                @Return: A generator yielding neighboring squares.
                """
                r, c = square
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < N and 0 <= nc < N:
                            yield (nr, nc)

            move_neighbors = list(neighbors((row, col)))
            return [square for square in move_neighbors if state.board.get(square) == SudokuBoard.empty]

        def eval_score(state):
            """
            Evaluate the score of the current game state.

            @Param state: A GameState object representing the current game state.
            @Return: A numeric value representing the evaluated score of the state.
            """
            score_first = state.scores[0]
            score_second = state.scores[1]
            if state.current_player == 1:
                captured_spaces = len(state.allowed_squares1) - len(state.allowed_squares2)
                expanding = len(state.allowed_squares1)
            else:
                captured_spaces = len(state.allowed_squares1) - len(state.allowed_squares2)
                expanding = -len(state.allowed_squares2)
            return (score_first - score_second) * 4 + captured_spaces * 0.5 + expanding * 0.5

        def is_move_completing_row_column_table(state: GameState, move: Move) -> int:
            """
            Check if the move completes a row, column, or subgrid.

            @Param state: A GameState object representing the current game state.
            @Param move: A Move object representing the cell and value to place.
            @Return: An integer representing the points earned by completing rows, columns, or subgrids.
            """
            row, col = move.square
            value = move.value
            N = state.board.N

            def is_row_complete():
                return all(state.board.get((row, c)) != SudokuBoard.empty for c in range(N))

            def is_column_complete():
                return all(state.board.get((r, col)) != SudokuBoard.empty for r in range(N))

            def is_subgrid_complete():
                region_height = state.board.region_height()
                region_width = state.board.region_width()
                region_start_row = (row // region_height) * region_height
                region_start_col = (col // region_width) * region_width

                for r in range(region_start_row, region_start_row + region_height):
                    for c in range(region_start_col, region_start_col + region_width):
                        if state.board.get((r, c)) == SudokuBoard.empty:
                            return False
                return True

            state.board.put(move.square, move.value)
            row_completed = is_row_complete()
            column_completed = is_column_complete()
            subgrid_completed = is_subgrid_complete()
            completed_count = sum([row_completed, column_completed, subgrid_completed])

            if completed_count == 0:
                return 0
            elif completed_count == 1:
                return 1
            elif completed_count == 2:
                return 3
            elif completed_count == 3:
                return 7

        def play_move(state: GameState, current_player, move: Move):
            """
            Apply a move to the game state.

            @Param state: A GameState object representing the current game state.
            @Param current_player: An integer (1 or 2) indicating the current player.
            @Param move: A Move object representing the cell and value to place.
            @Return: A GameState object with the move applied.
            """
            state.scores[current_player - 1] += is_move_completing_row_column_table(state, move)
            state.board.put(move.square, move.value)
            state.moves.append(move)
            state.current_player = 3 - current_player

            if current_player == 1:
                state.occupied_squares1.append(move.square)
            else:
                state.occupied_squares2.append(move.square)

            neighbors = get_neighbors(state, move)
            if current_player == 1:
                if state.allowed_squares1 is None:
                    state.allowed_squares1 = []
                state.allowed_squares1.extend(neighbors)
                state.allowed_squares1 = list(set(state.allowed_squares1))  # Remove duplicates
            else:
                if state.allowed_squares2 is None:
                    state.allowed_squares2 = []
                state.allowed_squares2.extend(neighbors)
                state.allowed_squares2 = list(set(state.allowed_squares2))  # Remove duplicates

            return state

        def possible(i, j, value):
            """
            Check if placing 'value' at (i, j) is possible.

            @Param i: Row index of the cell.
            @Param j: Column index of the cell.
            @Param value: Value to place in the cell.
            @Return: True if the move is possible, False otherwise.
            """
            return game_state.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in game_state.taboo_moves \
                   and (i, j) in game_state.player_squares()

        def get_children(state):
            """
            Generate all possible child states from the current game state.

            @Param state: A GameState object representing the current game state.
            @Return: A list of tuples (GameState, Move) representing possible child states and their associated moves.
            """
            all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                         for value in range(1, N + 1) if possible(i, j, value)]
            all_moves2 = [move for move in all_moves if is_valid_move(state.board, move.square, move.value)]
            child_states = []
            for move in all_moves2:
                new_state = copy.deepcopy(state)
                play_move(new_state, new_state.current_player, move)
                child_states.append((new_state, move))
            return child_states

        def minimax(state, depth, alpha, beta, isMaximisingPlayer):
            """
            Implementation of the minimax algorithm with alpha-beta pruning.

            @Param state: A GameState object representing the current board state.
            @Param depth: Integer indicating the current depth of the recursion.
            @Param alpha: The best value that the maximizing player can guarantee.
            @Param beta: The best value that the minimizing player can guarantee.
            @Param isMaximisingPlayer: Boolean indicating if the current player is the maximizer.
            @Return: A tuple (score, move), where score is the best score and move is the associated move.
            """
            if depth == 0:
                return eval_score(state), None

            children_and_move = get_children(state)
            if len(children_and_move) == 0:
                return eval_score(state), None

            if isMaximisingPlayer:
                max_value, best_move = float('-inf'), None
                for child, move in children_and_move:
                    child_value, _ = minimax(child, depth - 1, alpha, beta, False)  # Evaluate child
                    if child_value > max_value:  # Update best value and move
                        max_value, best_move = child_value, move
                    alpha = max(alpha, child_value)
                    if alpha >= beta:
                        break  # Beta cut-off
                return max_value, best_move
            else:
                min_value, best_move = float('inf'), None
                for child, move in children_and_move:
                    child_value, _ = minimax(child, depth - 1, alpha, beta, True)  # Evaluate child
                    if child_value < min_value:  # Update best value and move
                        min_value, best_move = child_value, move
                    beta = min(beta, child_value)
                    if alpha >= beta:
                        break  # Alpha cut-off
                return min_value, best_move


        all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                    for value in range(1, N + 1) if possible(i, j, value)]
        all_moves2 = [move for move in all_moves if is_valid_move(game_state.board, move.square, move.value)]
        move = random.choice(all_moves2)
        self.propose_move(move)  # Propose a move in case minimax is too slow

        if game_state.current_player == 1:
            alg = True
        else:
            alg = False

        # Run minimax to find the best move
        score, move = minimax(game_state, 2, float('-inf'), float('inf'), alg)
        print("Minimax", score)
        print(move)
        self.propose_move(move)

        # Continuously propose the move
        while True:
            time.sleep(0.2)
            self.propose_move(move)

