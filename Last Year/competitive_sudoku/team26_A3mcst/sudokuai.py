#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)
from math import sqrt, log
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
        def solve_sudoku(board):
            """
            Solves the given Sudoku puzzle using backtracking.

            @Param board: A SudokuBoard object representing the current game board.
            @Return: True if a solution is found, False otherwise.
            """
            # Find the first empty square
            for row in range(board.board_height()):  # Adjust to the board's height
                for col in range(board.board_width()):  # Adjust to the board's width
                    if board.get((row, col)) == 0:  # Empty cell found
                        for num in range(1, board.board_width() + 1):  # Possible numbers based on board size
                            if is_valid_move(board, (row, col), num):
                                board.put((row, col), num)  # Place number using put()

                                if solve_sudoku(board):  # Recursively attempt to solve
                                    return True

                                board.put((row, col), 0)  # Undo the move (backtrack)

                        return False  # No valid number found, backtrack

            # If no empty cells are left, the Sudoku is solved
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
        def eval_score_MC(state,player):
            """
            Evaluate the score of the current game state.

            @Param state: A GameState object representing the current game state.
            @Return: 1 if the current player wins, 0 otherwise.
            """
            score_first = state.scores[0]
            score_second = state.scores[1]
            
            if player == 1:
                return 1 if score_first > score_second else 0
            else:
                return 1 if score_second > score_first else 0
        def eval_score_heuristic(state,move):
            """
            Evaluate the score of the current game state.

            @Param state: A GameState object representing the current game state.
            @Return: A numeric value representing the evaluated score of the state.
            """
            score_first = state.scores[0]
            score_second = state.scores[1]
            if state.current_player == 1:
                expanding = len(state.allowed_squares1)
                old_score=score_first-score_second
            else:
                expanding = len(state.allowed_squares2)
                old_score=score_second-score_first

            new_state=play_move(copy.deepcopy(state),state.current_player,move)
            
            if state.current_player == 1:
                new_allowed=len(new_state.allowed_squares1)-expanding
                new_score= is_move_completing_row_column_table(new_state, move)+score_first -old_score
            
            else:
                new_allowed=len(new_state.allowed_squares2)-expanding
                new_score= is_move_completing_row_column_table(new_state, move)+score_second -old_score

            return   new_allowed * 0.5 +(new_score)*40



        def is_move_completing_row_column_table(state: GameState, move: Move) -> int:
            """
            Check if the move completes a row, column, or subgrid.

            @Param state: A GameState object representing the current game state.
            @Param move: A Move object representing the cell and value to place.
            @Return: An integer representing the points earned by completing rows, columns, or subgrids.
            """
            row, col = move.square
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
 #           state.current_player = 3 - current_player
            if(solve_sudoku(copy.deepcopy(state.board))==False):
                state.taboo_moves.append(move)
                return state

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



        class MCTSNode:
            def __init__(self, state, move=None, parent=None):
                self.state = state
                self.move = move
                self.parent = parent
                self.children = []
                self.visits = 0
                self.value = 0.0

            def is_fully_expanded(self):
                return len(self.children) > 0

            def best_child(self, exploration_weight=1.0):
                return max(
                    self.children,
                    key=lambda child: child.value / (child.visits + 1e-6) + exploration_weight * sqrt(log(self.visits + 1) / (child.visits + 1e-6))
                )

            def expand(self, possible_moves):
                for move in possible_moves:
                    new_state = copy.deepcopy(self.state)
                    play_move(new_state, new_state.current_player, move)
                    child_node = MCTSNode(new_state, move, self)
                    self.children.append(child_node)

            def update(self, result):
                self.visits += 1
                self.value += result


        def monte_carlo_tree_search(root_state, iterations):
            root = MCTSNode(root_state)

            for _ in range(iterations):
                # 1. Selection
                node = root
                while node.is_fully_expanded() and len(node.children) > 0:
                    node = node.best_child()
                # 2. Expansion
                possible_moves = [
                    Move((i, j), value) for i in range(root_state.board.N) for j in range(root_state.board.N)
                    for value in range(1, root_state.board.N + 1)
                    if possible(i, j, value) and is_valid_move(node.state.board, (i, j), value)
                ]
                if possible_moves:
                    node.expand(possible_moves)
                # 3. Simulation
                leaf_node = random.choice(node.children) if node.children else node
                result = simulate_game(leaf_node.state,root_state.current_player)
                # 4. Backpropagation
                current = leaf_node
                while current is not None:
                    current.update(result)
                    current = current.parent
                if (result==1):
                 #Propose for local moves
                 self.propose_move(root.best_child(exploration_weight=2).move)

            return root.best_child(exploration_weight=0).move


        def simulate_game(state,player):
            """
            Perform a random simulation from the given state until a terminal condition is reached.
            """

            current_state = copy.deepcopy(state)
            while not is_terminal_state(current_state):
                possible_moves = [
                    Move((i, j), value) for i in range(current_state.board.N) for j in range(current_state.board.N)
                    for value in range(1, current_state.board.N + 1)
                    if possible(i, j, value) and is_valid_move(current_state.board, (i, j), value)
                ]
                if not possible_moves:
                    break
                move = random.choice(possible_moves)

                current_state=play_move(current_state, current_state.current_player, move)

            return eval_score_MC(current_state,player)  # Return the score of the final state


        def is_terminal_state(state):
            """
            Determine if the state is terminal.
            """
            return all(state.board.get((i, j)) != SudokuBoard.empty for i in range(state.board.N) for j in range(state.board.N))


        def strategy_move(state,random_move):
            """
            Function that based on a given strategy returns the best move. This is the function dictating the moves in the beginning of the
            game when the tree is too large to be checked by minimax or there are no naked twins yet.If no move is caclutated it returns a random
            move.
            
            """
            all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
            for value in range(1, N + 1) if possible(i, j, value)]
            all_moves2 = [move for move in all_moves if is_valid_move(state.board, move.square, move.value)]
            child_states = []
            max=float('-inf')
            print("strategy")

            ret_move=random_move
            for move in all_moves2:
                value=eval_score_heuristic(state,move)
                if max <(value) :
                
                        max=value
                        ret_move=move


            return ret_move

        all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                    for value in range(1, N + 1) if possible(i, j, value)]
        all_moves2 = [move for move in all_moves if is_valid_move(game_state.board, move.square, move.value)]
        move = random.choice(all_moves2)  # Fallback random move

        self.propose_move(move)

        # Perform MCTS
        move=strategy_move(game_state,move)

        self.propose_move(move)
        best_move = monte_carlo_tree_search(game_state, iterations=100)
        print("best_move MCTS: ",best_move)
        if best_move is not None:
            self.propose_move(best_move)

        # Continuously propose the move
        while True:
            time.sleep(0.2)
            if best_move is not None:
                self.propose_move(best_move)
