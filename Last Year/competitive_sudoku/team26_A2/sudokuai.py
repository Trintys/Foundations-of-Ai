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

            return   new_allowed * 0.5 + (new_score)*40 



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
            state.current_player = 3 - current_player
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

        def get_children(state,solve,heuristic):
            """
            Generate all possible child states from the current game state.

            @Param state: A GameState object representing the current game state.
            @Return: A list of tuples (GameState, Move) representing possible child states and their associated moves.
            """
            all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                         for value in range(1, N + 1) if possible(i, j, value)]
            all_moves2 = [move for move in all_moves if is_valid_move(state.board, move.square, move.value)]
            child_states = []
            unique_moves=[]
            all_moves_heuristic=[]
            square_counts = {}  # Dictionary to count occurrences of each square
            for move in all_moves2:
                square = move.square
                square_counts[square] = square_counts.get(square, 0) + 1
            for move in all_moves2:
                square_count = square_counts[move.square]
                if square_count==1:
                    unique_moves.append(move)
            if heuristic==True:
                all_moves_heuristic=naked_twins(state,all_moves2)
            if len(all_moves2)==0:
                return child_states
            all_moves_heuristic.extend(unique_moves)
            for move in all_moves_heuristic:
                
                new_state = copy.deepcopy(state)
                play_move(new_state, new_state.current_player, move)
              #  if (solve_sudoku(copy.deepcopy(new_state.board))==True):
             #       child_states.append((new_state, move,eval_score(new_state)))
                if solve==True:
                    t=solve_sudoku(copy.deepcopy(new_state.board))
                else:
                    t=True
                child_states.append((new_state,move,eval_score(new_state),t))
            if (state.current_player==2):
                reverse=True
            else:
                reverse=False
            child_states.sort(key=lambda x: x[2], reverse=reverse)
        
            return child_states
        def get_subgrid(board : SudokuBoard, row, col):

            region_height = board.region_height()
            region_width = board.region_width()
            region_start_row = (row // region_height) * region_height
            region_start_col = (col // region_width) * region_width

            return region_start_row, region_start_col

        def naked_twins(state : GameState, possible_moves : list) -> dict:
            '''
            Strategy I
            '''
            naked_twins = {}
            board = state.board
            possible_pairs = {} # dictionary for the hidden pairs
            naked_pairs = {} # list of unique pair of cells
        
            # populating the dictionary 
            for move in possible_moves:
                key = move.square
                val = move.value
                possible_pairs.setdefault(key, []).append(val)

            for key, values in possible_pairs.items():
                if len(values) == 2 : # size of values should be 2
                    naked_pairs[key] = values


            for key1 in naked_pairs:
                key1_row, key1_col = get_subgrid(state.board, key1[0], key1[1])
                for key2 in naked_pairs:
                    if key1 != key2:
                        key2_row, key2_col = get_subgrid(board, key2[0], key2[1])
                        
                        if (key1_row == key2_row and key1_col == key2_col):
                            naked_twins[key1,key2] = naked_pairs[key1]

            
            move_list = []
            for (square1, square2), values in naked_twins.items():
                for square in (square1, square2):  # Iterate over both squares
                    for value in values:  # Iterate over the possible values
                        move = Move(square, value)  # Create a Move object
                        move_list.append(move)

            return move_list
        def minimax(state,allowed,heuristic,depth, alpha, beta, isMaximisingPlayer):
            """
            Implementation of the minimax algorithm with alpha-beta pruning.

            @Param state: A GameState object representing the current board state.
            @Param allowed: Boolean value based on if the move is allowed or not ( is taboo or not).
            @Param heuristic: Boolean value that dictates if the minimax will usea heuristic approach or an extensive search.
            @Param depth: Integer indicating the current depth of the recursion.
            @Param alpha: The best value that the maximizing player can guarantee.
            @Param beta: The best value that the minimizing player can guarantee.
            @Param isMaximisingPlayer: Boolean indicating if the current player is the maximizer.
            @Return: A tuple (score, move), where score is the best score and move is the associated move.
            """
            children_and_move = get_children(state,True,heuristic)

            if depth == 0:
                return eval_score(state), None

            if len(children_and_move) == 0:
                return eval_score(state), None


            if isMaximisingPlayer:

                max_value, best_move = float('-inf'), None
                for child, move,score,allowed in children_and_move:

                    #If the move is taboo never branch on it or play it to maintain tempo
                    if allowed==False:
                        continue
                    child_value, _ = minimax(child,allowed,False,depth - 1, alpha, beta, False)  # Evaluate child
                    if child_value > max_value:  # Update best value and move
                        max_value, best_move = child_value, move
                    alpha = max(alpha, child_value)
                    if alpha >= beta:
                        break  # Beta cut-off
                return max_value, best_move
            else:
                min_value, best_move = float('inf'), None
                for child, move,score,allowed in children_and_move:
                    
                    if allowed==False:
                        continue
                    child_value, _ = minimax(child,allowed,False,depth - 1, alpha, beta, True)  # Evaluate child
                    if child_value < min_value:  # Update best value and move
                        min_value, best_move = child_value, move
                    beta = min(beta, child_value)
                    if alpha >= beta:
                        break  # Alpha cut-off
                return min_value, best_move

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
        all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                    for value in range(1, N + 1) if possible(i, j, value)]
        all_moves2 = [move for move in all_moves if is_valid_move(game_state.board, move.square, move.value)]
        move = random.choice(all_moves2)
        #First propose a random move in case the strategy and minimax fail
        self.propose_move(move)
        if game_state.current_player == 1:
            alg = True
        else:
            alg = False

        move=strategy_move(game_state,move)
        self.propose_move(move)
        #then propose a strategic move
        score, move = minimax(game_state,True,True, 4, float('-inf'), float('inf'), alg)
        print("minimax2")
        #Finally propose a minimax move, after checking if a move has returned from the alg.
        if  move is not None:
            self.propose_move(move)

        # Continuously propose the move
        while True:
            time.sleep(0.2)
            if  move is not None:
             self.propose_move(move)

