#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import logging
logging.basicConfig(level=logging.INFO)

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
        self.legal_moves = set()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def possible(i, j, value):
            return game_state.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in game_state.taboo_moves \
                       and (i, j) in game_state.player_squares()
    
        def generate_legal_moves():
            allowed_moves = []

            playable_squares = game_state.player_squares()

            # if empty all squares are allowed so we should consider all squares
            if(playable_squares is None or len(playable_squares) == 0):
                playable_squares = [(i, j) for i in range(N) for j in range(N)]

            allowed_moves = [Move(square, value) for square in playable_squares
                            for value in range(1, N + 1) if self._is_valid_move(game_state, square, value)]
            return allowed_moves
        
        def _is_valid_move(square: tuple, num: int) -> bool:
            '''
                check whether the combaintion of square and number(value we want to input)
                is valid according to sudoku rules
            '''
            board = game_state.board
            subgrid_size = int(N ** 0.5)
            row, col = square

            # chech first if possible
            if(possible(row, col, num) == False):
                return False

            # Check row
            for j in range(N):
                if board.get((row, j)) == num:
                    return False

            # Check column
            for i in range(N):
                if board.get((i, col)) == num:
                    return False

            # Check subgrid
            start_row = (row // subgrid_size) * subgrid_size
            start_col = (col // subgrid_size) * subgrid_size
            for i in range(start_row, start_row + subgrid_size):
                for j in range(start_col, start_col + subgrid_size):
                    if board.get((i, j)) == num:
                        return False

            return True
        
        def generate_legal_moves():
            allowed_moves = []
            N = game_state.board.N

            playable_squares = game_state.player_squares()

            # if empty all squares are allowed so we should consider all squares
            if(playable_squares is None or len(playable_squares) == 0):
                playable_squares = [(i, j) for i in range(N) for j in range(N)]

            allowed_moves = [Move(square, value) for square in playable_squares
                            for value in range(1, N + 1) if _is_valid_move(square, value)]
            return allowed_moves

        all_moves = generate_legal_moves()

        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))




    