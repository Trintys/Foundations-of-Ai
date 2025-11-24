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
        self.legal_moves_cache = None   # optional, but useful later

    # Valid moves
    def _is_valid_move(self, game_state: GameState, square: tuple, num: int) -> bool:
        """
        Check whether (square, num) is a legal move for the current player.
        """
        board = game_state.board
        row, col = square
        N = board.N

        # Check if allowed and not taboo
        if board.get((row, col)) != SudokuBoard.empty:
            return False
        if TabooMove((row, col), num) in game_state.taboo_moves:
            return False
        if (row, col) not in game_state.player_squares():
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
        subgrid_h = board.region_height()
        subgrid_w = board.region_width()
        start_row = (row // subgrid_h) * subgrid_h
        start_col = (col // subgrid_w) * subgrid_w

        for i in range(start_row, start_row + subgrid_h):
            for j in range(start_col, start_col + subgrid_w):
                if board.get((i, j)) == num:
                    return False

        return True


    def generate_legal_moves(self, game_state: GameState):
        """
        Returns all legal moves for the *current player* in the given game_state.
        """
        N = game_state.board.N

        playable_squares = game_state.player_squares()

        # If no allowed squares are registered, use whole board
        if not playable_squares:
            playable_squares = [(i, j) for i in range(N) for j in range(N)]

        moves = []
        for square in playable_squares:
            for value in range(1, N + 1):
                if self._is_valid_move(game_state, square, value):
                    moves.append(Move(square, value))

        return moves

    # Evaluation function
    def evaluate_state(self, state: GameState, my_moves, opp_moves):
        """
        Evaluate how good the given GameState is for OUR AI.
        Higher = better.
        """

        # Score difference
        try:
            my_score = state.my_score
            opp_score = state.opponent_score
        except:
            my_score = state.scores[0]
            opp_score = state.scores[1]

        score_diff = my_score - opp_score

        # Mobility
        mobility = len(my_moves) - len(opp_moves)

        # Board control (optional)
        my_allowed = len(state.player_squares())
        try:
            opp_allowed = len(state.opponent_squares())
        except:
            opp_allowed = 0

        board_control = my_allowed - opp_allowed

        # Weighted sum of heuristics
        return (
            10 * score_diff +
            2 * mobility +
            0.5 * board_control
        )


    def compute_best_move(self, game_state: GameState) -> None:
        """
        For now: 
        - generate legal moves
        - pick a random valid one
        - keep proposing it (anytime design)
        """

        legal_moves = self.generate_legal_moves(game_state)

        if not legal_moves:
            # If no legal moves, propose nothing (game framework will handle skip)
            return

        # Pick a random safe move as fallback
        move = random.choice(legal_moves)
        self.propose_move(move)

        # Anytime requirement: keep proposing the same move
        while True:
            time.sleep(0.2)
            self.propose_move(move)