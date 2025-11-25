#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy
import logging

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

logging.basicConfig(level=logging.INFO)


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Competitive Sudoku AI using minimax + alpha-beta pruning.
    Evaluation is from Player 1's point of view:
      eval(state) = f(score_1 - score_2, territory, mobility, ...)
    When this AI is player 1 it tries to MAXIMIZE eval;
    when it is player 2 it tries to MINIMIZE eval.
    """

    def __init__(self):
        super().__init__()

    def _valid_move(self, game_state: GameState, square: tuple, num: int) -> bool:
        """
        Check whether (square, num) is a legal move for the current player,
        ignoring the oracle / taboo-from-unsolvable part.
        """
        board = game_state.board
        row, col = square
        N = board.N

        # Cell must be empty
        if board.get((row, col)) != SudokuBoard.empty:
            return False

        # Not a taboo move
        if TabooMove((row, col), num) in game_state.taboo_moves:
            return False

        # Must be in the current player's allowed cells
        if (row, col) not in game_state.player_squares():
            return False

        # Row constraint
        for j in range(N):
            if board.get((row, j)) == num:
                return False

        # Column constraint
        for i in range(N):
            if board.get((i, col)) == num:
                return False

        # Block constraint
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

        # If no allowed squares are registered, fall back to whole board
        if not playable_squares:
            playable_squares = [(i, j) for i in range(N) for j in range(N)]

        moves = []
        for square in playable_squares:
            for value in range(1, N + 1):
                if self._valid_move(game_state, square, value):
                    moves.append(Move(square, value))

        return moves

    def _neighbors_of(self, state: GameState, square: tuple):
        """
        8-neighborhood of a square, inside board.
        """
        N = state.board.N
        r, c = square
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    yield (nr, nc)

    def _regions_completed(self, board: SudokuBoard, square: tuple) -> int:
        """
        How many regions (row, column, block) are completed by this move?
        Assumes the board ALREADY contains the move at 'square'.
        Returns: 0, 1, 2, or 3.
        """
        row, col = square
        N = board.N

        # Row complete?
        row_complete = all(board.get((row, c)) != SudokuBoard.empty for c in range(N))

        # Column complete?
        col_complete = all(board.get((r, col)) != SudokuBoard.empty for r in range(N))

        # Block complete?
        rh = board.region_height()
        rw = board.region_width()
        start_row = (row // rh) * rh
        start_col = (col // rw) * rw

        block_complete = True
        for r in range(start_row, start_row + rh):
            for c in range(start_col, start_col + rw):
                if board.get((r, c)) == SudokuBoard.empty:
                    block_complete = False
                    break
            if not block_complete:
                break

        completed = sum([row_complete, col_complete, block_complete])
        return completed

    def _points_calculation(self, completed: int) -> int:
        """
        Convert number of completed regions into points, as per rules:
          0 -> 0, 1 -> 1, 2 -> 3, 3 -> 7
        """
        if completed == 0:
            return 0
        elif completed == 1:
            return 1
        elif completed == 2:
            return 3
        elif completed == 3:
            return 7
        return 0

    def _apply_move(self, state: GameState, move: Move) -> GameState:
        """
        Return a NEW GameState that results from playing 'move' in 'state',
        using an internal approximation of how the game evolves.
        This does NOT call the oracle and is only used inside search.
        """
        new_state = copy.deepcopy(state)
        current_player = new_state.current_player

        # Place the value on the board
        new_state.board.put(move.square, move.value)

        # Update score according to completed regions
        completed = self._regions_completed(new_state.board, move.square)
        gained = self._points_calculation(completed)
        new_state.scores[current_player - 1] += gained

        # Record move
        new_state.moves.append(move)

        # Update occupied squares
        if current_player == 1:
            if new_state.occupied_squares1 is None:
                new_state.occupied_squares1 = []
            new_state.occupied_squares1.append(move.square)
        else:
            if new_state.occupied_squares2 is None:
                new_state.occupied_squares2 = []
            new_state.occupied_squares2.append(move.square)

        # Update allowed squares by expanding from the played move
        neighbors = [sq for sq in self._neighbors_of(new_state, move.square)
                     if new_state.board.get(sq) == SudokuBoard.empty]

        if current_player == 1:
            if new_state.allowed_squares1 is None:
                new_state.allowed_squares1 = []
            new_state.allowed_squares1.extend(neighbors)
            new_state.allowed_squares1 = list(set(new_state.allowed_squares1))
        else:
            if new_state.allowed_squares2 is None:
                new_state.allowed_squares2 = []
            new_state.allowed_squares2.extend(neighbors)
            new_state.allowed_squares2 = list(set(new_state.allowed_squares2))

        # Clean allowed squares: remove filled cells for both players
        if getattr(new_state, "allowed_squares1", None) is not None:
            new_state.allowed_squares1 = [
                sq for sq in new_state.allowed_squares1
                if new_state.board.get(sq) == SudokuBoard.empty
            ]
        if getattr(new_state, "allowed_squares2", None) is not None:
            new_state.allowed_squares2 = [
                sq for sq in new_state.allowed_squares2
                if new_state.board.get(sq) == SudokuBoard.empty
            ]

        # Switch current player
        new_state.current_player = 3 - current_player

        return new_state

    def evaluate_state(self, state: GameState) -> float:
        """
        Evaluate a GameState from Player 1's perspective.
        Positive = good for player 1, negative = good for player 2.
        """
        # Score difference
        score_first = state.scores[0]
        score_second = state.scores[1]
        score_diff = score_first - score_second

        # Territory (allowed squares)
        allowed1 = len(getattr(state, "allowed_squares1", []) or [])
        allowed2 = len(getattr(state, "allowed_squares2", []) or [])
        territory_diff = allowed1 - allowed2

        # Simple heuristic: score difference dominates, territory is secondary
        value = 10.0 * score_diff + 0.5 * territory_diff
        return value

    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float):
        """
        Depth-limited minimax with alpha-beta pruning.
        Evaluation is always from Player 1's perspective.
        - When it's Player 1's turn: we MAXIMIZE.
        - When it's Player 2's turn: we MINIMIZE.
        Returns: (value, best_move)
        """
        legal_moves = self.generate_legal_moves(state)

        # Terminal or depth limit
        if depth == 0 or not legal_moves:
            return self.evaluate_state(state), None

        maximizing = (state.current_player == 1)

        if maximizing:
            best_val = float("-inf")
            best_move = None
            for move in legal_moves:
                child = self._apply_move(state, move)
                val, _ = self._minimax(child, depth - 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break  # beta cut-off
            return best_val, best_move
        else:
            best_val = float("inf")
            best_move = None
            for move in legal_moves:
                child = self._apply_move(state, move)
                val, _ = self._minimax(child, depth - 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)
                if beta <= alpha:
                    break  # alpha cut-off
            return best_val, best_move

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Anytime-style:
        1. Generate all legal moves.
        2. Immediately propose a random legal move as a safe fallback.
        3. Run a small minimax search (depth 2 by default) on a COPY of the state.
        4. If minimax finds something better, propose that as the new best move.
        5. Keep re-proposing the best move until the engine kills this process.
        """

        legal_moves = self.generate_legal_moves(game_state)

        if not legal_moves:
            # No legal moves -> framework will handle skip or loss
            return

        # Fallback: random safe move
        best_move = random.choice(legal_moves)
        self.propose_move(best_move)

        # Minimax search on a copy of the state
        try:
            root_state = copy.deepcopy(game_state)
            depth = 3 # Depth 3 seems to be working better
            value, move = self._minimax(root_state, depth, float("-inf"), float("inf"))
            logging.info(f"Minimax value {value} for player {game_state.current_player}")

            if move is not None:
                best_move = move
                self.propose_move(best_move)
        except Exception as e:
            logging.error(f"Error in minimax: {e}")

        # Re-proposing best_move
        while True:
            time.sleep(0.2)
            self.propose_move(best_move)
