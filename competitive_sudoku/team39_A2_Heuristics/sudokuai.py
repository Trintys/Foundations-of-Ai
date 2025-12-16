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
    Assignment 2 - Heuristic Minimax Agent

    Based on your A1 agent, extended with:
    - Heuristic move scoring and filtering
    - Move ordering for better alpha–beta pruning
    - Depth adaptation based on branching factor

    Evaluation is still from Player 1's perspective:
      eval(state) > 0 : good for player 1
      eval(state) < 0 : good for player 2
    """

    def __init__(self):
        super().__init__()
        self.rng = random.Random()


    # ============================================================
    # 1. LEGAL MOVE GENERATION (same logic as A1)
    # ============================================================

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


    # ============================================================
    # 2. NEIGHBORS + SCORE CALCULATION (from A1)
    # ============================================================

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


    # ============================================================
    # 3. APPLY MOVE (STATE SIMULATION, from A1)
    # ============================================================

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


    # ============================================================
    # 4. EVALUATION (same as A1, from Player 1 perspective)
    # ============================================================

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


    # ============================================================
    # 5. HEURISTICS: move scoring, forced moves, candidate selection
    # ============================================================

    def _immediate_reward_if_played(self, state: GameState, move: Move) -> int:
        """
        Approximate reward of playing `move` in `state` without modifying it.

        We don't actually put the value on the board: we just check
        whether all *other* cells in row/col/block are already filled.
        """
        board = state.board
        N = board.N
        r, c = move.square

        completed = 0

        # Row complete if all other cells in row are non-empty
        row_full = True
        for j in range(N):
            if j == c:
                continue
            if board.get((r, j)) == SudokuBoard.empty:
                row_full = False
                break
        if row_full:
            completed += 1

        # Column complete if all other cells in column are non-empty
        col_full = True
        for i in range(N):
            if i == r:
                continue
            if board.get((i, c)) == SudokuBoard.empty:
                col_full = False
                break
        if col_full:
            completed += 1

        # Block complete?
        h = board.region_height()
        w = board.region_width()
        start_r = (r // h) * h
        start_c = (c // w) * w

        block_full = True
        for i in range(start_r, start_r + h):
            for j in range(start_c, start_c + w):
                if (i, j) == (r, c):
                    continue
                if board.get((i, j)) == SudokuBoard.empty:
                    block_full = False
                    break
            if not block_full:
                break
        if block_full:
            completed += 1

        return self._points_calculation(completed)

    def _move_heuristic_score(self, state: GameState, move: Move) -> float:
        """
        Assign a heuristic score to a move, for move ordering and filtering.

        Components:
        - Immediate points if played
        - How constrained the position is (more filled neighbors = more tactical)
        - Proximity to own occupied territory
        - Centrality (closer to center is slightly better)
        """
        board = state.board
        N = board.N
        r, c = move.square

        reward = self._immediate_reward_if_played(state, move)

        # Center distance
        center = (N - 1) / 2.0
        dist_center = abs(r - center) + abs(c - center)
        centrality_score = -dist_center  # closer to center is better

        # Proximity to own territory
        if state.current_player == 1:
            occ = getattr(state, "occupied_squares1", []) or []
        else:
            occ = getattr(state, "occupied_squares2", []) or []

        if occ:
            min_dist_occ = min(abs(r - rr) + abs(c - cc) for (rr, cc) in occ)
            proximity_score = -min_dist_occ
        else:
            proximity_score = 0.0

        # Constraint score: number of distinct values in row/col/block
        numbers_row = {
            board.get((r, j))
            for j in range(N)
            if board.get((r, j)) != SudokuBoard.empty
        }
        numbers_col = {
            board.get((i, c))
            for i in range(N)
            if board.get((i, c)) != SudokuBoard.empty
        }
        h = board.region_height()
        w = board.region_width()
        start_r = (r // h) * h
        start_c = (c // w) * w
        numbers_block = {
            board.get((i, j))
            for i in range(start_r, start_r + h)
            for j in range(start_c, start_c + w)
            if board.get((i, j)) != SudokuBoard.empty
        }
        constraint_score = len(numbers_row | numbers_col | numbers_block)

        return (
            5.0 * reward +
            1.5 * constraint_score +
            0.5 * proximity_score +
            0.2 * centrality_score
        )

    def _find_forced_moves(self, legal_moves):
        """
        Forced moves = cells with exactly one legal value.
        Those are very strong candidates (likely in final solution).
        """
        by_cell = {}
        for m in legal_moves:
            by_cell.setdefault(m.square, []).append(m)

        forced = []
        for sq, moves in by_cell.items():
            if len(moves) == 1:
                forced.append(moves[0])
        return forced

    def _select_candidate_moves(self, state: GameState, legal_moves, max_candidates=12):
        """
        Heuristic move filtering:
        - If there are forced moves, only consider those.
        - Otherwise:
          - Score all moves with _move_heuristic_score
          - Take top-K moves
          - Add a couple random extra moves for exploration
        """
        if not legal_moves:
            return []

        forced = self._find_forced_moves(legal_moves)
        if forced:
            return forced

        scored = [(self._move_heuristic_score(state, m), m) for m in legal_moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        top_k = [m for (_, m) in scored[:max_candidates]]

        remaining = [m for (_, m) in scored[max_candidates:]]
        self.rng.shuffle(remaining)
        extra = remaining[:2]

        candidates = top_k + extra

        # Deduplicate (square,value)
        seen = set()
        result = []
        for m in candidates:
            key = (m.square, m.value)
            if key not in seen:
                seen.add(key)
                result.append(m)

        return result


    # ============================================================
    # 6. MINIMAX + ALPHA-BETA with HEURISTIC CHILD SELECTION
    # ============================================================

    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float):
        """
        Depth-limited minimax with alpha-beta pruning and heuristic child selection.
        Evaluation is always from Player 1's perspective:
        - Player 1 turn: maximize
        - Player 2 turn: minimize
        """
        legal_moves = self.generate_legal_moves(state)

        # Terminal or depth limit
        if depth == 0 or not legal_moves:
            return self.evaluate_state(state), None

        maximizing = (state.current_player == 1)

        # Heuristic narrowing of moves
        candidate_moves = self._select_candidate_moves(state, legal_moves)

        # Fallback if heuristic kills all moves (very unlikely)
        if not candidate_moves:
            candidate_moves = legal_moves

        if maximizing:
            best_val = float("-inf")
            best_move = None
            # moves are already "ordered" by heuristic in _select_candidate_moves
            for move in candidate_moves:
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
            for move in candidate_moves:
                child = self._apply_move(state, move)
                val, _ = self._minimax(child, depth - 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_move = move
                beta = min(beta, best_val)
                if beta <= alpha:
                    break  # alpha cut-off

            return best_val, best_move


    # ============================================================
    # 7. ANYTIME-STYLE MOVE SELECTION (A2 version)
    # ============================================================

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Anytime-style:
        1. Generate all legal moves.
        2. Immediately propose a random legal move as a safe fallback.
        3. Choose depth based on branching factor.
        4. Run heuristic minimax on a COPY of the state.
        5. If minimax finds something better, propose that as the new best move.
        6. Keep re-proposing the best move until the engine kills this process.
        """

        legal_moves = self.generate_legal_moves(game_state)

        if not legal_moves:
            # No legal moves -> framework will handle skip or loss
            return

        # Fallback: random safe move
        best_move = self.rng.choice(legal_moves)
        self.propose_move(best_move)

        # Decide depth based on branching factor
        num_moves = len(legal_moves)
        if num_moves > 40:
            depth = 2
        elif num_moves > 20:
            depth = 3
        else:
            depth = 3  # could try 4 in endgame, but be careful with time limits

        logging.info(f"Heuristic minimax start: num_moves={num_moves}, depth={depth}")

        # Minimax search on a copy of the state
        try:
            root_state = copy.deepcopy(game_state)
            value, move = self._minimax(root_state, depth, float("-inf"), float("inf"))

            logging.info(f"Heuristic minimax value {value} for player {game_state.current_player}")

            if move is not None:
                best_move = move
                self.propose_move(best_move)

        except Exception as e:
            logging.error(f"Error in heuristic minimax: {e}")

        # Re-proposing best_move (anytime)
        while True:
            time.sleep(0.2)
            self.propose_move(best_move)
