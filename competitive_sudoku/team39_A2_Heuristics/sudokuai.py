import time
import copy
import random
import logging

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

logging.basicConfig(level=logging.INFO)


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    A2 – Heuristic Negamax Alpha-Beta with safe beam + threat-aware ordering.

    Fix vs your previous A2:
    - Uses NEGAMAX so we never mess up player perspective signs.
    - Threat penalty is applied consistently.
    - Beam is only used when branching is huge; otherwise full expansion.
    - Iterative deepening anytime: keeps improving depth.
    """

    # ---- Tunables ----
    # Beam only when branching is huge (9x9 midgame). For 4x4 it will usually expand all.
    BEAM_TRIGGER = 35
    BEAM_WIDTH = 18
    EXTRA_RANDOM = 3

    # Threat handling
    OPP_THREAT_SAMPLE = 80
    THREAT_PENALTY = 12.0   # stronger than before (your logs show opponent getting 7 often)

    # Small nudges (don’t overdo, minimax should decide)
    SELF_REWARD_BONUS = 1.0
    TERRITORY_BONUS = 0.15

    def __init__(self):
        super().__init__()
        self.rng = random.Random()

    # ============================================================
    # Legal move generation
    # ============================================================

    def _valid_move(self, state: GameState, square: tuple, value: int) -> bool:
        board = state.board
        r, c = square
        N = board.N

        if board.get((r, c)) != SudokuBoard.empty:
            return False
        if TabooMove(square, value) in state.taboo_moves:
            return False
        if square not in state.player_squares():
            return False

        for j in range(N):
            if board.get((r, j)) == value:
                return False
        for i in range(N):
            if board.get((i, c)) == value:
                return False

        h = board.region_height()
        w = board.region_width()
        sr = (r // h) * h
        sc = (c // w) * w
        for i in range(sr, sr + h):
            for j in range(sc, sc + w):
                if board.get((i, j)) == value:
                    return False

        return True

    def generate_legal_moves(self, state: GameState):
        N = state.board.N
        squares = state.player_squares()
        if not squares:
            squares = [(i, j) for i in range(N) for j in range(N)]

        moves = []
        for sq in squares:
            for v in range(1, N + 1):
                if self._valid_move(state, sq, v):
                    moves.append(Move(sq, v))
        return moves

    # ============================================================
    # Apply move (simulation)
    # ============================================================

    def _neighbors(self, board: SudokuBoard, square):
        N = board.N
        r, c = square
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    yield (nr, nc)

    def _count_completed_regions_after_place(self, board: SudokuBoard, square) -> int:
        r, c = square
        N = board.N

        row_done = all(board.get((r, j)) != SudokuBoard.empty for j in range(N))
        col_done = all(board.get((i, c)) != SudokuBoard.empty for i in range(N))

        h = board.region_height()
        w = board.region_width()
        sr = (r // h) * h
        sc = (c // w) * w
        block_done = True
        for i in range(sr, sr + h):
            for j in range(sc, sc + w):
                if board.get((i, j)) == SudokuBoard.empty:
                    block_done = False
                    break
            if not block_done:
                break

        return int(row_done) + int(col_done) + int(block_done)

    def _points(self, completed: int) -> int:
        return {0: 0, 1: 1, 2: 3, 3: 7}[completed]

    def _apply_move(self, state: GameState, move: Move) -> GameState:
        s = copy.deepcopy(state)
        p = s.current_player
        r, c = move.square

        s.board.put((r, c), move.value)

        comp = self._count_completed_regions_after_place(s.board, (r, c))
        s.scores[p - 1] += self._points(comp)

        if p == 1:
            s.occupied_squares1.append((r, c))
        else:
            s.occupied_squares2.append((r, c))

        new_allowed = [sq for sq in self._neighbors(s.board, (r, c))
                       if s.board.get(sq) == SudokuBoard.empty]

        if p == 1:
            s.allowed_squares1 = list(set((s.allowed_squares1 or []) + new_allowed))
            s.allowed_squares1 = [sq for sq in s.allowed_squares1 if s.board.get(sq) == SudokuBoard.empty]
        else:
            s.allowed_squares2 = list(set((s.allowed_squares2 or []) + new_allowed))
            s.allowed_squares2 = [sq for sq in s.allowed_squares2 if s.board.get(sq) == SudokuBoard.empty]

        s.current_player = 3 - p
        return s

    # ============================================================
    # Evaluation
    # ============================================================

    def eval_p1(self, state: GameState) -> float:
        """Static evaluation from Player1's perspective."""
        score_diff = state.scores[0] - state.scores[1]
        terr1 = len(state.allowed_squares1 or [])
        terr2 = len(state.allowed_squares2 or [])
        return 10.0 * score_diff + 0.5 * (terr1 - terr2)

    def eval_current_player(self, state: GameState) -> float:
        """Convert eval_p1 into current player's perspective (for negamax)."""
        return self.eval_p1(state) if state.current_player == 1 else -self.eval_p1(state)

    # ============================================================
    # Threat + ordering heuristics (consistent with negamax)
    # ============================================================

    def _immediate_reward_if_played(self, state: GameState, move: Move) -> int:
        """Immediate points gained by this move (0/1/3/7), without mutating board."""
        board = state.board
        N = board.N
        r, c = move.square

        completed = 0

        if all(board.get((r, j)) != SudokuBoard.empty for j in range(N) if j != c):
            completed += 1
        if all(board.get((i, c)) != SudokuBoard.empty for i in range(N) if i != r):
            completed += 1

        h = board.region_height()
        w = board.region_width()
        sr = (r // h) * h
        sc = (c // w) * w
        block_ok = True
        for i in range(sr, sr + h):
            for j in range(sc, sc + w):
                if (i, j) == (r, c):
                    continue
                if board.get((i, j)) == SudokuBoard.empty:
                    block_ok = False
                    break
            if not block_ok:
                break
        if block_ok:
            completed += 1

        return self._points(completed)

    def _opponent_best_immediate_reward(self, state_after_my_move: GameState) -> int:
        """After my move, opponent to play: how many points can they get immediately (max)?"""
        opp_moves = self.generate_legal_moves(state_after_my_move)
        if not opp_moves:
            return 0

        if len(opp_moves) > self.OPP_THREAT_SAMPLE:
            self.rng.shuffle(opp_moves)
            opp_moves = opp_moves[:self.OPP_THREAT_SAMPLE]

        best = 0
        for m in opp_moves:
            r = self._immediate_reward_if_played(state_after_my_move, m)
            if r > best:
                best = r
                if best == 7:
                    break
        return best

    def _territory_delta_current_player(self, before: GameState, after: GameState) -> int:
        p = before.current_player
        if p == 1:
            return len(after.allowed_squares1 or []) - len(before.allowed_squares1 or [])
        else:
            return len(after.allowed_squares2 or []) - len(before.allowed_squares2 or [])

    def order_score(self, state: GameState, move: Move) -> float:
        """
        Higher is better for the CURRENT player (consistent with negamax).
        """
        my_reward = self._immediate_reward_if_played(state, move)
        child = self._apply_move(state, move)

        base = self.eval_current_player(child)  # already converted into player's perspective
        opp_threat = self._opponent_best_immediate_reward(child)
        terr_delta = self._territory_delta_current_player(state, child)

        return (
            base
            + self.SELF_REWARD_BONUS * my_reward
            + self.TERRITORY_BONUS * terr_delta
            - self.THREAT_PENALTY * opp_threat
        )

    def ordered_moves(self, state: GameState):
        moves = self.generate_legal_moves(state)
        if not moves:
            return []

        # Compute heuristic scores (current player's perspective)
        scored = [(self.order_score(state, m), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)  # higher is always better for side to move

        # Beam only when big branching
        if len(moves) >= self.BEAM_TRIGGER:
            main = [m for (_, m) in scored[:self.BEAM_WIDTH]]
            tail = [m for (_, m) in scored[self.BEAM_WIDTH:]]
            self.rng.shuffle(tail)
            main += tail[:self.EXTRA_RANDOM]
            return main

        # Otherwise, full expansion (important for small boards / endgame)
        return [m for (_, m) in scored]

    # ============================================================
    # Negamax alpha-beta
    # ============================================================

    def negamax(self, state: GameState, depth: int, alpha: float, beta: float, deadline: float):
        if time.time() >= deadline:
            return self.eval_current_player(state), None

        moves = self.generate_legal_moves(state)
        if depth == 0 or not moves:
            return self.eval_current_player(state), None

        best_val = float("-inf")
        best_move = None

        for m in self.ordered_moves(state):
            child = self._apply_move(state, m)
            val, _ = self.negamax(child, depth - 1, -beta, -alpha, deadline)
            val = -val

            if val > best_val:
                best_val = val
                best_move = m

            alpha = max(alpha, val)
            if alpha >= beta or time.time() >= deadline:
                break

        return best_val, best_move

    # ============================================================
    # Anytime compute_best_move (iterative deepening)
    # ============================================================

    def compute_best_move(self, game_state: GameState) -> None:
        legal = self.generate_legal_moves(game_state)
        if not legal:
            return

        # immediate fallback
        best = self.rng.choice(legal)
        self.propose_move(best)

        start = time.time()
        deadline = start + 0.9  # safe slice (works even with small think-times)

        depth = 1
        last_best = best

        while time.time() < deadline:
            val, move = self.negamax(copy.deepcopy(game_state), depth, float("-inf"), float("inf"), deadline)
            if time.time() >= deadline:
                break
            if move is not None:
                last_best = move
                self.propose_move(last_best)
            depth += 1

        best = last_best
        while True:
            time.sleep(0.2)
            self.propose_move(best)
