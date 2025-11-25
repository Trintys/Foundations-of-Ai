import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
import logging
import copy

logging.basicConfig(level=logging.INFO)
class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Assignment 1 Competitive Sudoku AI.
    Uses:
    - Legal move generator
    - Simulate move
    - Minimax + alpha-beta pruning
    - Anytime move selection
    """

    def __init__(self):
        super().__init__()


    # ============================================================
    # 1. LEGAL MOVE GENERATION (Friend 1's part)
    # ============================================================

    def _is_valid_move(self, game_state: GameState, square: tuple, num: int) -> bool:
        """
        Checks if placing `num` at `square` obeys:
        - empty cell
        - not taboo
        - allowed square rules
        - Sudoku row/column/subgrid uniqueness
        """
        board = game_state.board
        row, col = square
        N = board.N

        # Empty and allowed
        if board.get((row, col)) != SudokuBoard.empty:
            return False
        if TabooMove((row, col), num) in game_state.taboo_moves:
            return False
        if (row, col) not in game_state.player_squares():
            return False

        # Row
        for j in range(N):
            if board.get((row, j)) == num:
                return False

        # Column
        for i in range(N):
            if board.get((i, col)) == num:
                return False

        # Subgrid
        h = board.region_height()
        w = board.region_width()
        start_r = (row // h) * h
        start_c = (col // w) * w

        for i in range(start_r, start_r + h):
            for j in range(start_c, start_c + w):
                if board.get((i, j)) == num:
                    return False

        return True


    def generate_legal_moves(self, game_state: GameState):
        """
        Generate all legal (square,value) moves for the current player.
        """
        N = game_state.board.N
        playable = game_state.player_squares()

        if playable is None:
            playable = [(i, j) for i in range(N) for j in range(N)]

        moves = []
        for (r, c) in playable:
            for val in range(1, N + 1):
                if self._is_valid_move(game_state, (r, c), val):
                    moves.append(Move((r, c), val))

        return moves


    # ============================================================
    # 2. EVALUATION FUNCTION (Your part)
    # ============================================================

    def evaluate_state(self, state: GameState, my_moves, opp_moves):
        """
        Higher value = better for our AI.
        Uses:
        - score difference
        - mobility difference
        - board control (allowed region size)
        """

        # Scores
        my_score, opp_score = state.scores
        score_diff = my_score - opp_score

        # Mobility
        mobility = len(my_moves) - len(opp_moves)

        # Board control via allowed squares
        my_allowed = len(state.player_squares())
        try:
            # temporarily switch POV
            state.current_player = 2 if state.current_player == 1 else 1
            opp_allowed = len(state.player_squares())
            state.current_player = 2 if state.current_player == 1 else 1
        except:
            opp_allowed = 0

        board_control = my_allowed - opp_allowed

        return (
            10 * score_diff +
            2 * mobility +
            0.5 * board_control
        )


    # ============================================================
    # 3. SIMULATE MOVE (Correct implementation)
    # ============================================================

    def simulate_move(self, state: GameState, move: Move):
        """
        Simulates the effect of playing a move.
        MUST mimic professor's GameState logic EXACTLY:
        - put value on board
        - update occupied squares
        - expand allowed squares via adjacency
        - compute score (1/3/7)
        - switch player
        """
        new_state = copy.deepcopy(state)

        row, col = move.square
        value = move.value
        board = new_state.board
        N = board.N

        # 1. Place value
        board.put((row, col), value)

        # 2. Update occupied squares
        if new_state.current_player == 1:
            new_state.occupied_squares1.append((row, col))
        else:
            new_state.occupied_squares2.append((row, col))

        # 3. Expand allowed squares (adjacent empty neighbors)
        def neighbors(r, c):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        yield (nr, nc)

        new_adj = [
            sq for sq in neighbors(row, col)
            if board.get(sq) == SudokuBoard.empty
        ]

        if new_state.current_player == 1:
            if new_state.allowed_squares1 is None:
                new_state.allowed_squares1 = []
            new_state.allowed_squares1.extend(new_adj)
            new_state.allowed_squares1 = sorted(set(new_state.allowed_squares1))
        else:
            if new_state.allowed_squares2 is None:
                new_state.allowed_squares2 = []
            new_state.allowed_squares2.extend(new_adj)
            new_state.allowed_squares2 = sorted(set(new_state.allowed_squares2))

        # 4. Score calculation for row/column/subgrid completion
        h = board.region_height()
        w = board.region_width()

        completed = 0

        if all(board.get((row, c)) != SudokuBoard.empty for c in range(N)):
            completed += 1

        if all(board.get((r, col)) != SudokuBoard.empty for r in range(N)):
            completed += 1

        start_r = (row // h) * h
        start_c = (col // w) * w
        if all(
            board.get((r, c)) != SudokuBoard.empty
            for r in range(start_r, start_r + h)
            for c in range(start_c, start_c + w)
        ):
            completed += 1

        reward_table = {0: 0, 1: 1, 2: 3, 3: 7}
        reward = reward_table[completed]

        if new_state.current_player == 1:
            new_state.scores[0] += reward
        else:
            new_state.scores[1] += reward

        # 5. Switch player
        new_state.current_player = 2 if new_state.current_player == 1 else 1

        return new_state


    # ============================================================
    # 4. GET CHILDREN (Correct)
    # ============================================================

    def get_children(self, state: GameState):
        """
        Returns a list of (child_state, move) for every legal move.
        """
        children = []
        legal_moves = self.generate_legal_moves(state)

        for move in legal_moves:
            new_state = copy.deepcopy(state)
            new_state = self.simulate_move(new_state, move)
            children.append((new_state, move))

        return children


    # ============================================================
    # 5. MINIMAX + ALPHA-BETA
    # ============================================================

    def minimax(self, state, depth, alpha, beta, maximizing):
        """
        Recursively search future game states.
        """
        # Get possible moves
        my_moves = self.generate_legal_moves(state)

        # Leaf (depth 0 or no moves)
        if depth == 0 or len(my_moves) == 0:
            # Opponent moves for evaluation
            state.current_player = 2 if state.current_player == 1 else 1
            opp_moves = self.generate_legal_moves(state)
            state.current_player = 2 if state.current_player == 1 else 1

            return self.evaluate_state(state, my_moves, opp_moves), None

        children = self.get_children(state)

        if maximizing:
            best_val = float("-inf")
            best_move = None
            for child_state, move in children:
                value, _ = self.minimax(child_state, depth - 1, alpha, beta, False)
                if value > best_val:
                    best_val = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_val, best_move

        else:
            best_val = float("inf")
            best_move = None
            for child_state, move in children:
                value, _ = self.minimax(child_state, depth - 1, alpha, beta, True)
                if value < best_val:
                    best_val = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_val, best_move


    # ============================================================
    # 6. ANYTIME BEHAVIOR: compute_best_move
    # ============================================================

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Required by framework.  
        Anytime: must always propose moves even if minimax is slow.
        """

        legal_moves = self.generate_legal_moves(game_state)
        if not legal_moves:
            return

        # Random fallback
        fallback = random.choice(legal_moves)
        self.propose_move(fallback)

        # Determine maximizing/minimizing
        maximizing = (game_state.current_player == 1)

        # Run minimax search
        score, best_move = self.minimax(
            game_state,
            depth=3,            # tune later
            alpha=float("-inf"),
            beta=float("inf"),
            maximizing=maximizing
        )

        if best_move is None:
            best_move = fallback

        # Propose best move found
        self.propose_move(best_move)

        # Keep proposing (anytime agent)
        while True:
            time.sleep(0.2)
            self.propose_move(best_move)