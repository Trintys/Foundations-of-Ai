import random
import os
import copy
import logging
import torch
import torch.nn as nn

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai
logging.basicConfig(level=logging.INFO)

from .LDNet import LDNet

MODEL_FILE = "./team39_A2_LDLearning/model/ldnet_state.pth"

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Competitive Sudoku AI using minimax + alpha-beta pruning.
    Uses LDNet as evaluation at leaf nodes.
    """
    def __init__(self):
        super().__init__()

        from .LDNet import LDNet
        self.model = LDNet(lr=5e-3, discount_rate=0.6)

        try:
            self.model.load_state_dict(torch.load(MODEL_FILE))
            logging.info("Loaded LDNet model from state_dict.")
        except Exception:
            logging.info(F"No saved LDNet found — starting fresh. {MODEL_FILE}")

        self.model.eval()
    

    # -----------------------------------------------------------
    # Move legality and generation
    # -----------------------------------------------------------

    def _valid_move(self, game_state: GameState, square: tuple, num: int) -> bool:
        """
        Check whether (square, num) is a legal move for the current player,
        following the AI logic: empty, not taboo, in allowed squares, and Sudoku rules.
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

        # Must be in current player's allowed squares
        if (row, col) not in game_state.player_squares():
            return False

        # Sudoku row check
        for j in range(N):
            if board.get((row, j)) == num:
                return False

        # Sudoku column check
        for i in range(N):
            if board.get((i, col)) == num:
                return False

        # Sudoku subgrid check
        region_h = board.region_height()
        region_w = board.region_width()
        start_row = (row // region_h) * region_h
        start_col = (col // region_w) * region_w

        for i in range(start_row, start_row + region_h):
            for j in range(start_col, start_col + region_w):
                if board.get((i, j)) == num:
                    return False

        return True

    def generate_legal_moves(self, game_state: GameState):
        """
        Returns all legal moves for the current player.
        """
        N = game_state.board.N
        playable_squares = [(i, j) for i in range(N) for j in range(N)
                    if game_state.board.get((i, j)) == SudokuBoard.empty]
        
        # If no allowed squares, fall back to **all empty cells**.
        if not playable_squares:
            playable_squares = [(i, j) for i in range(N) for j in range(N)
                                if game_state.board.get((i, j)) == SudokuBoard.empty]

        moves = []
        for square in playable_squares:
            for value in range(1, N + 1):
                # only check row/col/block constraints
                if self._valid_move(game_state, square, value):
                    moves.append(Move(square, value))
        return moves


    # -----------------------------------------------------------
    # Internal game simulation for minimax
    # -----------------------------------------------------------

    def _neighbors_of(self, state, sq):
        r, c = sq
        N = state.board.N
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    yield (nr, nc)

    def _regions_completed(self, board, square):
        row, col = square
        N = board.N

        row_complete = all(board.get((row, j)) != SudokuBoard.empty for j in range(N))
        col_complete = all(board.get((i, col)) != SudokuBoard.empty for i in range(N))

        rh = board.region_height()
        rw = board.region_width()
        sr = (row // rh) * rh
        sc = (col // rw) * rw

        block_complete = True
        for i in range(sr, sr + rh):
            for j in range(sc, sc + rw):
                if board.get((i, j)) == SudokuBoard.empty:
                    block_complete = False
                    break

        return sum([row_complete, col_complete, block_complete])

    def _points_calculation(self, comp):
        return {0:0, 1:1, 2:3, 3:7}.get(comp, 0)

    def _apply_move(self, state: GameState, move: Move) -> GameState:
        """
        Apply move and return new GameState. 
        Updates scores, allowed_squares, and current_player.
        """
        new_state = copy.deepcopy(state)
        p = new_state.current_player
        new_state.board.put(move.square, move.value)

        # Update score based on completed regions
        comp = self._regions_completed(new_state.board, move.square)
        new_state.scores[p - 1] += self._points_calculation(comp)

        # Append to moves
        new_state.moves.append(move)

        # Update allowed squares for the next move
        N = new_state.board.N
        rh = new_state.board.region_height()
        rw = new_state.board.region_width()

        r, c = move.square
        neighbors = [(nr, nc) for nr in range(max(0, r-1), min(N, r+2))
                            for nc in range(max(0, c-1), min(N, c+2))
                            if new_state.board.get((nr, nc)) == SudokuBoard.empty]

        if p == 1:
            new_state.allowed_squares1 = list(set(new_state.allowed_squares1 + neighbors)) \
                                        if new_state.allowed_squares1 else neighbors
            # Remove filled cells
            new_state.allowed_squares1 = [sq for sq in new_state.allowed_squares1
                                        if new_state.board.get(sq) == SudokuBoard.empty]
        else:
            new_state.allowed_squares2 = list(set(new_state.allowed_squares2 + neighbors)) \
                                        if new_state.allowed_squares2 else neighbors
            new_state.allowed_squares2 = [sq for sq in new_state.allowed_squares2
                                        if new_state.board.get(sq) == SudokuBoard.empty]

        # Switch player
        new_state.current_player = 3 - p
        return new_state        
    # -----------------------------------------------------------
    # Minimax + alpha-beta
    # -----------------------------------------------------------

    # def _minimax(self, state: GameState, depth, alpha, beta):
    #     moves = self.generate_legal_moves(state)

    #     if depth == 0 or not moves:
    #         board_t, score_t = self.model.encode_state(state)
    #         val = self.model.predict(board_t, score_t)
    #         return val, None


    #     maximizing = (state.current_player == 1)

    #     if maximizing:
    #         best_val = float("-inf")
    #         best_move = None
    #         for mv in moves:
    #             child = self._apply_move(state, mv)
    #             val, _ = self._minimax(child, depth - 1, alpha, beta)
    #             if val > best_val:
    #                 best_val = val
    #                 best_move = mv
    #             alpha = max(alpha, best_val)
    #             if beta <= alpha:
    #                 break
    #         return best_val, best_move

    #     else:  # minimizing
    #         best_val = float("inf")
    #         best_move = None
    #         for mv in moves:
    #             child = self._apply_move(state, mv)
    #             val, _ = self._minimax(child, depth - 1, alpha, beta)
    #             if val < best_val:
    #                 best_val = val
    #                 best_move = mv
    #             beta = min(beta, best_val)
    #             if beta <= alpha:
    #                 break
    #         return best_val, best_move
        
    def _minimax(self, state: GameState, depth, alpha, beta):
        '''
        New version
        '''
        moves = self.generate_legal_moves(state)

        if depth == 0 or not moves:
            board_t, score_t = self.model.encode_state(state)
            val = self.model.predict(board_t, score_t)  # Neural network evaluation
            return val, None

        maximizing = (state.current_player == 1)

        if maximizing:
            best_val = float("-inf")
            best_move = None
            for mv in moves:
                child = self._apply_move(state, mv)
                val, _ = self._minimax(child, depth - 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_move = mv
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_move

        else:  # minimizing
            best_val = float("inf")
            best_move = None
            for mv in moves:
                child = self._apply_move(state, mv)
                val, _ = self._minimax(child, depth - 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_move = mv
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_val, best_move



    # -----------------------------------------------------------
    # Interface for the competitive engine
    # -----------------------------------------------------------
    def ensure_dir(self, path: str):
        p = os.path.dirname(path)
        if p and not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

    def compute_best_move(self, game_state: GameState) -> None:
        
        if self.model is None:
                if self.ensure_dir(MODEL_FILE):
                    try:
                        # Using state_dict (PyTorch recommended)
                        self.model.load_state_dict(torch.load(MODEL_FILE))
                        logging.info(f"Loaded LDNet model from {MODEL_FILE}")
                    except Exception as e:
                        logging.error(f"Failed to load LDNet model: {e}")
                else:
                    self.model = LDNet(lr=5e-3, discount_rate=0.6, device="cpu")
                    logging.info(f"No existing model found; starting with a fresh LDNet {MODEL_FILE}")
                self.model.eval()  # evaluation mode


        
        legal = self.generate_legal_moves(game_state)
        if not legal:
            return

        best_move = random.choice(legal)
        self.propose_move(best_move)

        try:
            root_state = copy.deepcopy(game_state)
            depth = 2 # Depth 3 seems to be working better
            value, move = self._minimax(root_state, depth, float("-inf"), float("inf"))
            
            if move is not None:
                best_move = move
                logging.info(f)
                self.propose_move(best_move)

        except Exception as e:
            logging.error(f"Error in minimax: {e}")

        # Re-proposing best_move
        # while True:
        #     time.sleep(0.2)
        #     self.propose_move(best_move)
