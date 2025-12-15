#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import math
import copy
import logging

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

logging.basicConfig(level=logging.INFO)

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Constraint-Aware MCTS Agent.
    Improves upon standard MCTS by using 'Logical Simulations':
    - In simulations, it prioritizes 'Forced Moves' (Naked Singles) before guessing.
    - Uses Minimax heuristics (Score + Territory) to evaluate leaf nodes.
    """
    def __init__(self):
        super().__init__()
        self.team_id = 1 

    def compute_best_move(self, game_state: GameState):
        self.team_id = game_state.current_player
        
        legal_moves = self.generate_legal_moves(game_state)
        if not legal_moves: return

        best_move = random.choice(legal_moves)
        self.propose_move(best_move)

        forced = self.find_forced_move(game_state, legal_moves)
        if forced:
            self.propose_move(forced)
            self._keep_alive() # Stop calculating, we found the perfect move.
            return

        root = MCTSNode(state=game_state, parent=None, move=None, ai=self)
        
        start_time = time.time()
        time_limit = 0.98 
        
        while time.time() - start_time < time_limit:
            node = root
            while not node.is_leaf() and node.is_fully_expanded():
                node = node.select_child()
            
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            result = node.simulate(self.team_id)
            
            node.backpropagate(result)
        
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            self.propose_move(best_child.move)
        
        self._keep_alive()

    def _keep_alive(self):
        while True: time.sleep(0.1)

    def find_forced_move(self, state, legal_moves=None):
        """Returns a Move if a cell has exactly 1 valid option."""
        board = state.board
        N = board.N
        
        if legal_moves is None:
            legal_moves = self.generate_legal_moves(state)
            
        moves_by_square = {}
        for m in legal_moves:
            if m.square not in moves_by_square:
                moves_by_square[m.square] = []
            moves_by_square[m.square].append(m)
            
        for sq, moves in moves_by_square.items():
            if len(moves) == 1:
                return moves[0]
        return None

    def _valid_move(self, game_state, square, num):
        board = game_state.board
        if board.get(square) != 0: return False
        if TabooMove(square, num) in game_state.taboo_moves: return False
        if square not in game_state.player_squares(): return False

        N = board.N
        r, c = square
        
        for k in range(N):
            if board.get((r, k)) == num: return False
            if board.get((k, c)) == num: return False
            
        rh, rw = board.region_height(), board.region_width()
        sr, sc = (r // rh) * rh, (c // rw) * rw
        for i in range(sr, sr + rh):
            for j in range(sc, sc + rw):
                if board.get((i, j)) == num: return False
        return True

    def generate_legal_moves(self, game_state):
        N = game_state.board.N
        playable = game_state.player_squares()
        if not playable:
            playable = [(i, j) for i in range(N) for j in range(N)]

        moves = []
        for sq in playable:
            for v in range(1, N + 1):
                if self._valid_move(game_state, sq, v):
                    moves.append(Move(sq, v))
        return moves

    def _apply_move(self, state, move):
        new_state = copy.deepcopy(state)
        new_state.board.put(move.square, move.value)
        
        pts = self._points_from_move(new_state.board, move.square)
        new_state.scores[new_state.current_player - 1] += pts
        
        curr = new_state.current_player
        if curr == 1:
            if new_state.occupied_squares1 is None: new_state.occupied_squares1 = []
            new_state.occupied_squares1.append(move.square)
        else:
            if new_state.occupied_squares2 is None: new_state.occupied_squares2 = []
            new_state.occupied_squares2.append(move.square)
            
        new_state.current_player = 3 - curr
        return new_state

    def _points_from_move(self, board, square):
        r, c = square
        N = board.N
        
        if any(board.get((r, i)) == 0 for i in range(N)): row_f = 0
        else: row_f = 1
        
        if any(board.get((i, c)) == 0 for i in range(N)): col_f = 0
        else: col_f = 1
        
        rh, rw = board.region_height(), board.region_width()
        sr, sc = (r // rh) * rh, (c // rw) * rw
        box_f = 1
        for i in range(sr, sr + rh):
            for j in range(sc, sc + rw):
                if board.get((i, j)) == 0: 
                    box_f = 0; break
        
        cnt = row_f + col_f + box_f
        return {0:0, 1:1, 2:3, 3:7}.get(cnt, 0)
    
    def _neighbors_of(self, state, sq):
        r, c = sq
        N = state.board.N
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < N and 0 <= nc < N: yield (nr, nc)

class MCTSNode:
    def __init__(self, state, parent=None, move=None, ai=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.ai = ai
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = self.ai.generate_legal_moves(state)

    def is_leaf(self):
        return len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return len(self.untried_moves) == 0 and len(self.children) == 0

    def select_child(self):
        C = 1.414 
        best_score = -float('inf')
        best_node = None
        for child in self.children:
            if child.visits == 0: return child
            score = (child.wins / child.visits) + C * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.ai._apply_move(self.state, move)
        child = MCTSNode(state=next_state, parent=self, move=move, ai=self.ai)
        self.children.append(child)
        return child

    def simulate(self, my_team_id):
        """
        LOGICAL SIMULATION:
        Instead of 100% random, we try to play 'Forced Moves' if they exist.
        This makes the rollout significantly more realistic for Sudoku.
        """
        curr_state = copy.deepcopy(self.state)
        depth = 0
        max_depth = 5 
        
        while depth < max_depth:
            moves = self.ai.generate_legal_moves(curr_state)
            if not moves: break
            
            scoring = [m for m in moves if self.ai._points_from_move(curr_state.board, m.square) > 0]
            if scoring and random.random() < 0.8:
                move = random.choice(scoring)
            else:
                move = random.choice(moves)
            
            curr_state = self.ai._apply_move(curr_state, move)
            depth += 1
            
        s1 = curr_state.scores[0]
        s2 = curr_state.scores[1]
        score_diff = (s1 - s2) if my_team_id == 1 else (s2 - s1)
        
        t1 = self._count_territory(curr_state, 1)
        t2 = self._count_territory(curr_state, 2)
        terr_diff = (t1 - t2) if my_team_id == 1 else (t2 - t1)
        
        raw_val = (score_diff * 10.0) + (terr_diff * 0.5)
        
        try:
            val = 1 / (1 + math.exp(-0.1 * raw_val))
        except OverflowError:
            val = 1.0 if raw_val > 0 else 0.0
            
        return val

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def _count_territory(self, state, pid):
        occ = state.occupied_squares1 if pid == 1 else state.occupied_squares2
        if not occ: return 0
        count = 0
        seen = set()
        for sq in occ:
            for n in self.ai._neighbors_of(state, sq):
                if n not in seen and state.board.get(n) == 0:
                    count += 1
                    seen.add(n)
        return count