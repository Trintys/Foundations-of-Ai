# train_net.py  (updated to use CNN LDNet)
import os
import copy
import logging
import importlib
import random

import torch
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from competitive_sudoku.sudoku import GameState, SudokuBoard

from LDNet import LDNet
from utils import count_completed_regions, display_board
from simulate_game import allowed_squares

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("selfplay_train")

# -------------------------
# Hyperparameters
# -------------------------
MODEL_FILE = "./model/ldnet_state.pth"
REGION_H = 3
REGION_W = 1
TRAINING_EPISODES = 1
MAX_MOVES_PER_EPISODE = 200
MINIMAX_DEPTH = 2
LEARNING_RATE = 5e-3
DISCOUNT = 0.5
VERBOSE = True

# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    p = os.path.dirname(path)
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

# -------------------------
# Initialize board and network
# -------------------------
initial_board = SudokuBoard(REGION_H, REGION_W)
N = initial_board.N
playmode = 'rows'

logger.info(f"Board N = {N}")

# NOTE: new CNN LDNet constructor does not take input_dim
net = LDNet(lr=LEARNING_RATE, discount_rate=DISCOUNT)
if os.path.exists(MODEL_FILE):
    logger.info("Loading existing LDNet state_dict...")
    net.load_state_dict(torch.load(MODEL_FILE))
else:
    logger.info("Created new LDNet.")

net.train()

# -------------------------
# Create two agents (self-play)
# -------------------------
module = importlib.import_module('team39_A2_LDLearning.sudokuai')
player1 = module.SudokuAI()
player2 = module.SudokuAI()
player1.player_number = 1
player2.player_number = 2

# inject same net
player1.model = net
player2.model = net

# -------------------------
# Statistics on players
# -------------------------
total_points_p1 = 0
total_points_p2 = 0
episodes_won_p1 = 0
episodes_won_p2 = 0
episodes_tied = 0

# -------------------------
# Training loop (self-play)
# -------------------------
ensure_dir(MODEL_FILE)

for episode in range(1, TRAINING_EPISODES + 1):
    logger.info(f"=== Episode {episode} ===")

    allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode)
    game_state = GameState(initial_board=initial_board,
                           allowed_squares1=allowed_squares1, occupied_squares1=[],
                           allowed_squares2=allowed_squares2, occupied_squares2=[])
    moves_done = 0

    # per-episode totals for monitoring
    episode_score_p1 = 0
    episode_score_p2 = 0

    while moves_done < MAX_MOVES_PER_EPISODE:
        player_num = game_state.current_player
        agent = player1 if player_num == 1 else player2

        if VERBOSE:
            print(f"Player {player_num} turn, move {moves_done}")
            display_board(game_state)

        # legal moves from current state
        legal_moves = agent.generate_legal_moves(game_state)
        if not legal_moves:
            logger.info(f"Player {player_num} has no legal moves — passing.")
            # skip TD update and reward since no move was made
            game_state.current_player = 3 - game_state.current_player
            continue  # next iteration

        # snapshot before move (for TD target)
        state_before = copy.deepcopy(game_state)

        # minimax (synchronous). fallback to random if something goes wrong
        try:
            value, chosen_move = agent._minimax(game_state, MINIMAX_DEPTH, float("-inf"), float("inf"))
            if chosen_move is None:
                chosen_move = random.choice(legal_moves)
        except Exception:
            logger.exception("Minimax failed, choosing random legal move")
            chosen_move = random.choice(legal_moves)

        # apply move using agent's _apply_move
        next_state = agent._apply_move(game_state, chosen_move)

        # only now give reward and TD update because a move was made
        num_regions = count_completed_regions(next_state.board, chosen_move.square, REGION_H, REGION_W)
        score_table = {0: 0, 1: 1, 2: 3, 3: 7}
        points = score_table.get(num_regions, 0)

        # update scores & occupied lists
        p_idx = player_num - 1
        next_state.scores[p_idx] += points
        if player_num == 1:
            next_state.occupied_squares1.append(chosen_move.square)
            episode_score_p1 += points
        else:
            next_state.occupied_squares2.append(chosen_move.square)
            episode_score_p2 += points

        # --- Encode states for CNN LDNet ---
        s_board, s_score = net.encode_state(state_before)
        ns_board, ns_score = net.encode_state(next_state)

        # TD(0) reward is only for a real move
        reward = 0.1 + points  # optional small bonus to encourage any move

        # TD update
        loss = net.doLDLearning(s_board, s_score, reward, ns_board, ns_score)

        logger.info(f"Episode {episode} Move {moves_done} Player {player_num} Move {chosen_move.square}={chosen_move.value} Reward={reward} Loss={loss:.5f}")

        # advance
        game_state = next_state
        moves_done += 1

        # termination check: no legal moves for both players
        if not (player1.generate_legal_moves(game_state) or player2.generate_legal_moves(game_state)):
            logger.info("No legal moves for both players. Ending episode.")
            break

    logger.info(f"Episode {episode} finished. Episode totals: P1={episode_score_p1}, P2={episode_score_p2}")

    # save model
    torch.save(net.state_dict(), MODEL_FILE)
    logger.info(f"Saved LDNet state_dict to {MODEL_FILE}")

logger.info("Training finished.")

logger.info("=== FINAL SUMMARY ===")
logger.info(f"Total Episodes: {TRAINING_EPISODES}")
logger.info(f"Player 1 total points: {total_points_p1}")
logger.info(f"Player 2 total points: {total_points_p2}")
logger.info(f"Player 1 episode wins: {episodes_won_p1}")
logger.info(f"Player 2 episode wins: {episodes_won_p2}")
logger.info(f"Tied episodes: {episodes_tied}")

