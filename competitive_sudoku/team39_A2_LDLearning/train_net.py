# train_net_improved.py --- Corrected self-play training for LDNet

import os
import copy
import logging
import importlib
import random
from collections import deque

import torch
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from competitive_sudoku.sudoku import GameState, SudokuBoard, Move, TabooMove
from LDNet import LDNet
from utils import count_completed_regions, display_board, board_is_full, is_really_legal, seed_board_with_N_values
from simulate_game import allowed_squares

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("selfplay_train")

# -------------------------
# Hyperparameters
# -------------------------
MODEL_FILE = "./model/ldnet_state.pth"
TRAINING_EPISODES = 1000
MAX_MOVES_PER_EPISODE = 200
LEARNING_RATE = 5e-3
DISCOUNT = 0.95
EPSILON = 0.15  # exploration rate
VERBOSE = True
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 32
INITIAL_SEEDING = True
# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    p = os.path.dirname(path)
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def filter_taboo_moves(moves, taboo_list):
    return [mv for mv in moves if TabooMove(mv.square, mv.value) not in taboo_list]

# -------------------------
# Initialize board and network
# -------------------------
REGION_H, REGION_W = 3, 3
initial_board = SudokuBoard(REGION_H, REGION_W)
N = initial_board.N
playmode = 'random'

logger.info(f"Board N = {N}")

net = LDNet(lr=LEARNING_RATE, discount_rate=DISCOUNT)
if os.path.exists(MODEL_FILE):
    logger.info("Loading existing LDNet state_dict...")
    net.load_state_dict(torch.load(MODEL_FILE))
else:
    logger.info("Created new LDNet.")
net.train()

# -------------------------
# Create agents
# -------------------------
module = importlib.import_module('team39_A2_LDLearning.sudokuai')
player1 = module.SudokuAI()
player2 = module.SudokuAI()
player1.player_number = 1
player2.player_number = 2
player1.model = net
player2.model = net

# -------------------------
# Replay buffer
# -------------------------
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
ensure_dir(MODEL_FILE)

# -------------------------
# Training loop
# -------------------------
for episode in range(1, TRAINING_EPISODES + 1):
    region_h = random.randint(2, 3)
    region_w = random.randint(2, 3)
    initial_board = SudokuBoard(region_h, region_w)

    N = initial_board.N
    allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode)
    game_state = GameState(
        initial_board=initial_board,
        allowed_squares1=allowed_squares1, occupied_squares1=[],
        allowed_squares2=allowed_squares2, occupied_squares2=[]
    )

    MAX_MOVES_PER_EPISODE = int((N*N) * 1.2)

    if(INITIAL_SEEDING):
        game_state = seed_board_with_N_values(game_state, player1, player2, N)

        logger.info("Initial seeded board:")
        display_board(game_state)

    moves_done = 0
    episode_score_p1 = 0
    episode_score_p2 = 0
    finished_players = set()

    logger.info(f"=== Episode {episode} with board {N} x {N} ===")

    while moves_done < MAX_MOVES_PER_EPISODE:
        player_num = game_state.current_player
        agent = player1 if player_num == 1 else player2
        logger.info(f"Player {player_num} is now playing")
        logger.info(f"Game_state current player {game_state.current_player}")

        # Get legal moves and filter taboo moves
        legal_moves = agent.generate_legal_moves(game_state)
        legal_moves = filter_taboo_moves(legal_moves, getattr(game_state, 'taboo_moves', []))
        legal_moves = [mv for mv in legal_moves if is_really_legal(game_state, mv, player_num)]

        if not legal_moves:
            logger.info(f"Player {player_num} has no legal moves left.")
            finished_players.add(player_num)
            if len(finished_players) == 2:
                logger.info("Both players have no legal moves. Terminating episode.")
                break
            game_state.current_player = 3 - game_state.current_player

        else:
            # Snapshot for TD update
            state_before = copy.deepcopy(game_state)

            # Epsilon-greedy move selection
            if random.random() < EPSILON:
                chosen_move = random.choice(legal_moves)
            else:
                best_val = float('-inf')
                best_move = None
                for mv in legal_moves:
                    next_state_sim = agent._apply_move(game_state, mv)
                    board_t, score_t = net.encode_state(next_state_sim)
                    val = net.predict(board_t, score_t)
                    if val > best_val:
                        best_val = val
                        best_move = mv
                chosen_move = best_move if best_move is not None else random.choice(legal_moves)

            # Apply the move
            next_state = agent._apply_move(game_state, chosen_move)

            # Calculate reward
            points = count_completed_regions(next_state.board, chosen_move.square, region_h, region_w)
            reward = 1.0 + points if points > 0 else -0.2

            # Update occupied squares and scores
            if player_num == 1:
                next_state.occupied_squares1.append(chosen_move.square)
                episode_score_p1 += points
            else:
                next_state.occupied_squares2.append(chosen_move.square)
                episode_score_p2 += points

            # Store in replay buffer
            s_board, s_score = net.encode_state(state_before)
            ns_board, ns_score = net.encode_state(next_state)
            replay_buffer.append((s_board, s_score, reward, ns_board, ns_score))

            game_state = next_state
            moves_done += 1
            finished_players.discard(player_num)

            # Logging
            logger.info(f"Move {moves_done}: Player 1 = {episode_score_p1}, Player 2 = {episode_score_p2}")
            logger.info(f"Player {player_num} played: {chosen_move}, points = {points}")
            if VERBOSE:
                display_board(game_state)

        # Termination check
        if board_is_full(game_state.board) or len(finished_players) == 2:
            logger.info("Episode terminated: board full or both players stuck")
            break


        # # Switch player at end
        # logger.info(f"current player: {game_state.current_player}")
        # game_state.current_player = 3 - game_state.current_player
        # logger.info(f"new current player: {game_state.current_player}")

        # Train on mini-batch from replay buffer
        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            for sb, ss, r, nsb, nss in batch:
                net.doLDLearning(sb, ss, r, nsb, nss)

    # Terminal rewards
    final_reward_p1 = (episode_score_p1 - episode_score_p2) / 20.0
    final_reward_p2 = -final_reward_p1

    for player, final_reward in [(player1, final_reward_p1), (player2, final_reward_p2)]:
        board_t, score_t = net.encode_state(game_state)
        zero_board = torch.zeros_like(board_t)
        zero_score = torch.zeros_like(score_t)
        net.doLDLearning(board_t, score_t, final_reward, zero_board, zero_score)

    # Save model
    torch.save(net.state_dict(), MODEL_FILE)
    logger.info(f"Episode {episode} finished. P1={episode_score_p1}, P2={episode_score_p2}")
    logger.info(f"Saved LDNet state_dict to {MODEL_FILE}")

logger.info("Training finished.")
