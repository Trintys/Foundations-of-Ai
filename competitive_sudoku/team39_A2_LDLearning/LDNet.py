# LDNet.py  --- CNN-based value network (size-agnostic)

import torch
import torch.nn as nn
from competitive_sudoku.sudoku import GameState
from typing import Optional


class LDNet(nn.Module):
    """
    CNN-based TD(0) value network for Competitive Sudoku.
    Works with ANY N×N board size (2x2 regions, 3x3 regions, etc.)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        discount_rate: float = 0.9
    ):
        super().__init__()
        self.discount_rate = discount_rate

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Global average pooling → vector of size 64
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final value head: 64 features + 1 score_diff feature → scalar
        self.value_head = nn.Sequential(
            nn.Linear(65, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # optimizer and loss
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ----------------------------------------------------
    # FORWARD
    # ----------------------------------------------------
    def forward(self, board_tensor, score_diff_tensor):
        """
        board_tensor: (1, 3, N, N)
        score_diff_tensor: (1, 1)
        returns: (1, 1)
        """
        x = self.cnn(board_tensor)         # (1, 64, N, N)
        x = self.global_pool(x)            # (1, 64, 1, 1)
        x = x.view(1, 64)                  # (1, 64)

        # concat score difference
        x = torch.cat([x, score_diff_tensor], dim=1)  # (1, 65)

        return self.value_head(x)          # (1, 1)

    # ----------------------------------------------------
    # PREDICT (no gradient)
    # ----------------------------------------------------
    def predict(self, board_tensor, score_diff_tensor):
        self.eval()
        with torch.no_grad():
            out = self.forward(board_tensor, score_diff_tensor)
            return float(out.item())

    # ----------------------------------------------------
    # TD(0) UPDATE
    # ----------------------------------------------------
    def doLDLearning(self, s_board, s_score, reward, ns_board, ns_score):
        """
        TD learning:
            target = r + γ * V(s')
        """
        self.train()

        # Move tensors to device
        s_board = s_board
        s_score = s_score
        ns_board = ns_board
        ns_score = ns_score

        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)

        # forward passes
        value = self.forward(s_board, s_score)
        with torch.no_grad():
            next_value = self.forward(ns_board, ns_score)

        # TD target
        target = reward_tensor + self.discount_rate * next_value

        loss = self.loss_fn(value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ----------------------------------------------------
    # ENCODE GAMESTATE → CNN INPUT
    # ----------------------------------------------------
    def encode_state(self, state: GameState):
        """
        Returns: 
        - board_tensor: (1, 3, N, N)
        - score_diff_tensor: (1, 1)
        """

        board = state.board
        N = board.N

        # Channels
        ch_value = torch.zeros((N, N), dtype=torch.float32)
        ch_p1 = torch.zeros((N, N), dtype=torch.float32)
        ch_p2 = torch.zeros((N, N), dtype=torch.float32)

        for r in range(N):
            for c in range(N):
                v = board.get((r, c))

                # normalize values
                if v != board.empty:
                    ch_value[r, c] = v / float(N)
                else:
                    ch_value[r, c] = 0.0

        # Fill player occupation maps
        for (r, c) in state.occupied_squares1:
            ch_p1[r, c] = 1.0
        for (r, c) in state.occupied_squares2:
            ch_p2[r, c] = 1.0

        # Stack channels: (3, N, N)
        board_tensor = torch.stack([ch_value, ch_p1, ch_p2], dim=0).unsqueeze(0)

        # Score difference
        score_diff = (state.scores[0] - state.scores[1]) / 50.0
        score_diff_tensor = torch.tensor([[score_diff]], dtype=torch.float32)

        return board_tensor, score_diff_tensor
