# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import torch


class ReplayBuffer:
    """Fixed-size circular buffer to store (state, next_state) pairs."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str = "cpu"):
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Add new states to memory."""
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step : self.buffer_size] = states[
                : self.buffer_size - self.step
            ]
            self.next_states[self.step : self.buffer_size] = next_states[
                : self.buffer_size - self.step
            ]
            self.states[: end_idx - self.buffer_size] = states[
                self.buffer_size - self.step :
            ]
            self.next_states[: end_idx - self.buffer_size] = next_states[
                self.buffer_size - self.step :
            ]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield random (state, next_state) batches."""
        for _ in range(num_mini_batch):
            sample_idxs = torch.randint(0, self.num_samples, (mini_batch_size,), device=self.device)
            yield (
                self.states[sample_idxs],
                self.next_states[sample_idxs],
            )
