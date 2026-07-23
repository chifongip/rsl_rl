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
from collections.abc import Generator


class ReplayBuffer:
    """Fixed-size circular buffer to store (state, next_state) pairs."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str = "cpu") -> None:
        """Initialize fixed-capacity state and next-state storage."""
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}.")
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}.")
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """Add new states to memory."""
        if states.ndim != 2 or states.shape[1] != self.states.shape[1]:
            raise ValueError(f"states must have shape (N, {self.states.shape[1]}), got {tuple(states.shape)}.")
        if next_states.shape != states.shape:
            raise ValueError(
                f"next_states must match states shape {tuple(states.shape)}, got {tuple(next_states.shape)}."
            )
        num_states = states.shape[0]
        if num_states == 0:
            return
        if num_states >= self.buffer_size:
            self.states.copy_(states[-self.buffer_size :])
            self.next_states.copy_(next_states[-self.buffer_size :])
            self.step = 0
            self.num_samples = self.buffer_size
            return

        indices = (torch.arange(num_states, device=self.device) + self.step) % self.buffer_size
        self.states[indices] = states
        self.next_states[indices] = next_states
        self.step = (self.step + num_states) % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + num_states)

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yield random (state, next_state) batches."""
        if self.num_samples == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        for _ in range(num_mini_batch):
            sample_idxs = torch.randint(0, self.num_samples, (mini_batch_size,), device=self.device)
            yield (
                self.states[sample_idxs],
                self.next_states[sample_idxs],
            )
