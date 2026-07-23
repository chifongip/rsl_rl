# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMP replay buffer boundary handling."""

import torch

import pytest

from rsl_rl.storage import ReplayBuffer


def test_oversized_insert_keeps_newest_samples() -> None:
    """An arbitrarily large insertion should retain the newest capacity samples."""
    buffer = ReplayBuffer(obs_dim=2, buffer_size=3)
    states = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    buffer.insert(states, states + 100)

    assert torch.equal(buffer.states, states[-3:])
    assert torch.equal(buffer.next_states, states[-3:] + 100)
    assert buffer.num_samples == 3


def test_wraparound_insert_preserves_all_recent_samples() -> None:
    """Multiple smaller insertions should wrap without corrupting samples."""
    buffer = ReplayBuffer(obs_dim=1, buffer_size=4)
    buffer.insert(torch.tensor([[0.0], [1.0], [2.0]]), torch.tensor([[1.0], [2.0], [3.0]]))
    buffer.insert(torch.tensor([[3.0], [4.0], [5.0]]), torch.tensor([[4.0], [5.0], [6.0]]))

    assert set(buffer.states.flatten().tolist()) == {2.0, 3.0, 4.0, 5.0}
    assert buffer.num_samples == 4


def test_empty_buffer_cannot_be_sampled() -> None:
    """Sampling before insertion should fail with a clear error."""
    buffer = ReplayBuffer(obs_dim=2, buffer_size=3)
    with pytest.raises(RuntimeError, match="empty replay buffer"):
        next(buffer.feed_forward_generator(1, 2))
