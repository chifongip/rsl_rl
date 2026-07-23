# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMP discriminator behavior."""

import torch

import pytest

from rsl_rl.modules import Discriminator


def test_reward_prediction_preserves_module_mode() -> None:
    """Reward inference should restore both training and evaluation modes."""
    discriminator = Discriminator(8, 0.1, [16], "cpu")
    inputs = torch.randn(3, 4)
    task_rewards = torch.randn(3)

    discriminator.eval()
    discriminator.predict_amp_reward(inputs, inputs, task_rewards)
    assert not discriminator.training

    discriminator.train()
    discriminator.predict_amp_reward(inputs, inputs, task_rewards)
    assert discriminator.training


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_layer_sizes": []},
        {"hidden_layer_sizes": [0]},
        {"amp_reward_coef": -1.0},
        {"task_reward_lerp": 1.1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict) -> None:
    """Invalid discriminator dimensions and reward settings should fail early."""
    defaults = {
        "input_dim": 8,
        "amp_reward_coef": 0.1,
        "hidden_layer_sizes": [16],
        "device": "cpu",
        "task_reward_lerp": 0.0,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        Discriminator(**defaults)
