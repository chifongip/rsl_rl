# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for AMP runner checkpoint loading."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from tensordict import TensorDict
from typing import ClassVar
from unittest.mock import Mock

from rsl_rl.env import VecEnv
from rsl_rl.runners import AMPOnPolicyRunner, OnPolicyRunner


class _Robot:
    body_names: ClassVar[list[str]] = ["body"]


class _AMPEnv(VecEnv):
    """Minimal environment satisfying the AMP construction contract."""

    num_envs = 4
    num_actions = 2
    max_episode_length = 100
    device = "cpu"
    step_dt = 0.02
    cfg: ClassVar[dict] = {}
    scene: ClassVar[dict] = {"robot": _Robot()}

    def __init__(self) -> None:
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    @property
    def unwrapped(self) -> _AMPEnv:
        return self

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, 8),
                "amp": torch.randn(self.num_envs, 15),
            },
            batch_size=[self.num_envs],
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        del actions
        return (
            self.get_observations(),
            torch.randn(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"time_outs": torch.zeros(self.num_envs)},
        )


def _write_motion(path: Path) -> None:
    num_frames = 8
    np.savez(
        path,
        fps=np.array(50.0),
        body_pos_w=np.zeros((num_frames, 1, 3), dtype=np.float32),
        body_quat_w=np.tile([1, 0, 0, 0], (num_frames, 1, 1)).astype(np.float32),
        body_lin_vel_w=np.zeros((num_frames, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((num_frames, 1, 3), dtype=np.float32),
    )


def _train_cfg(motion_path: Path) -> dict:
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [16]},
        "algorithm": {
            "class_name": "AMPPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "schedule": "fixed",
            "rnd_cfg": None,
            "amp_motion_files": str(motion_path),
            "amp_body_names": ["body"],
            "amp_anchor_name": "body",
            "amp_reward_coef": 0.1,
            "amp_discr_hidden_dims": [16],
            "amp_replay_buffer_size": 128,
        },
    }


def test_load_optimizer_choice_is_forwarded_to_algorithm(tmp_path: Path) -> None:
    """The runner should pass both optimizer-loading choices to the algorithm."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"iter": 7, "infos": {"source": "test"}}, checkpoint_path)

    runner = AMPOnPolicyRunner.__new__(AMPOnPolicyRunner)
    runner.alg = Mock()
    runner.alg.load.return_value = True
    runner.current_learning_iteration = 0

    for load_optimizer in (False, True):
        infos = runner.load(str(checkpoint_path), load_optimizer=load_optimizer)

        load_cfg = runner.alg.load.call_args.args[1]
        assert load_cfg["optimizer"] is load_optimizer
        assert runner.current_learning_iteration == 7
        assert infos == {"source": "test"}


def test_amp_runner_inherits_standard_runner_api() -> None:
    """AMP runner should retain standard inference and export behavior."""
    assert issubclass(AMPOnPolicyRunner, OnPolicyRunner)
    for method in ("get_inference_policy", "export_policy_to_jit", "export_policy_to_onnx"):
        assert getattr(AMPOnPolicyRunner, method) is getattr(OnPolicyRunner, method)


def test_inherited_runner_completes_amp_learning_iteration(tmp_path: Path) -> None:
    """The standard runner loop should train AMPPPO without specialization drift."""
    motion_path = tmp_path / "motion.npz"
    _write_motion(motion_path)
    runner = AMPOnPolicyRunner(_AMPEnv(), _train_cfg(motion_path))

    runner.learn(1)

    assert runner.current_learning_iteration == 0
    assert runner.alg.amp_storage.num_samples == 16
