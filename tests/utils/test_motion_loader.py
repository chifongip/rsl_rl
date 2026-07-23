# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for AMP motion transition sampling."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

import pytest

from rsl_rl.utils import AMPLoader


def _write_motion(path: Path, num_frames: int, fps: float = 30.0) -> None:
    body_velocities = np.arange(num_frames, dtype=np.float32).reshape(num_frames, 1, 1)
    body_velocities = np.repeat(body_velocities, 3, axis=-1)
    np.savez(
        path,
        fps=np.array(fps),
        joint_pos=np.zeros((num_frames, 1), dtype=np.float32),
        joint_vel=np.zeros((num_frames, 1), dtype=np.float32),
        body_pos_w=np.zeros((num_frames, 1, 3), dtype=np.float32),
        body_quat_w=np.tile([1, 0, 0, 0], (num_frames, 1, 1)).astype(np.float32),
        body_lin_vel_w=body_velocities,
        body_ang_vel_w=np.zeros((num_frames, 1, 3), dtype=np.float32),
    )


def test_samples_only_consecutive_motion_frames(tmp_path: Path) -> None:
    """The loader should never manufacture a final-frame self-transition."""
    motion_path = tmp_path / "motion.npz"
    _write_motion(motion_path, num_frames=3)
    loader = AMPLoader(
        str(motion_path),
        body_names=["body"],
        anchor_name="body",
        all_body_names=["body"],
        device="cpu",
    )

    state, next_state = next(loader.feed_forward_generator(1, 128))
    assert torch.all(next_state[:, 9:12] - state[:, 9:12] == 1)


def test_rejects_motion_with_fewer_than_two_frames(tmp_path: Path) -> None:
    """A motion must contain a current and next frame."""
    motion_path = tmp_path / "motion.npz"
    _write_motion(motion_path, num_frames=1)
    with pytest.raises(ValueError, match="at least two frames"):
        AMPLoader(
            str(motion_path),
            body_names=["body"],
            anchor_name="body",
            all_body_names=["body"],
            device="cpu",
        )


def test_resamples_motion_to_environment_timestep(tmp_path: Path) -> None:
    """Expert transitions should span exactly one environment control step."""
    motion_path = tmp_path / "motion.npz"
    _write_motion(motion_path, num_frames=4, fps=10.0)
    loader = AMPLoader(
        str(motion_path),
        body_names=["body"],
        anchor_name="body",
        all_body_names=["body"],
        device="cpu",
        step_dt=0.05,
    )

    state, next_state = next(loader.feed_forward_generator(1, 128))
    assert torch.allclose(
        next_state[:, 9:12] - state[:, 9:12],
        torch.full((128, 3), 0.5),
    )


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan")])
def test_rejects_invalid_fps(tmp_path: Path, fps: float) -> None:
    """Motion FPS must be positive and finite."""
    motion_path = tmp_path / "motion.npz"
    _write_motion(motion_path, num_frames=4, fps=fps)
    with pytest.raises(ValueError, match="invalid fps"):
        AMPLoader(
            str(motion_path),
            body_names=["body"],
            anchor_name="body",
            all_body_names=["body"],
            device="cpu",
            step_dt=0.02,
        )
