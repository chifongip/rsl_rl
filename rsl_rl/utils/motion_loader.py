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

"""Motion loader for Adversarial Motion Priors (AMP).

Loads motion-capture .npz files and provides expert demonstration batches
for discriminator training.  Math utilities are inlined to avoid external
dependencies.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Inline math utilities (replaces mjlab.utils.lab_api.math)
# ---------------------------------------------------------------------------

def _quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (w, x, y, z) to 3x3 rotation matrices.

    Args:
        q: (..., 4) tensor of quaternions.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors *v* by the inverse of quaternion *q*.

    Equivalent to ``R(q)^T @ v``.

    Args:
        q: (..., 4) unit quaternions (w, x, y, z).
        v: (..., 3) vectors.

    Returns:
        (..., 3) rotated vectors.
    """
    R = _quat_to_rotation_matrix(q)
    # R^T @ v  (batched matmul over last two dims)
    return (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)


def _subtract_frame_transforms(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the relative position and orientation of *body* w.r.t. *anchor*.

    Args:
        anchor_pos:  (..., 3) anchor world positions.
        anchor_quat: (..., 4) anchor world quaternions (w, x, y, z).
        body_pos:    (..., 3) body world positions.
        body_quat:   (..., 4) body world quaternions (w, x, y, z).

    Returns:
        rel_pos:  (..., 3) body position in anchor frame.
        rel_quat: (..., 4) body orientation in anchor frame.
    """
    # Relative position: R_anchor^T @ (body_pos - anchor_pos)
    rel_pos = _quat_apply_inverse(anchor_quat, body_pos - anchor_pos)

    # Relative orientation: q_rel = q_anchor^{-1} * q_body
    # For unit quaternions (w,x,y,z): inverse = (w, -x, -y, -z)
    q_anchor_inv = torch.cat(
        [anchor_quat[..., :1], -anchor_quat[..., 1:]], dim=-1
    )
    # Hamilton product q_anchor_inv * q_body
    aw, ax, ay, az = q_anchor_inv.unbind(-1)
    bw, bx, by, bz = body_quat.unbind(-1)
    rel_quat = torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )
    return rel_pos, rel_quat


# ---------------------------------------------------------------------------
# AMPLoader
# ---------------------------------------------------------------------------

class AMPLoader:
    """Load and serve AMP motion-capture demonstration data.

    Args:
        motion_file: Path to a single .npz file or a directory of .npz files.
        body_names: Names of the target bodies to track.
        anchor_name: Name of the anchor (root) body.
        all_body_names: Ordered list of *all* body names in the model.
            The index of each name in this list must match the body
            dimension in the .npz arrays.
        device: Torch device.
    """

    def __init__(
        self,
        motion_file: str,
        body_names: Sequence[str],
        anchor_name: str,
        all_body_names: Sequence[str],
        device: str = "cuda:0",
    ):
        assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

        # resolve name -> index
        all_names_list = list(all_body_names)
        self._body_indexes = [all_names_list.index(n) for n in body_names]
        self._anchor_indexes = all_names_list.index(anchor_name)
        self._num_bodies = len(self._body_indexes)

        # check if file or directory
        if os.path.isfile(motion_file):
            motion_files = [motion_file]
            motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
        elif os.path.isdir(motion_file):
            motion_names = []
            motion_files = []
            for root, _dirs, files in os.walk(motion_file):
                for filename in sorted(files):
                    if filename.endswith(".npz"):
                        motion_names.append(os.path.splitext(filename)[0])
                        motion_files.append(os.path.join(root, filename))
            motion_files, motion_names = (
                zip(*sorted(zip(motion_files, motion_names)))
                if motion_files
                else ([], [])
            )
            motion_files, motion_names = list(motion_files), list(motion_names)
            assert len(motion_files) > 0, f"No npz files found in directory: {motion_file}"
        else:
            raise ValueError(f"Path is neither a file nor a directory: {motion_file}")

        # store all motion data lists
        self.motion_names = motion_names
        self._body_pos_b_list: list[torch.Tensor] = []
        self._body_ori_b_list: list[torch.Tensor] = []
        self._body_lin_vel_b_list: list[torch.Tensor] = []
        self._body_ang_vel_b_list: list[torch.Tensor] = []

        # process each motion file
        for motion_idx, (motion_name, motion_path) in enumerate(
            zip(motion_names, motion_files)
        ):
            print(f"Processing motion {motion_idx + 1}/{len(motion_files)}: {motion_name}")
            data = np.load(motion_path)

            if motion_idx == 0:
                self.fps = data["fps"]

            _dof_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            _dof_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            _body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            _body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            _body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            _body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            time_step_total = _dof_pos.shape[0]
            if time_step_total < 2:
                raise ValueError(
                    f"Motion '{motion_name}' must contain at least two frames, got {time_step_total}."
                )

            # allocate storage for this motion
            _body_pos_b = torch.zeros(
                (time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device
            )
            _body_ori_b = torch.zeros(
                (time_step_total, self._num_bodies, 6), dtype=torch.float32, device=device
            )
            _body_lin_vel_b = torch.zeros(
                (time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device
            )
            _body_ang_vel_b = torch.zeros(
                (time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device
            )

            # process all frames
            for frame_idx in tqdm(
                range(time_step_total),
                desc=f"Preloading AMP data for {motion_name}",
            ):
                # anchor and body data for this frame
                tgt_anchor_pos_w = (
                    _body_pos_w[frame_idx, self._anchor_indexes, :]
                    .squeeze()
                    .unsqueeze(0)
                    .repeat(self._num_bodies, 1)
                )
                tgt_anchor_quat_w = (
                    _body_quat_w[frame_idx, self._anchor_indexes, :]
                    .squeeze()
                    .unsqueeze(0)
                    .repeat(self._num_bodies, 1)
                )
                tgt_body_pos_w = _body_pos_w[frame_idx, self._body_indexes, :]
                tgt_body_quat_w = _body_quat_w[frame_idx, self._body_indexes, :]
                tgt_body_lin_vel_w = _body_lin_vel_w[frame_idx, self._body_indexes, :]
                tgt_body_ang_vel_w = _body_ang_vel_w[frame_idx, self._body_indexes, :]

                # body position/orientation relative to anchor (local frame)
                tgt_robot_body_pos_b, tgt_robot_body_quat_b = _subtract_frame_transforms(
                    tgt_anchor_pos_w,
                    tgt_anchor_quat_w,
                    tgt_body_pos_w,
                    tgt_body_quat_w,
                )

                # convert orientation quaternion to first two columns of rotation matrix
                mat = _quat_to_rotation_matrix(tgt_robot_body_quat_b)
                tgt_robot_body_ori_b = mat[..., :, :2].reshape(self._num_bodies, 6)

                # velocities in each body's local frame
                tgt_body_lin_vel_b = _quat_apply_inverse(
                    tgt_body_quat_w, tgt_body_lin_vel_w
                )
                tgt_body_ang_vel_b = _quat_apply_inverse(
                    tgt_body_quat_w, tgt_body_ang_vel_w
                )

                # store
                _body_pos_b[frame_idx] = tgt_robot_body_pos_b
                _body_ori_b[frame_idx] = tgt_robot_body_ori_b
                _body_lin_vel_b[frame_idx] = tgt_body_lin_vel_b
                _body_ang_vel_b[frame_idx] = tgt_body_ang_vel_b

            # append to lists
            self._body_pos_b_list.append(_body_pos_b)
            self._body_ori_b_list.append(_body_ori_b)
            self._body_lin_vel_b_list.append(_body_lin_vel_b)
            self._body_ang_vel_b_list.append(_body_ang_vel_b)

        # backward-compatible default (first motion)
        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / self.fps
        self._body_pos_b = self._body_pos_b_list[0]
        self._body_ori_b = self._body_ori_b_list[0]
        self._body_lin_vel_b = self._body_lin_vel_b_list[0]
        self._body_ang_vel_b = self._body_ang_vel_b_list[0]

    @property
    def observation_dim(self) -> int:
        num_bodies = len(self._body_indexes)
        # pos(3) + ori(6) + lin_vel(3) + ang_vel(3) = 15 per body
        return (3 + 6 + 3 + 3) * num_bodies

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Yield (state, next_state) pairs sampled from motion data."""
        num_motions = len(self._body_pos_b_list)

        for batch_idx in range(num_mini_batch):
            # cycle through motions round-robin
            motion_idx = batch_idx % num_motions

            current_body_pos_b = self._body_pos_b_list[motion_idx]
            current_body_ori_b = self._body_ori_b_list[motion_idx]
            current_body_lin_vel_b = self._body_lin_vel_b_list[motion_idx]
            current_body_ang_vel_b = self._body_ang_vel_b_list[motion_idx]
            current_time_step_total = current_body_pos_b.shape[0]

            # random frame indices
            idxs = torch.randint(
                0,
                current_time_step_total - 1,
                (mini_batch_size,),
                device=current_body_pos_b.device,
            )

            batch_body_pos_b = current_body_pos_b[idxs]
            batch_body_ori_b = current_body_ori_b[idxs]
            batch_body_lin_vel_b = current_body_lin_vel_b[idxs]
            batch_body_ang_vel_b = current_body_ang_vel_b[idxs]
            s = torch.cat(
                [
                    batch_body_pos_b.reshape(mini_batch_size, -1),
                    batch_body_ori_b.reshape(mini_batch_size, -1),
                    batch_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )

            next_idxs = idxs + 1
            batch_next_body_pos_b = current_body_pos_b[next_idxs]
            batch_next_body_ori_b = current_body_ori_b[next_idxs]
            batch_next_body_lin_vel_b = current_body_lin_vel_b[next_idxs]
            batch_next_body_ang_vel_b = current_body_ang_vel_b[next_idxs]
            s_next = torch.cat(
                [
                    batch_next_body_pos_b.reshape(mini_batch_size, -1),
                    batch_next_body_ori_b.reshape(mini_batch_size, -1),
                    batch_next_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_next_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )
            yield s, s_next
