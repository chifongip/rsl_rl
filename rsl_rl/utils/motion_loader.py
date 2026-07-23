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

import numpy as np
import os
import torch
from collections.abc import Generator, Sequence

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
    rotation = _quat_to_rotation_matrix(q)
    # R^T @ v  (batched matmul over last two dims)
    return (rotation.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)


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
    q_anchor_inv = torch.cat([anchor_quat[..., :1], -anchor_quat[..., 1:]], dim=-1)
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


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
    """Interpolate unit quaternions along the shortest path."""
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    close = dot > 0.9995
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(1e-8)
    slerp = torch.sin((1.0 - fraction) * theta) / sin_theta * q0 + torch.sin(fraction * theta) / sin_theta * q1
    lerp = q0 + fraction * (q1 - q0)
    return torch.nn.functional.normalize(torch.where(close, lerp, slerp), dim=-1)


def _resample_motion(
    values: torch.Tensor,
    source_fps: float,
    step_dt: float,
    *,
    quaternion: bool = False,
) -> torch.Tensor:
    """Resample a time-major motion tensor to the environment timestep."""
    duration = (values.shape[0] - 1) / source_fps
    num_target_frames = int(duration / step_dt + 1e-9) + 1
    if num_target_frames < 2:
        raise ValueError(f"Motion duration {duration:.6f}s is shorter than step_dt={step_dt:.6f}s.")
    target_times = torch.arange(num_target_frames, device=values.device) * step_dt
    source_positions = (target_times * source_fps).clamp(max=values.shape[0] - 1)
    lower = source_positions.floor().long()
    upper = (lower + 1).clamp(max=values.shape[0] - 1)
    fraction = (source_positions - lower).reshape(num_target_frames, *([1] * (values.ndim - 1)))
    if quaternion:
        return _quat_slerp(values[lower], values[upper], fraction)
    return torch.lerp(values[lower], values[upper], fraction)


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
        step_dt: float | None = None,
    ) -> None:
        """Load, validate, resample, and transform motion clips."""
        if not os.path.exists(motion_file):
            raise ValueError(f"Invalid motion path: {motion_file}")
        if step_dt is not None and (not np.isfinite(step_dt) or step_dt <= 0):
            raise ValueError(f"step_dt must be positive and finite, got {step_dt}.")
        if not body_names:
            raise ValueError("body_names must not be empty.")

        # resolve name -> index
        all_names_list = list(all_body_names)
        missing_names = [name for name in [*body_names, anchor_name] if name not in all_names_list]
        if missing_names:
            raise ValueError(f"Unknown AMP body names: {missing_names}.")
        self._body_indexes = [all_names_list.index(n) for n in body_names]
        self._anchor_indexes = all_names_list.index(anchor_name)
        self._num_bodies = len(self._body_indexes)
        self.step_dt = step_dt

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
            motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names))) if motion_files else ([], [])
            motion_files, motion_names = list(motion_files), list(motion_names)
            if not motion_files:
                raise ValueError(f"No npz files found in directory: {motion_file}")
        else:
            raise ValueError(f"Path is neither a file nor a directory: {motion_file}")

        # store all motion data lists
        self.motion_names = motion_names
        self._body_pos_b_list: list[torch.Tensor] = []
        self._body_ori_b_list: list[torch.Tensor] = []
        self._body_lin_vel_b_list: list[torch.Tensor] = []
        self._body_ang_vel_b_list: list[torch.Tensor] = []

        self.source_fps: list[float] = []

        # process each motion file
        for motion_idx, (motion_name, motion_path) in enumerate(zip(motion_names, motion_files)):
            print(f"Processing motion {motion_idx + 1}/{len(motion_files)}: {motion_name}")
            with np.load(motion_path) as data:
                required = {
                    "fps",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                }
                missing = sorted(required.difference(data.files))
                if missing:
                    raise ValueError(f"Motion '{motion_name}' is missing fields: {missing}.")
                source_fps = float(np.asarray(data["fps"]).item())
                arrays = {key: np.asarray(data[key]) for key in required if key != "fps"}
            if not np.isfinite(source_fps) or source_fps <= 0:
                raise ValueError(f"Motion '{motion_name}' has invalid fps={source_fps}.")
            shapes = {key: value.shape for key, value in arrays.items()}
            if arrays["body_pos_w"].ndim != 3:
                raise ValueError(
                    f"Motion '{motion_name}' field 'body_pos_w' has invalid shape "
                    f"{arrays['body_pos_w'].shape}; expected a 3D tensor."
                )
            expected_frames = arrays["body_pos_w"].shape[0]
            expected_body_shape = (expected_frames, len(all_names_list))
            for key, value in arrays.items():
                expected_width = 4 if key == "body_quat_w" else 3
                if value.ndim != 3 or value.shape[:2] != expected_body_shape or value.shape[2] != expected_width:
                    raise ValueError(
                        f"Motion '{motion_name}' field '{key}' has invalid shape {shapes[key]}; "
                        f"expected ({expected_frames}, {len(all_names_list)}, {expected_width})."
                    )
                if not np.isfinite(value).all():
                    raise ValueError(f"Motion '{motion_name}' field '{key}' contains non-finite values.")

            body_pos_world = torch.as_tensor(arrays["body_pos_w"], dtype=torch.float32, device=device)
            body_quat_world = torch.as_tensor(arrays["body_quat_w"], dtype=torch.float32, device=device)
            body_lin_vel_world = torch.as_tensor(arrays["body_lin_vel_w"], dtype=torch.float32, device=device)
            body_ang_vel_world = torch.as_tensor(arrays["body_ang_vel_w"], dtype=torch.float32, device=device)
            quat_norm = body_quat_world.norm(dim=-1, keepdim=True)
            if (quat_norm < 1e-8).any():
                raise ValueError(f"Motion '{motion_name}' contains a zero-norm quaternion.")
            body_quat_world = body_quat_world / quat_norm

            time_step_total = expected_frames
            if time_step_total < 2:
                raise ValueError(f"Motion '{motion_name}' must contain at least two frames, got {time_step_total}.")
            if step_dt is not None:
                body_pos_world = _resample_motion(body_pos_world, source_fps, step_dt)
                body_quat_world = _resample_motion(body_quat_world, source_fps, step_dt, quaternion=True)
                body_lin_vel_world = _resample_motion(body_lin_vel_world, source_fps, step_dt)
                body_ang_vel_world = _resample_motion(body_ang_vel_world, source_fps, step_dt)

            time_step_total = body_pos_world.shape[0]
            anchor_pos_w = body_pos_world[:, self._anchor_indexes : self._anchor_indexes + 1].expand(
                -1, self._num_bodies, -1
            )
            anchor_quat_w = body_quat_world[:, self._anchor_indexes : self._anchor_indexes + 1].expand(
                -1, self._num_bodies, -1
            )
            body_pos_w = body_pos_world[:, self._body_indexes]
            body_quat_w = body_quat_world[:, self._body_indexes]
            body_lin_vel_w = body_lin_vel_world[:, self._body_indexes]
            body_ang_vel_w = body_ang_vel_world[:, self._body_indexes]

            body_pos_local, body_quat_b = _subtract_frame_transforms(
                anchor_pos_w,
                anchor_quat_w,
                body_pos_w,
                body_quat_w,
            )
            mat = _quat_to_rotation_matrix(body_quat_b)
            body_ori_local = mat[..., :, :2].reshape(time_step_total, self._num_bodies, 6)
            body_lin_vel_local = _quat_apply_inverse(body_quat_w, body_lin_vel_w)
            body_ang_vel_local = _quat_apply_inverse(body_quat_w, body_ang_vel_w)

            # append to lists
            self._body_pos_b_list.append(body_pos_local)
            self._body_ori_b_list.append(body_ori_local)
            self._body_lin_vel_b_list.append(body_lin_vel_local)
            self._body_ang_vel_b_list.append(body_ang_vel_local)
            self.source_fps.append(source_fps)

        # backward-compatible default (first motion)
        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.fps = 1.0 / step_dt if step_dt is not None else self.source_fps[0]
        self.motion_total_time = (self.time_step_total - 1) / self.fps
        self._body_pos_b = self._body_pos_b_list[0]
        self._body_ori_b = self._body_ori_b_list[0]
        self._body_lin_vel_b = self._body_lin_vel_b_list[0]
        self._body_ang_vel_b = self._body_ang_vel_b_list[0]

    @property
    def observation_dim(self) -> int:
        """Return flattened AMP feature dimension."""
        num_bodies = len(self._body_indexes)
        # pos(3) + ori(6) + lin_vel(3) + ang_vel(3) = 15 per body
        return (3 + 6 + 3 + 3) * num_bodies

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
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
