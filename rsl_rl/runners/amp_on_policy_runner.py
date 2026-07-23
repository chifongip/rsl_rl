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

"""On-policy runner specialization for Adversarial Motion Priors."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import AMPPPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class AMPOnPolicyRunner(OnPolicyRunner):
    """Use the reliable on-policy runner while logging AMP-shaped rewards."""

    alg: AMPPPO

    def _process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> torch.Tensor:
        """Process AMP rewards and return the shaped rewards for logging."""
        self.alg.process_env_step(obs, rewards, dones, extras)
        if self.alg.last_amp_rewards is None:
            raise RuntimeError("AMPPPO did not produce rewards for the processed step.")
        return self.alg.last_amp_rewards

    def load(
        self,
        path: str,
        load_cfg: dict | bool | None = None,
        strict: bool = True,
        map_location: str | None = None,
        *,
        load_optimizer: bool | None = None,
    ) -> dict:
        """Load a checkpoint, accepting the legacy ``load_optimizer`` option."""
        if isinstance(load_cfg, bool):
            load_optimizer = load_cfg
            load_cfg = None
        if load_optimizer is not None:
            load_cfg = dict(load_cfg or {})
            load_cfg.setdefault("actor", True)
            load_cfg.setdefault("critic", True)
            load_cfg.setdefault("iteration", True)
            load_cfg.setdefault("rnd", True)
            load_cfg["optimizer"] = load_optimizer
        return super().load(path, load_cfg, strict, map_location)
