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

"""On-policy runner for AMP (Adversarial Motion Priors) training.

Extends the standard :class:`OnPolicyRunner` with:
- AMP discriminator reward shaping during rollouts.
- Discriminator + AMP normalizer in save/load checkpoints.
- ``AMPOnPolicyRunner.construct_algorithm`` that builds :class:`AMPPPO`.
"""

from __future__ import annotations

import os
import time

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import AMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan
from rsl_rl.utils.logger import Logger


class AMPOnPolicyRunner:
    """On-policy runner for training with Adversarial Motion Priors.

    Constructs an :class:`AMPPPO` algorithm and runs the standard PPO
    training loop, replacing environment rewards with AMP discriminator
    rewards at each step.

    The environment must return observations as a :class:`TensorDict` with
    at least ``"actor"``, ``"critic"``, and ``"amp"`` keys.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ):
        self.cfg = train_cfg
        self.device = device
        self.env = env

        # Multi-GPU
        self._configure_multi_gpu()

        # Observations — must be a TensorDict with "actor", "critic", "amp"
        obs_td = self.env.get_observations()
        if isinstance(obs_td, tuple):
            obs_td = obs_td[0]

        # Build AMPPPO via its factory
        self.alg: AMPPPO = AMPPPO.construct_algorithm(
            obs_td, self.env, self.cfg, self.device
        )

        # Logger (matches v6 OnPolicyRunner pattern)
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        # Training state
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.current_learning_iteration = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Run the AMP training loop."""
        # Initialize logger writer
        self.logger.init_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observations (full TensorDict)
        obs_td = self.env.get_observations()
        if isinstance(obs_td, tuple):
            obs_td = obs_td[0]
        obs_td = obs_td.to(self.device)
        amp_obs = obs_td["amp"]

        # Sync parameters across GPUs
        if self.is_distributed:
            self.alg.broadcast_parameters()

        self.alg.train_mode()  # switch to train mode (for dropout for example)
        self.alg.discriminator.train()  # train_mode() only covers actor+critic

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # ---- Rollout ----
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_td)
                    obs_td, rewards, dones, infos = self.env.step(
                        actions.to(self.env.device)
                    )
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs_td, rewards, dones)
                    obs_td = obs_td.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = obs_td["amp"]

                    # Terminal state handling: mjlab auto-resets before
                    # computing obs, so post-reset amp obs belong to the new
                    # episode. Use pre-step amp_obs as the best approximation
                    # of the terminal amp observation.
                    next_amp_obs_with_term = next_amp_obs.clone()
                    reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids] = amp_obs[reset_env_ids]

                    # Replace reward with AMP discriminator reward
                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        rewards,
                        normalizer=self.alg.amp_normalizer,
                    )[0]

                    amp_obs = next_amp_obs.clone()

                    # Inject terminal-adjusted amp obs so the replay buffer
                    # stores valid transitions (not cross-episode).
                    infos.setdefault("observations", {})
                    infos["observations"]["amp"] = next_amp_obs_with_term

                    self.alg.process_env_step(obs_td, rewards, dones, infos)
                    # Book keeping
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    self.logger.process_env_step(rewards, dones, infos, intrinsic_rewards)

                # Compute returns
                self.alg.compute_returns(obs_td)

            collection_time = time.time() - start
            start = time.time()

            # ---- Update ----
            loss_dict = self.alg.update()
            learn_time = time.time() - start

            self.current_learning_iteration = it

            # ---- Log ----
            self.logger.log(
                it=it,
                start_it=start_iter,
                total_it=tot_iter,
                collect_time=collection_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.actor.output_std,
                rnd_weight=self.alg.rnd.weight if self.alg.rnd else None,
            )

            # ---- Save ----
            if self.logger.writer is not None and it % self.save_interval == 0:
                self.save(
                    os.path.join(self.logger.log_dir, f"model_{it}.pt"), infos={"iter": it}
                )

        # Final save
        if self.logger.writer is not None:
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"),
                infos={"iter": self.current_learning_iteration},
            )
            self.logger.stop_logging_writer()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None):
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos or {}
        # Save obs normalizer state (inside MLPModel state_dict, but also
        # store separately for cross-format compatibility)
        saved_dict["actor_obs_norm_state_dict"] = self.alg._raw_actor.obs_normalizer.state_dict()
        saved_dict["critic_obs_norm_state_dict"] = self.alg._raw_critic.obs_normalizer.state_dict()
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        self.alg.load(loaded_dict)
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # Restore obs normalizer state if saved separately
        if "actor_obs_norm_state_dict" in loaded_dict:
            self.alg._raw_actor.obs_normalizer.load_state_dict(
                loaded_dict["actor_obs_norm_state_dict"]
            )
        if "critic_obs_norm_state_dict" in loaded_dict:
            self.alg._raw_critic.obs_normalizer.load_state_dict(
                loaded_dict["critic_obs_norm_state_dict"]
            )
        return loaded_dict.get("infos", {})

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_inference_policy(self, device=None):
        self.alg.actor.eval()
        self.alg.discriminator.eval()
        if device is not None:
            self.alg.actor.to(device)

        def _inference(obs):
            if isinstance(obs, TensorDict):
                return self.alg.actor(obs)
            # Plain tensor — wrap in a minimal TensorDict
            td = TensorDict({"actor": obs}, batch_size=obs.shape[0], device=obs.device)
            return self.alg.actor(td)

        return _inference

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------

    def _configure_multi_gpu(self):
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank "
                f"'{self.gpu_local_rank}'."
            )

        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.gpu_global_rank,
            world_size=self.gpu_world_size,
        )
        torch.cuda.set_device(self.gpu_local_rank)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def add_git_repo_to_log(self, repo_file_path: str):
        self.logger.git_status_repos.append(repo_file_path)
