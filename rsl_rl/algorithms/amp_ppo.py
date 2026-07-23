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

"""PPO with Adversarial Motion Priors (AMP) discriminator loss.

Adapts the standard PPO algorithm to include an AMP discriminator that shapes
rewards based on similarity to expert motion demonstrations.  Designed for the
rsl_rl v6 architecture (TensorDict observations, separate MLPModel
actor/critic, RolloutStorage.Batch iteration).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from collections.abc import Iterable
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import Discriminator
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import (
    AMPLoader,
    Normalizer,
    resolve_callable,
    resolve_obs_groups,
    resolve_optimizer,
)


class AMPPPO(PPO):
    """PPO variant that adds an AMP discriminator loss.

    The discriminator distinguishes expert motion transitions from policy
    transitions, producing a shaped reward signal.  The discriminator loss
    (MSE + gradient penalty) is added to the standard PPO surrogate + value
    losses.

    All v6 PPO features (RND, symmetry, multi-GPU) are preserved.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # AMP-specific
        discriminator: nn.Module,
        amp_data: AMPLoader,
        amp_normalizer: Normalizer,
        amp_replay_buffer_size: int = 100_000,
        min_std: torch.Tensor | None = None,
        # Standard PPO parameters
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # Optional extras
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize PPO and AMP discriminator training components."""
        # ---- AMP components (set before super().__init__ so optimizer picks
        #      them up) ----
        self.discriminator = discriminator
        self.discriminator.to(device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        if amp_normalizer.mean.numel() != discriminator.input_dim // 2:
            raise ValueError("AMP normalizer dimension must equal half the discriminator input dimension.")
        self.amploss_coef = 1.0
        self.min_std = min_std
        if min_std is not None and (min_std < 0).any():
            raise ValueError("min_std values must be nonnegative.")
        self.intrinsic_rewards: torch.Tensor | None = None
        self.last_amp_rewards: torch.Tensor | None = None
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)

        # ---- Parent PPO init (creates optimizer over actor + critic) ----
        # We override optimizer creation below to also cover discriminator.
        super().__init__(
            actor,
            critic,
            storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # Rebuild the optimizer to include discriminator parameters. Parameters
        # are deduplicated because actor and critic may share CNN encoders.
        seen_parameters: set[int] = set()

        def unique_parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
            unique: list[nn.Parameter] = []
            for parameter in parameters:
                if id(parameter) not in seen_parameters:
                    seen_parameters.add(id(parameter))
                    unique.append(parameter)
            return unique

        parameter_groups = [
            {"params": unique_parameters(self._raw_actor.parameters()), "name": "actor"},
            {"params": unique_parameters(self._raw_critic.parameters()), "name": "critic"},
            {
                "params": unique_parameters(self.discriminator.trunk.parameters()),
                "weight_decay": 10e-4,
                "name": "amp_trunk",
            },
            {
                "params": unique_parameters(self.discriminator.amp_linear.parameters()),
                "weight_decay": 10e-2,
                "name": "amp_head",
            },
        ]
        self.optimizer = resolve_optimizer(optimizer)(parameter_groups, lr=learning_rate)

        # Cache the actor's state-independent std parameter for min_std
        # clamping, preserving its scalar/log parameterization.
        self._actor_std_param: torch.Tensor | None = None
        self._actor_std_is_log = False
        if self.min_std is not None:
            distribution = getattr(self._raw_actor, "distribution", None)
            if distribution is not None:
                if hasattr(distribution, "std_param"):
                    self._actor_std_param = distribution.std_param
                elif hasattr(distribution, "log_std_param"):
                    self._actor_std_param = distribution.log_std_param
                    self._actor_std_is_log = True

    # ------------------------------------------------------------------
    # Factory (replaces PPO.construct_algorithm for AMP tasks)
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str = "cpu") -> AMPPPO:
        """Build actor, critic, discriminator, and AMPPPO from a config dict.

        Follows the same pattern as :meth:`PPO.construct_algorithm` but also
        creates the AMP components.
        """
        alg_cfg = cfg["algorithm"]
        actor_cfg = cfg["actor"]
        critic_cfg = cfg["critic"]

        # Observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(
            obs,
            cfg.get("obs_groups", {}),
            default_sets=default_sets,
        )
        obs_groups = cfg["obs_groups"]

        # Actor & critic
        actor_class = resolve_callable(actor_cfg.get("class_name", "MLPModel"))
        actor_kwargs = {k: v for k, v in actor_cfg.items() if k != "class_name"}
        actor: MLPModel = actor_class(obs, obs_groups, "actor", env.num_actions, **actor_kwargs)
        print(f"Actor Model: {actor}")
        critic_class = resolve_callable(critic_cfg.get("class_name", "MLPModel"))
        critic_kwargs = {k: v for k, v in critic_cfg.items() if k != "class_name"}
        # Share CNN encoders if configured
        if cfg["algorithm"].pop("share_cnn_encoders", None) and hasattr(actor, "cnns") and actor.cnns:
            critic_kwargs["cnns"] = actor.cnns
        critic: MLPModel = critic_class(obs, obs_groups, "critic", 1, **critic_kwargs)
        print(f"Critic Model: {critic}")

        # Rollout storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # AMP components
        robot_entity = env.unwrapped.scene["robot"]
        all_body_names = robot_entity.body_names
        amp_data = AMPLoader(
            motion_file=alg_cfg["amp_motion_files"],
            body_names=alg_cfg["amp_body_names"],
            anchor_name=alg_cfg["amp_anchor_name"],
            all_body_names=all_body_names,
            device=device,
            step_dt=float(env.unwrapped.step_dt),
        )
        if "amp" not in obs:
            raise ValueError("AMP training requires an 'amp' observation group.")
        if obs["amp"].ndim != 2 or obs["amp"].shape[-1] != amp_data.observation_dim:
            raise ValueError(
                f"AMP observation must have shape (num_envs, {amp_data.observation_dim}), "
                f"got {tuple(obs['amp'].shape)}."
            )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = Discriminator(
            amp_data.observation_dim * 2,
            alg_cfg["amp_reward_coef"],
            alg_cfg["amp_discr_hidden_dims"],
            device,
            alg_cfg.get("amp_task_reward_lerp", 0.0),
        )

        # min_std
        min_std_values = list(alg_cfg.get("min_normalized_std", []))
        if any(value < 0 for value in min_std_values):
            raise ValueError("min_normalized_std values must be nonnegative.")
        num_actions = env.num_actions
        if not min_std_values:
            min_std_values = [0.0] * num_actions
        elif len(min_std_values) == 1:
            min_std_values = min_std_values * num_actions
        elif len(min_std_values) < num_actions:
            pad_value = min_std_values[-1]
            min_std_values = min_std_values + [pad_value] * (num_actions - len(min_std_values))
        elif len(min_std_values) > num_actions:
            min_std_values = min_std_values[:num_actions]
        min_std = torch.tensor(min_std_values, device=device, requires_grad=False)

        # RND config
        cfg["algorithm"] = resolve_rnd_config(alg_cfg, obs, obs_groups, env)

        # Symmetry config
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Build AMPPPO
        alg = AMPPPO(
            actor=actor,
            critic=critic,
            storage=storage,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            amp_replay_buffer_size=alg_cfg.get("amp_replay_buffer_size", 100_000),
            min_std=min_std,
            num_learning_epochs=alg_cfg.get("num_learning_epochs", 5),
            num_mini_batches=alg_cfg.get("num_mini_batches", 4),
            clip_param=alg_cfg.get("clip_param", 0.2),
            gamma=alg_cfg.get("gamma", 0.99),
            lam=alg_cfg.get("lam", 0.95),
            value_loss_coef=alg_cfg.get("value_loss_coef", 1.0),
            entropy_coef=alg_cfg.get("entropy_coef", 0.01),
            learning_rate=alg_cfg.get("learning_rate", 0.001),
            max_grad_norm=alg_cfg.get("max_grad_norm", 1.0),
            optimizer=alg_cfg.get("optimizer", "adam"),
            use_clipped_value_loss=alg_cfg.get("use_clipped_value_loss", True),
            schedule=alg_cfg.get("schedule", "adaptive"),
            desired_kl=alg_cfg.get("desired_kl", 0.01),
            normalize_advantage_per_mini_batch=alg_cfg.get("normalize_advantage_per_mini_batch", False),
            device=device,
            rnd_cfg=alg_cfg.get("rnd_cfg", None),
            symmetry_cfg=alg_cfg.get("symmetry_cfg", None),
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )

        # Ensure obs normalizer buffers are on the correct device.
        # EmpiricalNormalization registers CPU buffers in __init__;
        # while PPO.__init__ calls model.to(device), an explicit call
        # guards against edge cases in the full training pipeline.
        for model in (alg._raw_actor, alg._raw_critic):
            norm = getattr(model, "obs_normalizer", None)
            if norm is not None and hasattr(norm, "_mean"):
                norm.to(device)

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and record AMP observations for this step."""
        self.amp_transition.observations = obs["amp"]
        return super().act(obs)

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Shape rewards, record valid AMP transitions, and delegate PPO storage."""
        current_amp_obs = self.amp_transition.observations
        next_amp_obs = obs["amp"]
        valid_mask = ~dones.bool()

        legacy_next_obs = extras.get("observations", {}).get("amp")
        if legacy_next_obs is not None:
            next_amp_obs = legacy_next_obs.to(self.device)
            valid_mask = torch.ones_like(dones, dtype=torch.bool)

        terminal_amp_obs = extras.get("terminal_observations", {}).get("amp")
        if terminal_amp_obs is not None:
            next_amp_obs = next_amp_obs.clone()
            done_mask = dones.bool()
            terminal_amp_obs = terminal_amp_obs.to(self.device)
            next_amp_obs[done_mask] = terminal_amp_obs[done_mask]
            valid_mask = torch.ones_like(dones, dtype=torch.bool)

        shaped_rewards = rewards.clone()
        if valid_mask.any():
            shaped_rewards[valid_mask] = self.discriminator.predict_amp_reward(
                current_amp_obs[valid_mask],
                next_amp_obs[valid_mask],
                rewards[valid_mask],
                normalizer=self.amp_normalizer,
            )[0]
            self.amp_storage.insert(
                current_amp_obs[valid_mask],
                next_amp_obs[valid_mask],
            )
        self.last_amp_rewards = shaped_rewards

        super().process_env_step(obs, shaped_rewards, dones, extras)
        self.amp_transition.clear()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run one PPO + AMP discriminator update epoch."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_rnd_loss: float | None = 0.0 if self.rnd else None
        mean_symmetry_loss: float | None = 0.0 if self.symmetry else None
        effective_updates = 0
        skipped_non_finite_batches = 0

        # Mini-batch generators
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        num_total_batches = self.num_learning_epochs * self.num_mini_batches
        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        amp_policy_generator = self.amp_storage.feed_forward_generator(num_total_batches, mini_batch_size)
        amp_expert_generator = self.amp_data.feed_forward_generator(num_total_batches, mini_batch_size)

        for batch, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            original_batch_size = batch.observations.batch_size[0]

            # ---- Per-mini-batch advantage normalization ----
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # ---- Symmetry augmentation (before actor forward, matches ppo.py) ----
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # ---- Recompute log-probs and values ----
            # Forward pass to update the actor's distribution
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0] if batch.hidden_states else None,
                stochastic_output=True,
            )
            actions_log_prob_batch = self.actor.get_output_log_prob(batch.actions)
            value_batch = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1] if batch.hidden_states else None,
            )

            batch_is_finite = torch.isfinite(batch.returns).all() and torch.isfinite(value_batch).all()
            if not self._all_ranks_true(batch_is_finite):
                skipped_non_finite_batches += 1
                continue

            entropy_batch = self.actor.output_entropy[:original_batch_size]

            # ---- Adaptive LR ----
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ---- Surrogate loss ----
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ---- Value loss ----
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (value_batch - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # ---- Symmetry loss ----
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # ---- RND loss ----
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None

            # ---- AMP discriminator loss ----
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state_norm = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state_norm = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state_norm = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state_norm = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                policy_d = self.discriminator(torch.cat([policy_state_norm, policy_next_state_norm], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state_norm, expert_next_state_norm], dim=-1))
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    expert_state_norm, expert_next_state_norm, lambda_=10
                )
            else:
                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)
            expert_loss = functional.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            loss = loss + self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            if not self._all_ranks_true(torch.isfinite(loss)):
                skipped_non_finite_batches += 1
                continue

            # ---- Backward ----
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd and rnd_loss is not None:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            # Clamp min std
            self._clamp_actor_std()

            # Update AMP normalizer with globally aggregated raw data.
            if self.amp_normalizer is not None:
                self._update_amp_normalizer(policy_state, expert_state)

            # Accumulate losses
            effective_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            if mean_rnd_loss is not None and rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # ---- Average losses ----
        n = max(effective_updates, 1)
        self.storage.clear()
        loss_dict = {
            "value": mean_value_loss / n,
            "surrogate": mean_surrogate_loss / n,
            "entropy": mean_entropy / n,
            "amp": mean_amp_loss / n,
            "amp_grad_pen": mean_grad_pen_loss / n,
            "amp_policy_pred": mean_policy_pred / n,
            "amp_expert_pred": mean_expert_pred / n,
            "skipped_non_finite_batches": float(skipped_non_finite_batches),
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss / n
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss / n
        return loss_dict

    def _clamp_actor_std(self) -> None:
        """Clamp a state-independent actor standard deviation to ``min_std``."""
        if self._actor_std_param is None or self.min_std is None:
            return
        with torch.no_grad():
            if self.min_std.numel() == 1:
                clamped_min = self.min_std.expand_as(self._actor_std_param)
            elif self.min_std.numel() != self._actor_std_param.numel():
                clamped_min = self.min_std.min().expand_as(self._actor_std_param)
            else:
                clamped_min = self.min_std
            clamped_min = clamped_min.clamp_min(1e-6)
            if self._actor_std_is_log:
                clamped_min = torch.log(clamped_min)
            self._actor_std_param.clamp_(min=clamped_min)

    def _update_amp_normalizer(self, policy_state: torch.Tensor, expert_state: torch.Tensor) -> None:
        """Update identical AMP statistics on every distributed rank."""
        samples = torch.cat([policy_state, expert_state], dim=0).float()
        batch_sum = samples.sum(dim=0)
        batch_square_sum = samples.square().sum(dim=0)
        batch_count = torch.tensor(float(samples.shape[0]), device=samples.device)
        if self.is_multi_gpu:
            torch.distributed.all_reduce(batch_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(batch_square_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(batch_count, op=torch.distributed.ReduceOp.SUM)
        mean = batch_sum / batch_count
        var = (batch_square_sum / batch_count - mean.square()).clamp_min(0.0)
        self.amp_normalizer.update_from_moments(
            mean.cpu(),
            var.cpu(),
            int(batch_count.item()),
        )
        self.amp_normalizer.invalidate_cache()

    def _all_ranks_true(self, condition: torch.Tensor | bool) -> bool:
        """Return one synchronized decision for non-finite batch handling."""
        condition_tensor = torch.as_tensor(condition, device=self.device, dtype=torch.int32)
        if self.is_multi_gpu:
            torch.distributed.all_reduce(condition_tensor, op=torch.distributed.ReduceOp.MIN)
        return bool(condition_tensor.item())

    def train_mode(self) -> None:
        """Set PPO and discriminator modules to training mode."""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Set PPO and discriminator modules to evaluation mode."""
        super().eval_mode()
        self.discriminator.eval()

    # ------------------------------------------------------------------
    # Save / Load (v6 format)
    # ------------------------------------------------------------------

    def save(self) -> dict:
        """Serialize algorithm state for checkpointing."""
        result = super().save()
        result["discriminator_state_dict"] = self.discriminator.state_dict()
        result["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
        return result

    def load(
        self,
        loaded_dict: dict,
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> bool:
        """Load algorithm state from a checkpoint dict."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_amp = load_cfg is None or load_cfg.get("amp", True)
        if load_amp and "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if load_amp and "amp_normalizer_state_dict" in loaded_dict:
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
        elif load_amp and "amp_normalizer" in loaded_dict:
            legacy_normalizer = loaded_dict["amp_normalizer"]
            self.amp_normalizer.load_state_dict({
                "mean": legacy_normalizer.mean,
                "var": legacy_normalizer.var,
                "count": legacy_normalizer.count,
                "epsilon": legacy_normalizer.epsilon,
                "clip_obs": legacy_normalizer.clip_obs,
            })
        return load_iteration

    # ------------------------------------------------------------------
    # Multi-GPU (includes discriminator)
    # ------------------------------------------------------------------

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [
            self._raw_actor.state_dict(),
            self._raw_critic.state_dict(),
            self.discriminator.state_dict(),
        ]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        self.discriminator.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        all_params = list(
            chain(
                self._raw_actor.parameters(),
                self._raw_critic.parameters(),
                self.discriminator.parameters(),
            )
        )
        if self.rnd:
            all_params = list(chain(all_params, self.rnd.parameters()))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
