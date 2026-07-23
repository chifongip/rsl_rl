# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused regression tests for the AMP PPO integration."""

from __future__ import annotations

import copy
import torch
from collections.abc import Generator, Iterator
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.models import CNNModel, MLPModel, RNNModel
from rsl_rl.modules import Discriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import Normalizer

NUM_ENVS = 4
NUM_STEPS = 4
NUM_ACTIONS = 3
AMP_DIM = 6


class _AMPData:
    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        for _ in range(num_mini_batch):
            yield torch.randn(mini_batch_size, AMP_DIM), torch.randn(mini_batch_size, AMP_DIM)


def _make_obs(include_image: bool = False) -> TensorDict:
    data = {
        "policy": torch.randn(NUM_ENVS, 8),
        "amp": torch.randn(NUM_ENVS, AMP_DIM),
    }
    if include_image:
        data["image"] = torch.randn(NUM_ENVS, 1, 16, 16)
    return TensorDict(data, batch_size=[NUM_ENVS])


def _build_amp(
    *,
    std_type: str = "scalar",
    optimizer: str = "adam",
    critic_recurrent: bool = False,
) -> AMPPPO:
    obs = _make_obs()
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": std_type,
        },
    )
    critic_class = RNNModel if critic_recurrent else MLPModel
    critic_kwargs = (
        {"hidden_dims": [16], "rnn_hidden_dim": 8, "rnn_num_layers": 1} if critic_recurrent else {"hidden_dims": [16]}
    )
    critic = critic_class(obs, obs_groups, "critic", 1, **critic_kwargs)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    return AMPPPO(
        actor,
        critic,
        storage,
        discriminator=Discriminator(AMP_DIM * 2, 0.1, [16], "cpu"),
        amp_data=_AMPData(),
        amp_normalizer=Normalizer(AMP_DIM),
        min_std=torch.full((NUM_ACTIONS,), 0.05),
        optimizer=optimizer,
        schedule="fixed",
    )


def test_shared_cnn_parameters_are_only_registered_once() -> None:
    """Shared actor/critic encoders should occur in exactly one optimizer group."""
    obs = _make_obs(include_image=True)
    obs_groups = {
        "actor": ["policy", "image"],
        "critic": ["policy", "image"],
    }
    cnn_cfg = {"output_channels": [4], "kernel_size": 3, "stride": 2}
    actor = CNNModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        cnn_cfg=cnn_cfg,
        distribution_cfg={"class_name": "GaussianDistribution"},
    )
    critic = CNNModel(
        obs,
        obs_groups,
        "critic",
        1,
        hidden_dims=[16],
        cnns=actor.cnns,
    )
    alg = AMPPPO(
        actor,
        critic,
        RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS]),
        discriminator=Discriminator(AMP_DIM * 2, 0.1, [16], "cpu"),
        amp_data=_AMPData(),
        amp_normalizer=Normalizer(AMP_DIM),
        schedule="fixed",
    )

    optimizer_parameters = [parameter for group in alg.optimizer.param_groups for parameter in group["params"]]
    assert len(optimizer_parameters) == len({id(parameter) for parameter in optimizer_parameters})


def test_optimizer_configuration_is_honored() -> None:
    """AMP should construct the optimizer selected by configuration."""
    optimizer_classes = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    for name, optimizer_class in optimizer_classes.items():
        assert isinstance(_build_amp(optimizer=name).optimizer, optimizer_class)


def test_log_std_uses_scalar_space_minimum() -> None:
    """Log-parameterized policies should interpret min_std in scalar space."""
    alg = _build_amp(std_type="log")
    alg._actor_std_param.data.fill_(-20.0)
    alg._clamp_actor_std()
    assert torch.allclose(
        alg._actor_std_param.exp(),
        torch.full((NUM_ACTIONS,), 0.05),
    )


def test_scalar_std_uses_scalar_space_minimum() -> None:
    """Scalar-parameterized policies should clamp directly to min_std."""
    alg = _build_amp(std_type="scalar")
    alg._actor_std_param.data.fill_(-20.0)
    alg._clamp_actor_std()
    assert torch.allclose(
        alg._actor_std_param,
        torch.full((NUM_ACTIONS,), 0.05),
    )


def test_recurrent_critic_hidden_state_and_generator_are_used() -> None:
    """A recurrent critic should store state and select recurrent minibatches."""
    alg = _build_amp(critic_recurrent=True)
    obs = _make_obs()
    alg.act(obs)
    alg.act(obs)
    assert alg.transition.hidden_states[0] is None
    assert alg.transition.hidden_states[1] is not None

    generator_used = {"recurrent": False}

    def recurrent_generator(*_args: object) -> Iterator[object]:
        generator_used["recurrent"] = True
        return iter(())

    def feed_forward_generator(*_args: object) -> Iterator[object]:
        raise AssertionError("Feed-forward generator must not be used with a recurrent critic.")

    alg.storage.recurrent_mini_batch_generator = recurrent_generator
    alg.storage.mini_batch_generator = feed_forward_generator
    alg.update()
    assert generator_used["recurrent"]


def test_terminal_without_observation_is_excluded_from_amp_replay() -> None:
    """Missing terminal observations should preserve task reward and skip replay."""
    alg = _build_amp()
    obs = _make_obs()
    alg.act(obs)
    next_obs = _make_obs()
    rewards = torch.arange(NUM_ENVS, dtype=torch.float32)
    dones = torch.tensor([True, False, False, False])

    alg.process_env_step(next_obs, rewards, dones, {})

    assert alg.amp_storage.num_samples == NUM_ENVS - 1
    assert alg.last_amp_rewards[0] == rewards[0]


def test_true_terminal_observation_is_used_for_amp_replay() -> None:
    """Provided terminal observations should form valid AMP transitions."""
    alg = _build_amp()
    obs = _make_obs()
    alg.act(obs)
    next_obs = _make_obs()
    terminal_obs = torch.full((NUM_ENVS, AMP_DIM), 42.0)
    dones = torch.tensor([True, False, False, False])

    alg.process_env_step(
        next_obs,
        torch.ones(NUM_ENVS),
        dones,
        {"terminal_observations": {"amp": terminal_obs}},
    )

    assert alg.amp_storage.num_samples == NUM_ENVS
    assert torch.equal(alg.amp_storage.next_states[0], terminal_obs[0])


def _build_ppo_amp_pair() -> tuple[PPO, AMPPPO, TensorDict]:
    """Construct PPO and AMP PPO with identical actor and critic parameters."""
    torch.manual_seed(7)
    obs = _make_obs()
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[16])
    ppo = PPO(
        actor,
        critic,
        RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS]),
        num_learning_epochs=1,
        num_mini_batches=2,
        schedule="fixed",
    )
    amp = AMPPPO(
        copy.deepcopy(actor),
        copy.deepcopy(critic),
        RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS]),
        discriminator=Discriminator(AMP_DIM * 2, 0.1, [16], "cpu"),
        amp_data=_AMPData(),
        amp_normalizer=Normalizer(AMP_DIM),
        num_learning_epochs=1,
        num_mini_batches=2,
        schedule="fixed",
    )
    amp.amploss_coef = 0.0
    return ppo, amp, obs


def _fill_identical_rollout(algorithm: PPO, obs: TensorDict) -> None:
    """Populate an algorithm without applying AMP reward shaping."""
    torch.manual_seed(11)
    for step in range(NUM_STEPS):
        step_obs = obs.clone()
        step_obs["policy"] = obs["policy"] + step * 0.01
        algorithm.act(step_obs)
        algorithm.transition.rewards = torch.linspace(0.0, 1.0, NUM_ENVS)
        algorithm.transition.dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
        algorithm.storage.add_transition(algorithm.transition)
        algorithm.transition.clear()
    algorithm.compute_returns(obs)


def test_amp_disabled_matches_ppo_parameter_update() -> None:
    """AMP-specific losses must not perturb the underlying PPO update."""
    ppo, amp, obs = _build_ppo_amp_pair()
    _fill_identical_rollout(ppo, obs)
    _fill_identical_rollout(amp, obs)
    amp.amp_storage.insert(
        torch.randn(NUM_ENVS * NUM_STEPS, AMP_DIM),
        torch.randn(NUM_ENVS * NUM_STEPS, AMP_DIM),
    )

    torch.manual_seed(19)
    ppo_losses = ppo.update()
    torch.manual_seed(19)
    amp_losses = amp.update()

    for ppo_param, amp_param in zip(ppo.actor.parameters(), amp.actor.parameters()):
        assert torch.allclose(ppo_param, amp_param, atol=1e-7, rtol=1e-6)
    for ppo_param, amp_param in zip(ppo.critic.parameters(), amp.critic.parameters()):
        assert torch.allclose(ppo_param, amp_param, atol=1e-7, rtol=1e-6)
    for key in ("value", "surrogate", "entropy"):
        assert amp_losses[key] == pytest.approx(ppo_losses[key], abs=1e-7)


def test_checkpoint_restores_learning_rate_and_portable_normalizer() -> None:
    """AMP loading should retain PPO LR semantics and rebuild normalizer caches."""
    alg = _build_amp()
    alg.amp_normalizer.update(torch.randn(32, AMP_DIM))
    alg.amp_normalizer.normalize_torch(torch.randn(2, AMP_DIM), "cpu")
    expected_mean = alg.amp_normalizer.mean.clone()
    for group in alg.optimizer.param_groups:
        group["lr"] = 2.5e-4
    state = alg.save()

    restored = _build_amp()
    restored.load(state)

    assert restored.learning_rate == pytest.approx(2.5e-4)
    assert torch.equal(restored.amp_normalizer.mean, expected_mean)
    assert restored.amp_normalizer._cached_device is None
    assert restored.amp_normalizer._cached_mean is None
