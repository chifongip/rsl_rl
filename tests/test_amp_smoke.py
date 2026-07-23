"""Smoke test for the AMP module ported to rsl_rl v6.

Tests that AMPPPO can be constructed, act, process steps, update, and
save/load without errors.
"""

import os
import sys
import tempfile

import numpy as np
import torch
from tensordict import TensorDict

# Ensure the local rsl_rl is importable (higher priority than any installed copy)
_rsl_rl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _rsl_rl_root)

# Force reimport from the local path if an installed copy was loaded first
if "rsl_rl" in sys.modules:
    del sys.modules["rsl_rl"]
for key in list(sys.modules):
    if key.startswith("rsl_rl."):
        del sys.modules[key]

from rsl_rl.algorithms import AMPPPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import Discriminator
from rsl_rl.storage import ReplayBuffer, RolloutStorage
from rsl_rl.utils import AMPLoader, Normalizer


# ---- Minimal VecEnv stub for testing ----

class _StubRobot:
    body_names = ["pelvis", "left_foot", "right_foot", "torso"]


class _StubScene:
    robot = _StubRobot()


class _StubEnv:
    num_envs = 4
    num_actions = 12
    max_episode_length = 200
    episode_length_buf = torch.zeros(4, dtype=torch.long)
    device = "cpu"
    step_dt = 0.02

    def __init__(self, obs_groups):
        self._obs_groups = obs_groups
        self.scene = _StubScene()
        self.cfg = {}

    def get_observations(self):
        obs = {}
        for name, dim in self._obs_groups.items():
            obs[name] = torch.randn(self.num_envs, dim)
        td = TensorDict(obs, batch_size=[self.num_envs])
        return td, {}

    def step(self, actions):
        obs = {}
        for name, dim in self._obs_groups.items():
            obs[name] = torch.randn(self.num_envs, dim)
        td = TensorDict(obs, batch_size=[self.num_envs])
        rewards = torch.randn(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        # Force one done to test terminal handling
        dones[0] = True
        return td, rewards, dones, {}


# ---- Helper: create a minimal .npz motion file ----

def _make_test_npz(path: str, num_frames: int = 20, num_bodies: int = 4, num_joints: int = 12):
    np.savez(
        path,
        fps=np.array(30.0),
        joint_pos=np.random.randn(num_frames, num_joints).astype(np.float32),
        joint_vel=np.random.randn(num_frames, num_joints).astype(np.float32),
        body_pos_w=np.random.randn(num_frames, num_bodies, 3).astype(np.float32),
        body_quat_w=np.tile([1, 0, 0, 0], (num_frames, num_bodies, 1)).astype(np.float32),
        body_lin_vel_w=np.random.randn(num_frames, num_bodies, 3).astype(np.float32),
        body_ang_vel_w=np.random.randn(num_frames, num_bodies, 3).astype(np.float32),
    )


# ---- Tests ----

def test_smoke():
    """End-to-end smoke test: construct → act → step → compute_returns → update → save/load."""
    device = "cpu"
    num_envs = 4
    actor_obs_dim = 48
    critic_obs_dim = 60
    amp_obs_dim = 60  # (3+6+3+3) * 4 bodies = 60
    num_actions = 12
    num_steps = 24

    obs_groups = {
        "actor": actor_obs_dim,
        "critic": critic_obs_dim,
        "amp": amp_obs_dim,
    }
    env = _StubEnv(obs_groups)

    # Create a minimal TensorDict for construction
    obs_td = TensorDict(
        {name: torch.zeros(num_envs, dim) for name, dim in obs_groups.items()},
        batch_size=[num_envs],
    )

    # Resolve obs groups dict
    obs_groups_cfg = {"actor": ["actor"], "critic": ["critic"]}

    # Create actor and critic
    actor = MLPModel(
        obs_td, obs_groups_cfg, "actor", num_actions,
        hidden_dims=[64, 64], activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = MLPModel(
        obs_td, obs_groups_cfg, "critic", 1,
        hidden_dims=[64, 64], activation="elu",
    )

    # Create rollout storage
    storage = RolloutStorage("rl", num_envs, num_steps, obs_td, [num_actions], device)

    # Create test motion data
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test_motion.npz")
        _make_test_npz(npz_path, num_frames=20, num_bodies=4, num_joints=num_actions)

        amp_data = AMPLoader(
            motion_file=npz_path,
            body_names=["pelvis", "left_foot", "right_foot", "torso"],
            anchor_name="torso",
            all_body_names=["pelvis", "left_foot", "right_foot", "torso"],
            device=device,
        )

    amp_normalizer = Normalizer(amp_data.observation_dim)
    discriminator = Discriminator(
        amp_data.observation_dim * 2,
        amp_reward_coef=0.1,
        hidden_layer_sizes=[64, 32],
        device=device,
        task_reward_lerp=0.5,
    )

    # Construct AMPPPO
    alg = AMPPPO(
        actor=actor,
        critic=critic,
        storage=storage,
        discriminator=discriminator,
        amp_data=amp_data,
        amp_normalizer=amp_normalizer,
        amp_replay_buffer_size=1000,
        min_std=torch.tensor([0.05] * num_actions, device=device),
        num_learning_epochs=2,
        num_mini_batches=2,
        device=device,
    )

    print("[1/6] Construction OK")

    # ---- Rollout ----
    obs_result = env.get_observations()
    obs_td = obs_result[0]  # TensorDict

    for step in range(num_steps):
        actions = alg.act(obs_td)
        assert actions.shape == (num_envs, num_actions), f"Bad action shape: {actions.shape}"

        next_obs_td, rewards, dones, infos = env.step(actions)

        # Simulate amp_obs in extras
        infos.setdefault("observations", {})
        infos["observations"]["amp"] = next_obs_td["amp"]

        alg.process_env_step(next_obs_td, rewards, dones, infos)
        obs_td = next_obs_td

    print("[2/6] Rollout OK")

    # ---- Compute returns ----
    alg.compute_returns(obs_td)
    print("[3/6] Compute returns OK")

    # ---- Update ----
    loss_dict = alg.update()
    print(f"[4/6] Update OK — losses: {loss_dict}")

    # Verify loss dict keys
    expected_keys = {
        "value",
        "surrogate",
        "entropy",
        "amp",
        "amp_grad_pen",
        "amp_policy_pred",
        "amp_expert_pred",
    }
    assert expected_keys.issubset(loss_dict.keys()), f"Missing loss keys: {expected_keys - loss_dict.keys()}"

    # Check no NaN/Inf
    for k, v in loss_dict.items():
        assert np.isfinite(v), f"Non-finite loss {k}={v}"

    # ---- Save / Load ----
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = f.name

    state = alg.save()
    torch.save(state, save_path)

    loaded = torch.load(save_path, weights_only=False)
    alg.load(loaded)
    os.unlink(save_path)
    print("[5/6] Save/Load OK")

    # ---- Discriminator reward ----
    state = torch.randn(4, amp_obs_dim)
    next_state = torch.randn(4, amp_obs_dim)
    task_reward = torch.randn(4)
    reward, d = discriminator.predict_amp_reward(state, next_state, task_reward)
    assert reward.shape == (4,), f"Bad reward shape: {reward.shape}"
    assert d.shape == (4, 1), f"Bad discriminator shape: {d.shape}"
    print("[6/6] Discriminator reward OK")

    print("\n✅ All AMP smoke tests passed!")


if __name__ == "__main__":
    test_smoke()
