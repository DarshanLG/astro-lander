import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astro_lander.envs import (EnvConfig, LandingMetrics, RandomizationSpec, RewardShaping,
                               lander_potential, make_env, make_vec_env)
from astro_lander.ppo import ActorCritic, PPOConfig, Policy, RunningMeanStd, ObsNormalizer, train
from astro_lander.evaluate import run_episodes, summarize


def rollout(env, policy=None, seed=0, max_steps=2000):
    obs, _ = env.reset(seed=seed)
    rewards, infos = [], []
    for _ in range(max_steps):
        a = env.action_space.sample() if policy is None else policy(obs)
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        if term or trunc:
            return rewards, info
    raise AssertionError("episode did not end")


def test_default_shaping_matches_raw_env():
    import gymnasium as gym
    raw = gym.make("LunarLander-v3")
    ours = make_env(EnvConfig(shaping="default"), seed=0)
    for env in (raw, ours):
        env.action_space.seed(0)
    r1, _ = rollout(raw, seed=3)
    r2, _ = rollout(ours, seed=3)
    assert np.allclose(r1, r2)


def test_metrics_true_return_independent_of_shaping():
    rets = {}
    for scheme in RewardShaping.SCHEMES:
        env = make_env(EnvConfig(shaping=scheme), seed=0)
        env.action_space.seed(0)
        rewards, info = rollout(env, seed=7)
        rets[scheme] = info["episode_metrics"]["true_return"]
    assert len({round(v, 6) for v in rets.values()}) == 1


def test_pbrs_telescopes():
    """Undiscounted check: with gamma=1, sum of PBRS terms = Phi(terminal)=0 - Phi(s0)."""
    gamma = 1.0
    env = make_env(EnvConfig(shaping="pbrs", gamma=gamma), seed=0)
    sparse = make_env(EnvConfig(shaping="sparse", gamma=gamma), seed=0)
    env.action_space.seed(1); sparse.action_space.seed(1)
    obs0, _ = env.reset(seed=11)
    r_pbrs, info = rollout(env, seed=11)
    r_sparse, _ = rollout(sparse, seed=11)
    assert info["episode_metrics"]["outcome"] in ("crashed", "out_of_bounds", "landed")
    assert math.isclose(sum(r_pbrs) - sum(r_sparse), -lander_potential(obs0), rel_tol=1e-6, abs_tol=1e-6)


def test_sparse_reward_is_zero_except_fuel_and_terminal():
    env = make_env(EnvConfig(shaping="sparse"), seed=0)
    obs, _ = env.reset(seed=0)
    obs, r, term, trunc, _ = env.step(0)  # no-op, no fuel
    assert r == 0.0


def test_domain_randomization_changes_physics():
    spec = RandomizationSpec(gravity=(-11.0, -7.0), wind_power=(0.0, 15.0), turbulence_power=(0.0, 1.5))
    env = make_env(EnvConfig(randomization=spec), seed=0)
    seen = set()
    for s in range(5):
        _, info = env.reset(seed=s)
        g = info["physics"]["gravity"]
        assert -11.0 <= g <= -7.0
        assert env.unwrapped.world.gravity[1] == pytest.approx(g)
        seen.add(round(g, 4))
    assert len(seen) == 5


def test_outcomes_are_valid():
    env = make_env(EnvConfig(), seed=0)
    for s in range(5):
        _, info = rollout(env, seed=s)
        assert info["episode_metrics"]["outcome"] in ("landed", "crashed", "out_of_bounds", "timeout")


def test_running_mean_std():
    rms = RunningMeanStd((3,))
    data = np.random.default_rng(0).normal(5, 2, size=(10_000, 3))
    for chunk in np.array_split(data, 37):
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(0), atol=1e-3)
    assert np.allclose(rms.var, data.var(0), atol=1e-2)


@pytest.mark.parametrize("continuous", [False, True])
def test_actor_critic_shapes(continuous):
    net = ActorCritic(8, 2 if continuous else 4, continuous)
    obs = torch.randn(5, 8)
    a, logp, ent, v = net.act(obs)
    assert logp.shape == (5,) and ent.shape == (5,) and v.shape == (5,)
    _, logp2, _, _ = net.act(obs, a)
    assert torch.allclose(logp, logp2)


def test_config_roundtrip():
    cfg = PPOConfig(env=EnvConfig(shaping="pbrs", randomization=RandomizationSpec(gravity=(-11, -8))))
    cfg2 = PPOConfig.from_dict(cfg.to_dict())
    assert cfg2.env.shaping == "pbrs" and cfg2.env.randomization.gravity == (-11, -8)


@pytest.mark.parametrize("continuous", [False, True])
def test_train_smoke_and_reload(tmp_path, continuous):
    cfg = PPOConfig(run_name="t", total_timesteps=2048, num_envs=2, num_steps=512, num_minibatches=4,
                    env=EnvConfig(continuous=continuous))
    run = train(cfg, tmp_path, verbose=False)
    pol = Policy.load(run / "final.pt")
    df = run_episodes(pol, cfg.env, n_episodes=2)
    s = summarize(df)
    assert s["episodes"] == 2 and (run / "metrics.csv").exists()
