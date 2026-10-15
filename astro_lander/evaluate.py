"""Evaluation harness: held-out seeds, task-level metrics, stress tests and GIF rendering.

All numbers reported in the README come from here, never from training logs:
training-time returns are measured on the shaped reward with a stochastic policy,
evaluation uses the deterministic policy on the *true* environment reward.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .envs import EnvConfig, RandomizationSpec, make_env

EVAL_SEED_OFFSET = 1_000_000  # training never uses seeds this high


def run_episodes(policy: Callable, env_cfg: EnvConfig, n_episodes: int = 100,
                 seed_offset: int = EVAL_SEED_OFFSET) -> pd.DataFrame:
    """Runs `n_episodes` with the policy; one row of task metrics per episode."""
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.shaping = "default"  # metrics wrapper records the true reward regardless
    env_cfg.randomization = RandomizationSpec()
    env = make_env(env_cfg, seed=seed_offset)
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        m = dict(info["episode_metrics"])
        m["seed"] = seed_offset + ep
        rows.append(m)
    env.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    landed = df[df.outcome == "landed"]
    return dict(
        episodes=len(df),
        mean_return=float(df.true_return.mean()),
        std_return=float(df.true_return.std(ddof=0)),
        solved_rate=float((df.true_return >= 200).mean()),  # Gymnasium's "solved" bar per episode
        success_rate=float(df.on_pad.mean()),               # landed and came to rest between the flags
        landed_rate=float((df.outcome == "landed").mean()),
        crash_rate=float(df.outcome.isin(["crashed", "out_of_bounds"]).mean()),
        timeout_rate=float((df.outcome == "timeout").mean()),
        mean_fuel=float(df.fuel.mean()),
        mean_length=float(df.length.mean()),
        mean_abs_landing_x=float(landed.landing_x.abs().mean()) if len(landed) else float("nan"),
        mean_touchdown_speed=float(landed.touchdown_speed.mean()) if len(landed) else float("nan"),
    )


def stress_test(policy: Callable, base: EnvConfig, n_episodes: int = 30,
                winds=(0.0, 5.0, 10.0, 15.0, 20.0), gravities=(-6.0, -8.0, -10.0, -12.0),
                turbulence: float = 1.0) -> pd.DataFrame:
    """Success rate over a grid of (wind, gravity). wind=0 means wind disabled."""
    rows = []
    for g in gravities:
        for w in winds:
            cfg = replace(copy.deepcopy(base), gravity=g, enable_wind=w > 0, wind_power=w,
                          turbulence_power=turbulence if w > 0 else 0.0)
            s = summarize(run_episodes(policy, cfg, n_episodes))
            rows.append(dict(gravity=g, wind_power=w, **s))
    return pd.DataFrame(rows)


def render_gif(policy: Callable, env_cfg: EnvConfig, path: str | Path, seed: int = EVAL_SEED_OFFSET,
               fps: int = 30, frame_skip: int = 2, max_frames: int = 400) -> dict:
    import imageio.v2 as imageio

    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.randomization = RandomizationSpec()
    env = make_env(env_cfg, seed=seed, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, t, done = [env.render()], 0, False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs))
        done = term or trunc
        t += 1
        if t % frame_skip == 0 and len(frames) < max_frames:
            frames.append(env.render())
    frames += [env.render()] * 15  # hold the final frame
    env.close()
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    return info["episode_metrics"]
