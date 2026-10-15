"""Environment construction: Lunar Lander + metrics, reward shaping and domain randomization.

Wrapper order (inner -> outer):
    LunarLander-v3 (+ TimeLimit) -> LandingMetrics -> DomainRandomization -> RewardShaping

`LandingMetrics` sits closest to the raw env so it always records the *true* task
return and outcome, no matter which shaped reward the agent is trained on. Every
comparison in this project is made on those true metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

ENV_ID = "LunarLander-v3"
HELIPAD_HALF_WIDTH = 0.2  # in normalized x units; the flags sit at roughly +-0.2

OUTCOMES = ("landed", "crashed", "out_of_bounds", "timeout")


def fuel_cost(u, action) -> float:
    """Engine power used this step, using the env's own bookkeeping (main 0.3, side 0.03)."""
    if u.continuous:
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        m = (np.clip(a[0], 0, 1) + 1) * 0.5 if a[0] > 0 else 0.0
        s = np.clip(abs(a[1]), 0.5, 1.0) if abs(a[1]) > 0.5 else 0.0
    else:
        a = int(action)
        m = 1.0 if a == 2 else 0.0
        s = 1.0 if a in (1, 3) else 0.0
    return 0.30 * m + 0.03 * s


# --------------------------------------------------------------------------- metrics
class LandingMetrics(gym.Wrapper):
    """Records task-level metrics and writes them into `info` at episode end.

    info["episode_metrics"] = {
        true_return, length, outcome, on_pad, fuel, landing_x, touchdown_speed
    }
    `fuel` uses the env's own engine-power bookkeeping (main 0.3/step, side 0.03/step
    at full power), so it is the same quantity the default reward penalizes.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()

    def _reset_stats(self):
        self.true_return = 0.0
        self.length = 0
        self.fuel = 0.0
        self.max_touchdown_speed = 0.0
        self._prev_contact = False

    def reset(self, **kwargs):
        self._reset_stats()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.true_return += float(reward)
        self.length += 1
        self.fuel += fuel_cost(self.env.unwrapped, action)
        contact = bool(obs[6] > 0.5 or obs[7] > 0.5)
        if contact and not self._prev_contact:
            self.max_touchdown_speed = max(self.max_touchdown_speed, float(math.hypot(obs[2], obs[3])))
        self._prev_contact = contact

        if terminated or truncated:
            u = self.env.unwrapped
            x = float(obs[0])
            if truncated and not terminated:
                outcome = "timeout"
            elif abs(x) >= 1.0:
                outcome = "out_of_bounds"
            elif u.game_over:
                outcome = "crashed"
            elif not u.lander.awake:
                outcome = "landed"
            else:  # should not happen, keep it explicit
                outcome = "crashed"
            info["episode_metrics"] = dict(
                true_return=self.true_return,
                length=self.length,
                outcome=outcome,
                on_pad=bool(outcome == "landed" and abs(x) <= HELIPAD_HALF_WIDTH),
                fuel=self.fuel,
                landing_x=x,
                touchdown_speed=self.max_touchdown_speed,
            )
        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------------- shaping
def lander_potential(obs: np.ndarray) -> float:
    """Phi(s): the same terms Gymnasium uses inside its default reward.

    -100 * distance to pad - 100 * speed - 100 * |tilt| + 10 per leg in contact.
    """
    x, y, vx, vy, ang, _, l1, l2 = obs[:8]
    return float(
        -100.0 * math.hypot(x, y)
        - 100.0 * math.hypot(vx, vy)
        - 100.0 * abs(ang)
        + 10.0 * l1
        + 10.0 * l2
    )


class RewardShaping(gym.Wrapper):
    """Replaces the env reward with one of several schemes.

    default : Gymnasium's built-in reward (dense shaping difference + fuel + terminal +-100).
    sparse  : terminal +-100 and fuel only. The signal the task "really" cares about.
    pbrs    : sparse + gamma*Phi(s') - Phi(s). Potential-based shaping (Ng, Harada &
              Russell, 1999) provably leaves the optimal policy unchanged while making
              credit assignment dense. Unlike the default reward it uses the agent's
              discount, which is what the guarantee requires.
    naive   : sparse + a per-step bonus for being close / slow / upright (a *state* bonus,
              not a difference of potentials). Intuitive, but it pays the agent for
              hovering near the pad, which is a textbook source of reward hacking.
    """

    SCHEMES = ("default", "sparse", "pbrs", "naive")

    def __init__(self, env: gym.Env, scheme: str = "default", gamma: float = 0.99):
        super().__init__(env)
        if scheme not in self.SCHEMES:
            raise ValueError(f"unknown shaping scheme {scheme!r}; choose from {self.SCHEMES}")
        self.scheme = scheme
        self.gamma = gamma
        self._phi = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi = lander_potential(obs)
        return obs, info

    def _terminal_reward(self, obs, terminated) -> float:
        if not terminated:
            return 0.0
        u = self.env.unwrapped
        if u.game_over or abs(obs[0]) >= 1.0:
            return -100.0
        if not u.lander.awake:
            return 100.0
        return 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.scheme == "default":
            return obs, reward, terminated, truncated, info

        fuel = fuel_cost(self.env.unwrapped, action)
        r = self._terminal_reward(obs, terminated) - fuel

        if self.scheme == "pbrs":
            # Absorbing terminal state has potential 0 (required for policy invariance).
            phi_next = 0.0 if terminated else lander_potential(obs)
            r += self.gamma * phi_next - self._phi
            self._phi = phi_next
        elif self.scheme == "naive":
            x, y, vx, vy, ang = obs[:5]
            closeness = max(0.0, 1.0 - math.hypot(x, y))
            calm = max(0.0, 1.0 - math.hypot(vx, vy))
            upright = max(0.0, 1.0 - abs(ang))
            r += 1.0 * closeness * calm * upright  # paid every step you stay "nice"

        return obs, float(r), terminated, truncated, info


# --------------------------------------------------------------------------- robustness
@dataclass
class RandomizationSpec:
    """Per-episode physics randomization ranges (uniform). None = keep env default."""

    gravity: tuple[float, float] | None = None          # env allows (-12, 0)
    wind_power: tuple[float, float] | None = None       # env recommends (0, 20)
    turbulence_power: tuple[float, float] | None = None  # env recommends (0, 2)

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.gravity, self.wind_power, self.turbulence_power))


class DomainRandomization(gym.Wrapper):
    """Samples new gravity / wind / turbulence at every reset.

    LunarLander reads `gravity` when it builds the Box2D world in reset() and reads
    `wind_power` / `turbulence_power` every step, so setting them before reset is enough.
    """

    def __init__(self, env: gym.Env, spec: RandomizationSpec, seed: int | None = None):
        super().__init__(env)
        self.rand_spec = spec
        self.rng = np.random.default_rng(seed)
        self.current: dict[str, float] = {}

    def reset(self, **kwargs):
        u = self.env.unwrapped
        s = self.rand_spec
        self.current = {}
        if s.gravity is not None:
            u.gravity = float(self.rng.uniform(*s.gravity))
            self.current["gravity"] = u.gravity
        if s.wind_power is not None or s.turbulence_power is not None:
            u.enable_wind = True
            if s.wind_power is not None:
                u.wind_power = float(self.rng.uniform(*s.wind_power))
                self.current["wind_power"] = u.wind_power
            if s.turbulence_power is not None:
                u.turbulence_power = float(self.rng.uniform(*s.turbulence_power))
                self.current["turbulence_power"] = u.turbulence_power
        obs, info = self.env.reset(**kwargs)
        info["physics"] = dict(self.current)
        return obs, info


# --------------------------------------------------------------------------- factory
@dataclass
class EnvConfig:
    continuous: bool = False
    shaping: str = "default"
    gamma: float = 0.99
    randomization: RandomizationSpec = field(default_factory=RandomizationSpec)
    # fixed physics (used for evaluation stress tests)
    gravity: float = -10.0
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5
    max_episode_steps: int = 1000


def make_env(cfg: EnvConfig, seed: int = 0, render_mode: str | None = None) -> gym.Env:
    env = gym.make(
        ENV_ID,
        continuous=cfg.continuous,
        gravity=cfg.gravity,
        enable_wind=cfg.enable_wind,
        wind_power=cfg.wind_power,
        turbulence_power=cfg.turbulence_power,
        render_mode=render_mode,
        max_episode_steps=cfg.max_episode_steps,
    )
    env = LandingMetrics(env)
    if cfg.randomization.active:
        env = DomainRandomization(env, cfg.randomization, seed=seed + 10_000)
    env = RewardShaping(env, cfg.shaping, cfg.gamma)
    env.action_space.seed(seed)
    return env


def make_vec_env(cfg: EnvConfig, num_envs: int, seed: int = 0) -> gym.vector.VectorEnv:
    fns = [lambda i=i: make_env(cfg, seed=seed + i) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
