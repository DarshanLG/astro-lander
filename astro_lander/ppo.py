"""Proximal Policy Optimization, implemented from scratch in PyTorch.

Single-file, readable implementation of PPO-Clip (Schulman et al., 2017) with the
details that actually matter in practice (see "The 37 Implementation Details of PPO",
Huang et al., 2022):

* vectorized rollouts with correct bootstrapping on time-limit truncation
* Generalized Advantage Estimation (GAE-lambda)
* clipped surrogate objective + clipped value loss + entropy bonus
* per-minibatch advantage normalization, global gradient clipping
* running observation normalization and return-based reward scaling
* orthogonal initialization, linear learning-rate annealing
* discrete (Categorical) and continuous (diagonal Gaussian) action spaces
"""
from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .envs import EnvConfig, RandomizationSpec, make_vec_env


# --------------------------------------------------------------------------- config
@dataclass
class PPOConfig:
    run_name: str = "ppo"
    seed: int = 1
    total_timesteps: int = 1_000_000
    num_envs: int = 16
    num_steps: int = 1024            # rollout length per env -> batch = num_envs * num_steps
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.999
    gae_lambda: float = 0.98
    num_minibatches: int = 256       # minibatch of 64; fewer, larger minibatches leave the agent hovering
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden_sizes: tuple[int, ...] = (64, 64)
    norm_obs: bool = True
    norm_reward: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    log_every_updates: int = 1
    checkpoint_every_updates: int = 10

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        d = dict(d)
        env = dict(d.pop("env", {}))
        rnd = env.pop("randomization", {}) or {}
        rnd = {k: (tuple(v) if v is not None else None) for k, v in rnd.items()}
        env_cfg = EnvConfig(**env, randomization=RandomizationSpec(**rnd))
        if "hidden_sizes" in d:
            d["hidden_sizes"] = tuple(d["hidden_sizes"])
        return cls(**d, env=env_cfg)


# --------------------------------------------------------------------------- normalizers
class RunningMeanStd:
    """Welford-style parallel running mean / variance (Chan et al.)."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        b_mean, b_var, b_count = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, d):
        self.mean = np.asarray(d["mean"], np.float64)
        self.var = np.asarray(d["var"], np.float64)
        self.count = d["count"]


class ObsNormalizer:
    def __init__(self, shape, clip: float = 10.0, enabled: bool = True):
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.enabled = enabled

    def __call__(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        if not self.enabled:
            return obs.astype(np.float32)
        if update:
            self.rms.update(obs)
        return np.clip((obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip).astype(np.float32)


class RewardScaler:
    """Scales rewards by the running std of the discounted return (as in VecNormalize)."""

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, enabled: bool = True):
        self.rms = RunningMeanStd(())
        self.ret = np.zeros(num_envs)
        self.gamma, self.clip, self.enabled = gamma, clip, enabled

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return reward
        self.ret = self.ret * self.gamma + reward
        self.rms.update(self.ret)
        self.ret[done] = 0.0
        return np.clip(reward / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip)


# --------------------------------------------------------------------------- network
def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_std: float) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(d, h)), nn.Tanh()]
        d = h
    layers.append(layer_init(nn.Linear(d, out_dim), std=out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, continuous: bool, hidden=(64, 64)):
        super().__init__()
        self.continuous = continuous
        self.critic = mlp(obs_dim, hidden, 1, out_std=1.0)
        self.actor = mlp(obs_dim, hidden, act_dim, out_std=0.01)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def dist(self, obs: torch.Tensor):
        out = self.actor(obs)
        if self.continuous:
            return Normal(out, self.log_std.expand_as(out).exp())
        return Categorical(logits=out)

    def act(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        d = self.dist(obs)
        if action is None:
            action = d.sample()
        logp = d.log_prob(action)
        ent = d.entropy()
        if self.continuous:
            logp, ent = logp.sum(-1), ent.sum(-1)
        return action, logp, ent, self.value(obs)

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.actor(obs)
        return out.clamp(-1, 1) if self.continuous else out.argmax(-1)


# --------------------------------------------------------------------------- policy wrapper
class Policy:
    """Trained agent + its observation normalizer, ready for evaluation / export."""

    def __init__(self, net: ActorCritic, obs_norm: ObsNormalizer, cfg: PPOConfig):
        self.net, self.obs_norm, self.cfg = net.eval(), obs_norm, cfg

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, deterministic: bool = True):
        x = torch.as_tensor(self.obs_norm(np.asarray(obs, np.float64)[None]))
        if deterministic:
            a = self.net.deterministic(x)[0]
        else:
            a = self.net.act(x)[0][0]
        return a.numpy() if self.net.continuous else int(a)

    def save(self, path: str | Path):
        torch.save(
            {"net": self.net.state_dict(), "obs_rms": self.obs_norm.rms.state_dict(),
             "obs_norm_enabled": self.obs_norm.enabled, "config": self.cfg.to_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PPOConfig.from_dict(ck["config"])
        obs_dim, act_dim = 8, (2 if cfg.env.continuous else 4)
        net = ActorCritic(obs_dim, act_dim, cfg.env.continuous, cfg.hidden_sizes)
        net.load_state_dict(ck["net"])
        norm = ObsNormalizer((obs_dim,), enabled=ck["obs_norm_enabled"])
        norm.rms.load_state_dict(ck["obs_rms"])
        return cls(net, norm, cfg)


# --------------------------------------------------------------------------- training
def train(cfg: PPOConfig, out_dir: str | Path = "runs", verbose: bool = True) -> Path:
    run_dir = Path(out_dir) / cfg.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)

    cfg.env.gamma = cfg.gamma  # PBRS must use the agent's discount
    envs = make_vec_env(cfg.env, cfg.num_envs, seed=cfg.seed * 1000)
    obs_dim = envs.single_observation_space.shape[0]
    continuous = cfg.env.continuous
    act_dim = envs.single_action_space.shape[0] if continuous else envs.single_action_space.n

    net = ActorCritic(obs_dim, act_dim, continuous, cfg.hidden_sizes)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=1e-5)
    obs_norm = ObsNormalizer((obs_dim,), enabled=cfg.norm_obs)
    rew_scale = RewardScaler(cfg.num_envs, cfg.gamma, enabled=cfg.norm_reward)

    T, N = cfg.num_steps, cfg.num_envs
    b_obs = torch.zeros((T, N, obs_dim))
    b_act = torch.zeros((T, N, act_dim)) if continuous else torch.zeros((T, N), dtype=torch.long)
    b_logp, b_rew, b_done, b_val = (torch.zeros((T, N)) for _ in range(4))

    raw_obs, _ = envs.reset(seed=cfg.seed * 1000)
    next_obs = torch.as_tensor(obs_norm(raw_obs, update=True))
    next_done = torch.zeros(N)
    ep_window: deque = deque(maxlen=100)
    shaped_ret = np.zeros(N)
    shaped_window: deque = deque(maxlen=100)

    num_updates = cfg.total_timesteps // cfg.batch_size
    global_step, t0 = 0, time.time()
    log_f = open(run_dir / "metrics.csv", "w", newline="")
    writer = None
    ep_f = open(run_dir / "episodes.csv", "w", newline="")
    ep_writer = csv.writer(ep_f)
    ep_writer.writerow(["global_step", "true_return", "shaped_return", "length", "outcome", "on_pad", "fuel"])

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            opt.param_groups[0]["lr"] = cfg.learning_rate * (1.0 - (update - 1.0) / num_updates)

        # ---------------- rollout
        for t in range(T):
            global_step += N
            b_obs[t], b_done[t] = next_obs, next_done
            with torch.no_grad():
                action, logp, _, value = net.act(next_obs)
            b_act[t], b_logp[t], b_val[t] = action, logp, value

            env_action = action.numpy()
            raw_obs, reward, term, trunc, info = envs.step(env_action)
            done = np.logical_or(term, trunc)
            shaped_ret += reward

            r = rew_scale(np.asarray(reward, np.float64), done)
            # Bootstrap through time-limit truncation: the episode did not really end,
            # so credit the value of the state we were cut off in (value is in scaled units).
            if trunc.any() and "final_obs" in info:
                for i in np.where(trunc & ~term)[0]:
                    fo = torch.as_tensor(obs_norm(np.asarray(info["final_obs"][i], np.float64)[None]))
                    with torch.no_grad():
                        r[i] += cfg.gamma * net.value(fo).item()
            b_rew[t] = torch.as_tensor(r, dtype=torch.float32)
            next_obs = torch.as_tensor(obs_norm(raw_obs, update=True))
            next_done = torch.as_tensor(done, dtype=torch.float32)

            if done.any():
                fi = info["final_info"]
                for i in np.where(done)[0]:
                    m = {k: fi["episode_metrics"][k][i] for k in fi["episode_metrics"] if not k.startswith("_")}
                    ep_window.append(m)
                    shaped_window.append(shaped_ret[i])
                    ep_writer.writerow([global_step, round(float(m["true_return"]), 3), round(float(shaped_ret[i]), 3),
                                        int(m["length"]), m["outcome"], int(m["on_pad"]), round(float(m["fuel"]), 3)])
                    shaped_ret[i] = 0.0

        # ---------------- GAE
        with torch.no_grad():
            next_value = net.value(next_obs)
            adv = torch.zeros_like(b_rew)
            last = torch.zeros(N)
            for t in reversed(range(T)):
                nonterminal = 1.0 - (next_done if t == T - 1 else b_done[t + 1])
                nv = next_value if t == T - 1 else b_val[t + 1]
                delta = b_rew[t] + cfg.gamma * nv * nonterminal - b_val[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
                adv[t] = last
            ret = adv + b_val

        # ---------------- optimize
        f_obs = b_obs.reshape(-1, obs_dim)
        f_act = b_act.reshape(-1, act_dim) if continuous else b_act.reshape(-1)
        f_logp, f_adv, f_ret, f_val = b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1), b_val.reshape(-1)

        clipfracs, approx_kl = [], torch.tensor(0.0)
        idx = np.arange(cfg.batch_size)
        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = idx[s : s + cfg.minibatch_size]
                _, new_logp, entropy, new_v = net.act(f_obs[mb], f_act[mb])
                log_ratio = new_logp - f_logp[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = f_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()

                if cfg.clip_vloss:
                    v_unclipped = (new_v - f_ret[mb]) ** 2
                    v_clipped = f_val[mb] + (new_v - f_val[mb]).clamp(-cfg.clip_coef, cfg.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - f_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_v - f_ret[mb]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        # ---------------- logging
        y_pred, y_true = f_val.numpy(), f_ret.numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        row = dict(
            update=update,
            global_step=global_step,
            true_return=np.mean([m["true_return"] for m in ep_window]) if ep_window else np.nan,
            shaped_return=np.mean(shaped_window) if shaped_window else np.nan,
            success_rate=np.mean([m["on_pad"] for m in ep_window]) if ep_window else np.nan,
            landed_rate=np.mean([m["outcome"] == "landed" for m in ep_window]) if ep_window else np.nan,
            crash_rate=np.mean([m["outcome"] in ("crashed", "out_of_bounds") for m in ep_window]) if ep_window else np.nan,
            timeout_rate=np.mean([m["outcome"] == "timeout" for m in ep_window]) if ep_window else np.nan,
            ep_length=np.mean([m["length"] for m in ep_window]) if ep_window else np.nan,
            fuel=np.mean([m["fuel"] for m in ep_window]) if ep_window else np.nan,
            pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropy=ent_loss.item(),
            approx_kl=approx_kl.item(), clipfrac=float(np.mean(clipfracs)),
            explained_var=explained_var, lr=opt.param_groups[0]["lr"],
            sps=int(global_step / (time.time() - t0)),
        )
        if writer is None:
            writer = csv.DictWriter(log_f, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        log_f.flush()
        ep_f.flush()
        if verbose and update % cfg.log_every_updates == 0:
            print(f"[{cfg.run_name}] upd {update}/{num_updates} step {global_step:>8d} "
                  f"ret {row['true_return']:8.1f} succ {row['success_rate']:.2f} "
                  f"timeout {row['timeout_rate']:.2f} kl {row['approx_kl']:.4f} sps {row['sps']}", flush=True)
        if update % cfg.checkpoint_every_updates == 0:
            Policy(net, obs_norm, cfg).save(run_dir / "checkpoints" / f"step_{global_step}.pt")
            net.train()

    Policy(net, obs_norm, cfg).save(run_dir / "final.pt")
    log_f.close()
    ep_f.close()
    envs.close()
    return run_dir
