"""Reference baseline: Stable-Baselines3 PPO with the same hyperparameters as our PPO.

Used only to validate the from-scratch implementation, evaluated with the *same*
harness (astro_lander.evaluate) on the same held-out seeds.
    python scripts/sb3_baseline.py --seed 1
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from astro_lander.envs import EnvConfig, make_env
from astro_lander.evaluate import run_episodes, summarize
from astro_lander.ppo import PPOConfig

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=1)
p.add_argument("--continuous", action="store_true")
p.add_argument("--out", default="runs")
a = p.parse_args()

c = PPOConfig()
env_cfg = EnvConfig(continuous=a.continuous)
name = f"sb3_{'cont' if a.continuous else 'disc'}_default_s{a.seed}"
run_dir = Path(a.out) / name
run_dir.mkdir(parents=True, exist_ok=True)

venv = VecNormalize(VecMonitor(DummyVecEnv([lambda i=i: make_env(env_cfg, seed=a.seed * 1000 + i) for i in range(c.num_envs)]),
                                filename=str(run_dir / "monitor.csv")),
                    gamma=c.gamma)
model = PPO("MlpPolicy", venv, n_steps=c.num_steps, batch_size=c.minibatch_size, n_epochs=c.update_epochs,
            gamma=c.gamma, gae_lambda=c.gae_lambda, ent_coef=c.ent_coef, vf_coef=c.vf_coef,
            learning_rate=lambda f: c.learning_rate * f, clip_range=c.clip_coef, max_grad_norm=c.max_grad_norm,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
            seed=a.seed, device="cpu", verbose=0)
t0 = time.time()
model.learn(c.total_timesteps)
train_time = time.time() - t0
model.save(run_dir / "model")
venv.save(str(run_dir / "vecnormalize.pkl"))

rms = venv.obs_rms
def policy(obs):
    o = np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10)
    return model.predict(o[None], deterministic=True)[0][0]

df = run_episodes(policy, env_cfg, 100)
df.to_csv(run_dir / "eval_episodes.csv", index=False)
s = summarize(df) | {"train_time_s": train_time}
(run_dir / "eval_summary.json").write_text(json.dumps(s, indent=2))
print(name, json.dumps(s, indent=2))
