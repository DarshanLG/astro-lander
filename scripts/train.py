"""Train a PPO agent. Example:
    python scripts/train.py --config configs/discrete_default.json --seed 1
    python scripts/train.py --shaping pbrs --seed 2 --total-timesteps 1000000
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astro_lander.ppo import PPOConfig, train
from astro_lander.envs import RandomizationSpec

p = argparse.ArgumentParser()
p.add_argument("--config", type=str, default=None)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--shaping", choices=["default", "sparse", "pbrs", "naive"], default=None)
p.add_argument("--continuous", action="store_true")
p.add_argument("--randomize", action="store_true", help="train with gravity/wind/turbulence randomization")
p.add_argument("--total-timesteps", type=int, default=None)
p.add_argument("--run-name", type=str, default=None)
p.add_argument("--out", type=str, default="runs")
a = p.parse_args()

cfg = PPOConfig.from_dict(json.loads(Path(a.config).read_text())) if a.config else PPOConfig()
if a.seed is not None: cfg.seed = a.seed
if a.shaping: cfg.env.shaping = a.shaping
if a.continuous: cfg.env.continuous = True
if a.randomize:
    cfg.env.randomization = RandomizationSpec(gravity=(-11.5, -7.0), wind_power=(0.0, 15.0), turbulence_power=(0.0, 1.5))
if a.total_timesteps: cfg.total_timesteps = a.total_timesteps
cfg.run_name = a.run_name or f"{'cont' if cfg.env.continuous else 'disc'}_{cfg.env.shaping}{'_rand' if cfg.env.randomization.active else ''}_s{cfg.seed}"
print("run dir:", train(cfg, a.out))
