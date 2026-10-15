"""Export a trained policy (actor MLP + observation normalizer) to JSON for the web demo.
    python scripts/export_policy.py runs/disc_default_s1/final.pt web/policies/nominal.json --label "Nominal"
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from astro_lander.ppo import Policy

p = argparse.ArgumentParser()
p.add_argument("checkpoint"); p.add_argument("out"); p.add_argument("--label", default="")
a = p.parse_args()
pol = Policy.load(a.checkpoint)
linears = [m for m in pol.net.actor if hasattr(m, "weight")]
rms = pol.obs_norm.rms
spec = dict(
    label=a.label, continuous=pol.cfg.env.continuous,
    obs_mean=(rms.mean if pol.obs_norm.enabled else np.zeros(8)).round(6).tolist(),
    obs_var=(rms.var if pol.obs_norm.enabled else np.ones(8) - 1e-8).round(6).tolist(),
    layers=[dict(W=l.weight.detach().numpy().round(6).tolist(), b=l.bias.detach().numpy().round(6).tolist()) for l in linears],
    trained=dict(shaping=pol.cfg.env.shaping, randomized=pol.cfg.env.randomization.active,
                 timesteps=pol.cfg.total_timesteps, seed=pol.cfg.seed),
)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(spec, separators=(",", ":")))
print("wrote", a.out, Path(a.out).stat().st_size // 1024, "KB")
