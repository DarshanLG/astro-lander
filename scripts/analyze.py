"""Evaluate every trained run on held-out seeds and produce the README figures/tables.

    python scripts/analyze.py            # evaluate + plot everything found in runs/
    python scripts/analyze.py --no-eval  # re-plot from cached eval files
Outputs go to results/.
"""
import argparse, glob, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astro_lander.envs import EnvConfig
from astro_lander.evaluate import render_gif, run_episodes, stress_test, summarize
from astro_lander.ppo import Policy

RUNS, OUT = Path("runs"), Path("results")
OUT.mkdir(exist_ok=True)

# Validated categorical slots (light surface) + text tokens
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SCHEME_ORDER = ["default", "pbrs_freeze", "pbrs_carry", "pbrs", "naive", "sparse"]
SCHEME_LABEL = {"default": "Gymnasium default", "pbrs": "PBRS, Φ(end)=0 (textbook)", "pbrs_carry": "PBRS, Φ(end) from crash state", "pbrs_freeze": "PBRS, final step unshaped", "naive": "Naive state bonus", "sparse": "Sparse (terminal only)"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})

N_EVAL = 100


def parse(name):
    m = re.match(r"(sb3_)?(disc|cont)_(\w+?)(_rand)?_s(\d+)$", name)
    if not m:
        return None
    return dict(name=name, impl="sb3" if m.group(1) else "scratch", space=m.group(2), shaping=m.group(3),
                randomized=bool(m.group(4)), seed=int(m.group(5)))


def evaluate_all(do_eval=True):
    rows = []
    for d in sorted(RUNS.iterdir()):
        info = parse(d.name)
        if not info:
            continue
        cache = d / "eval_summary.json"
        if info["impl"] == "scratch":
            if not (d / "final.pt").exists():
                continue
            if do_eval and not cache.exists():
                pol = Policy.load(d / "final.pt")
                df = run_episodes(pol, EnvConfig(continuous=pol.cfg.env.continuous), N_EVAL)
                df.to_csv(d / "eval_episodes.csv", index=False)
                cache.write_text(json.dumps(summarize(df), indent=2))
                print("evaluated", d.name, flush=True)
        if cache.exists():
            rows.append(info | json.loads(cache.read_text()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eval_all_runs.csv", index=False)
    return df


def curve(episodes: pd.DataFrame, col: str, step_col="global_step", bins=np.arange(0, 2_000_001, 25_000), window=100):
    e = episodes.sort_values(step_col)
    roll = e[col].astype(float).rolling(window, min_periods=20).mean()
    idx = np.searchsorted(e[step_col].values, bins[1:], side="right") - 1
    vals = np.where(idx >= 0, roll.values[np.clip(idx, 0, None)], np.nan)
    vals[bins[1:] > e[step_col].max()] = np.nan
    return bins[1:], vals


def load_episodes(name):
    d = RUNS / name
    if (d / "episodes.csv").exists():
        e = pd.read_csv(d / "episodes.csv")
        e["timeout"] = (e.outcome == "timeout").astype(float)
        return e
    if (d / "monitor.csv").exists():  # SB3
        m = pd.read_csv(d / "monitor.csv", skiprows=1)
        m["global_step"] = m.l.cumsum()
        return m.rename(columns={"r": "true_return"})
    return None


def band(ax, names, col, color, label, scale=1.0):
    curves = []
    for n in names:
        e = load_episodes(n)
        if e is not None and col in e:
            x, y = curve(e, col)
            curves.append(y * scale)
    if not curves:
        return None
    Y = np.vstack(curves)
    ok = ~np.all(np.isnan(Y), axis=0)
    x, Y = x[ok], Y[:, ok]
    mean = np.nanmean(Y, 0)
    ax.fill_between(x, np.nanmin(Y, 0), np.nanmax(Y, 0), color=color, alpha=0.14, linewidth=0)
    ax.plot(x, mean, color=color, label=f"{label} (n={len(curves)})")
    return x, mean


def fmt_steps(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v else "0"))


def fig_implementation(ev):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (impl, label) in enumerate([("scratch", "From-scratch PPO (this repo)"), ("sb3", "Stable-Baselines3 PPO")]):
        names = ev[(ev.impl == impl) & (ev.space == "disc") & (ev.shaping == "default") & ~ev.randomized].name
        band(ax, names, "true_return", SERIES[i], label)
    ax.axhline(200, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(30_000, 205, "solved (200)", color=INK2, fontsize=9, va="bottom")
    ax.set(title="Same hyperparameters, same environment: learning curves match",
           xlabel="environment steps", ylabel="episode return (100-ep. mean)", ylim=(-400, 300))
    fmt_steps(ax); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_implementation.png", dpi=160); plt.close(fig)


def fig_shaping(ev):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    panels = [("true_return", "True task return", (-650, 320), 1), ("on_pad", "Landed on the pad (%)", (0, 100), 100),
              ("timeout", "Hovered until time-out (%)", (0, 100), 100)]
    for (col, title, ylim, scale), ax in zip(panels, axes):
        for i, s in enumerate(SCHEME_ORDER):
            names = ev[(ev.impl == "scratch") & (ev.space == "disc") & (ev.shaping == s) & ~ev.randomized].name
            band(ax, names, col, SERIES[i], SCHEME_LABEL[s], scale)
        ax.set(title=title, xlabel="environment steps", ylim=ylim); fmt_steps(ax)
    axes[0].axhline(200, color=INK2, lw=1, ls=(0, (4, 3)))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Reward shaping ablation: same agent, six reward functions (mean and min-max over seeds)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(OUT / "fig_reward_shaping.png", dpi=160); plt.close(fig)


def fig_stress(nominal_name, robust_name):
    grids = {}
    for label, name in [("Calm-trained agent", nominal_name), ("Domain-randomized agent", robust_name)]:
        cache = RUNS / name / "stress.csv"
        if not cache.exists():
            pol = Policy.load(RUNS / name / "final.pt")
            stress_test(pol, EnvConfig(), n_episodes=40).to_csv(cache, index=False)
            print("stress-tested", name, flush=True)
        grids[label] = pd.read_csv(cache)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#9cc1ee", "#2a78d6", "#0e3d75"])
    for ax, (label, g) in zip(axes, grids.items()):
        piv = g.pivot(index="gravity", columns="wind_power", values="success_rate").sort_index(ascending=False) * 100
        ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(piv.columns)), [("off" if w == 0 else f"{w:g}") for w in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{gv:g}" for gv in piv.index])
        ax.set(xlabel="wind power (+ turbulence 1.0 when on)", ylabel="gravity (m/s²)", title=label)
        ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9.5, color=INK if v < 55 else "#ffffff")
        ax.add_patch(matplotlib.patches.Rectangle((-0.5, 1.5), 1, 1, fill=False, ec=INK, lw=1.5))  # training condition
    fig.suptitle("On-pad landing rate under unseen conditions (40 held-out episodes per cell; box = default env)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_robustness.png", dpi=160); plt.close(fig)
    return grids


def table(ev):
    g = ev.groupby(["impl", "space", "shaping", "randomized"])
    cols = ["mean_return", "solved_rate", "success_rate", "crash_rate", "timeout_rate", "mean_fuel"]
    agg = g[cols].agg(["mean", "std"])
    agg["n_seeds"] = g.size()
    agg.to_csv(OUT / "summary_table.csv")
    return agg


def pick_best(ev, mask):
    sub = ev[mask]
    return sub.sort_values("mean_return", ascending=False).name.iloc[0] if len(sub) else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-eval", action="store_true"); a = ap.parse_args()
    ev = evaluate_all(not a.no_eval)
    print(ev[["name", "mean_return", "solved_rate", "success_rate", "crash_rate", "timeout_rate"]].to_string())
    fig_implementation(ev)
    fig_shaping(ev)
    base = (ev.impl == "scratch") & (ev.space == "disc") & (ev.shaping == "default")
    nominal, robust = pick_best(ev, base & ~ev.randomized), pick_best(ev, base & ev.randomized)
    if nominal and robust:
        fig_stress(nominal, robust)
        (OUT / "showcase.json").write_text(json.dumps({"nominal": nominal, "robust": robust}))
        for tag, name, cfg in [("calm", nominal, EnvConfig()),
                               ("storm_nominal", nominal, EnvConfig(enable_wind=True, wind_power=18, turbulence_power=1.5)),
                               ("storm_robust", robust, EnvConfig(enable_wind=True, wind_power=18, turbulence_power=1.5))]:
            if not (OUT / f"landing_{tag}.gif").exists():
                print(tag, render_gif(Policy.load(RUNS / name / "final.pt"), cfg, OUT / f"landing_{tag}.gif", seed=1_000_003))
    print(table(ev).round(3).to_string())
