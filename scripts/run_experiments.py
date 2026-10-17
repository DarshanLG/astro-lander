"""Run the full experiment matrix, a few jobs in parallel. Works on Windows, macOS and Linux.

    python scripts/run_experiments.py                 # everything, 3 seeds
    python scripts/run_experiments.py --only shaping  # one group
    python scripts/run_experiments.py --jobs 4 --seeds 1 2

Finished runs are skipped, so you can stop and restart at any time.
Groups:
    impl      from-scratch PPO vs Stable-Baselines3 (default reward)
    shaping   reward-shaping ablation: default / pbrs / pbrs_carry / pbrs_freeze / naive / sparse
    robust    domain-randomized training (2M steps)
    cont      continuous-action agent (2M steps)
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def jobs_for(group, seed):
    t = [PY, "scripts/train.py", "--seed", str(seed)]
    if group == "impl":
        return [(f"runs/disc_default_s{seed}/final.pt", t + ["--shaping", "default"]),
                (f"runs/sb3_disc_default_s{seed}/eval_summary.json", [PY, "scripts/sb3_baseline.py", "--seed", str(seed)])]
    if group == "shaping":
        return [(f"runs/disc_{s}_s{seed}/final.pt", t + ["--shaping", s]) for s in ("default", "pbrs", "pbrs_carry", "pbrs_freeze", "naive", "sparse")]
    if group == "robust":
        return [(f"runs/disc_default_rand_s{seed}/final.pt", t + ["--randomize", "--total-timesteps", "2000000"])]
    if group == "cont":
        return [(f"runs/cont_default_s{seed}/final.pt", t + ["--config", "configs/continuous.json"])]
    raise ValueError(group)


def run(job):
    done_file, cmd = job
    if (ROOT / done_file).exists():
        return f"skip   {' '.join(cmd[1:])}"
    log = ROOT / "runs" / "logs" / (Path(done_file).parent.name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "OMP_NUM_THREADS": "1"})
    status = "done  " if rc == 0 else "FAILED"
    return f"{status} {' '.join(cmd[1:])}  ({(time.time() - t0) / 60:.1f} min, log: {log.relative_to(ROOT)})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--only", choices=["impl", "shaping", "robust", "cont"], nargs="+",
                    default=["impl", "shaping", "robust", "cont"])
    a = ap.parse_args()
    jobs, seen = [], set()
    for g in a.only:
        for s in a.seeds:
            for j in jobs_for(g, s):
                if j[0] not in seen:
                    seen.add(j[0]); jobs.append(j)
    print(f"{len(jobs)} jobs, {a.jobs} in parallel. Logs in runs/logs/. Ctrl+C to stop; finished runs are kept.")
    with ThreadPoolExecutor(a.jobs) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)
