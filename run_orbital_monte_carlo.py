
import os
import re
import sys
import subprocess
import statistics
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
EKF_SCRIPT = PROJECT_DIR / "orbital_ekf.py"
SEEDS = range(101, 111)  # 10 independent noise realizations

METRIC_PATTERNS = {
    "position_rmse": r"3D position RMSE\s*:\s*([\d.eE+-]+)\s*m",
    "velocity_rmse": r"3D velocity RMSE\s*:\s*([\d.eE+-]+)\s*m/s",
    "mean_nis": r"Mean NIS\s*:\s*([\d.eE+-]+)",
    "coverage_1sigma": r"Coordinate 1-sigma coverage\s*:\s*([\d.]+)%",
    "coverage_2sigma": r"Coordinate 2-sigma coverage\s*:\s*([\d.]+)%",
}

results = []

# ============================================================
# RUN EXPERIMENTS
# ============================================================
print("=" * 76)
print("       ORBITAL EKF — MONTE CARLO VALIDATION")
print("=" * 76)
print(f"EKF script : {EKF_SCRIPT.name}")
print(f"Seeds      : {SEEDS.start} to {SEEDS.stop - 1}")
print(f"Runs       : {len(SEEDS)}")
print("-" * 76)

if not EKF_SCRIPT.exists():
    raise FileNotFoundError(
        f"Cannot find {EKF_SCRIPT}. Keep both Python files in the same folder."
    )

for seed in SEEDS:
    env = os.environ.copy()
    env["ORBITAL_EKF_SEED"] = str(seed)

    try:
        completed = subprocess.run(
            [sys.executable, str(EKF_SCRIPT)],
            cwd=str(PROJECT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        print(f"Seed {seed}: FAILED — run exceeded 180 seconds")
        continue

    if completed.returncode != 0:
        print(f"Seed {seed}: FAILED — EKF exited with an error")
        print(completed.stderr[-1500:])
        continue

    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, completed.stdout)
        if not match:
            break
        metrics[name] = float(match.group(1))

    if len(metrics) != len(METRIC_PATTERNS):
        print(f"Seed {seed}: FAILED — could not parse all metrics")
        print(completed.stdout[-1500:])
        continue

    metrics["seed"] = seed
    results.append(metrics)

    print(
        f"Seed {seed:3d} | "
        f"Position RMSE: {metrics['position_rmse']:7.3f} m | "
        f"Velocity RMSE: {metrics['velocity_rmse']:.6f} m/s | "
        f"NIS: {metrics['mean_nis']:.3f}"
    )

# ============================================================
# AGGREGATE RESULTS
# ============================================================
print("\n" + "=" * 76)
print("                    AGGREGATED RESULTS")
print("=" * 76)

if not results:
    print("No successful runs. Check the errors above.")
    sys.exit(1)

def summarize(label, key, unit="", target=None):
    values = [r[key] for r in results]
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0

    print(f"{label:<27}: {mean:.4f} {unit}")
    print(f"{'  Run-to-run std. dev.':<27}: {std:.4f} {unit}")
    print(f"{'  Minimum / maximum':<27}: "
          f"{min(values):.4f} / {max(values):.4f} {unit}")

    if target is not None:
        print(f"{'  Reference target':<27}: {target} {unit}")
    print()

summarize("Position RMSE", "position_rmse", "m")
summarize("Velocity RMSE", "velocity_rmse", "m/s")
summarize("Mean NIS", "mean_nis", "", target="approximately 3")
summarize("Coordinate 1-sigma coverage", "coverage_1sigma", "%",
          target="approximately 68%")
summarize("Coordinate 2-sigma coverage", "coverage_2sigma", "%",
          target="approximately 95%")

print("-" * 76)
print(f"Successful runs: {len(results)} / {len(SEEDS)}")
print("Note: coverage references assume calibrated Gaussian errors.")
print("      These runs test one scenario, not all orbital conditions.")
print("=" * 76)
