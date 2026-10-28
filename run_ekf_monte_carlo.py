"""Reproducible, isolated multi-seed EKF and collision-forecast evaluation."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parent
EKF_SCRIPT = PROJECT_ROOT / "orbital_ekf.py"
FORECAST_SCRIPT = PROJECT_ROOT / "orbital_collision_forecast.py"
DEFAULT_SEEDS = (7, 21, 42, 99, 123)
METRIC_PATTERNS = {
    "position_rmse_m": re.compile(
        r"3D position RMSE\s*:\s*([0-9.eE+-]+)\s*m"
    ),
    "velocity_rmse_mps": re.compile(
        r"3D velocity RMSE\s*:\s*([0-9.eE+-]+)\s*m/s"
    ),
    "mean_nis": re.compile(r"Mean NIS\s*:\s*([0-9.eE+-]+)"),
    "tca_mae_s": re.compile(
        r"Mean absolute TCA error\s*:\s*([0-9.eE+-]+)\s*s"
    ),
    "miss_mae_m": re.compile(
        r"Mean absolute miss error\s*:\s*([0-9.eE+-]+)\s*m"
    ),
}
def _fresh_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def _run_command(command, cwd: Path, env, log_path: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    log_path.write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"{Path(command[-1]).name} exited with code "
            f"{completed.returncode}; see {log_path}"
        )
    return completed.stdout


def _extract_metrics(ekf_output: str, forecast_output: str) -> dict:
    result = {}
    for key in ("position_rmse_m", "velocity_rmse_mps", "mean_nis"):
        match = METRIC_PATTERNS[key].search(ekf_output)
        if match is None:
            raise RuntimeError(f"Could not parse {key} from EKF report.")
        result[key] = float(match.group(1))

    for key in ("tca_mae_s", "miss_mae_m"):
        match = METRIC_PATTERNS[key].search(forecast_output)
        if match is None:
            raise RuntimeError(f"Could not parse {key} from forecast report.")
        result[key] = float(match.group(1))
    return result


def _summarize(records: list[dict]) -> dict:
    successful = [record for record in records if record["status"] == "ok"]
    summary = {}
    for mode in ("truth-informed", "sensor-only"):
        mode_records = [record for record in successful if record["mode"] == mode]
        if not mode_records:
            continue
        summary[mode] = {}
        for metric in METRIC_PATTERNS:
            values = [record[metric] for record in mode_records]
            mean = sum(values) / len(values)
            standard_deviation = (
                (sum((value - mean) ** 2 for value in values) / (len(values) - 1))
                ** 0.5
                if len(values) > 1
                else None
            )
            half_width = (
                float(student_t.ppf(0.975, len(values) - 1))
                * standard_deviation
                / (len(values) ** 0.5)
                if len(values) > 1
                else None
            )
            summary[mode][metric] = {
                "n": len(values),
                "mean": mean,
                "standard_deviation": standard_deviation,
                "ci95_low": max(0.0, mean - half_width)
                if half_width is not None
                else None,
                "ci95_high": mean + half_width if half_width is not None else None,
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("truth-informed", "sensor-only"),
        default=("truth-informed", "sensor-only"),
    )
    parser.add_argument("--warmup-seconds", type=float, default=400.0)
    parser.add_argument("--evaluation-start-seconds", type=float, default=400.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide one or more distinct random seeds.")
    if args.warmup_seconds < 0 or args.evaluation_start_seconds < 0:
        parser.error("Warmup and evaluation start times must be non-negative.")

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = PROJECT_ROOT / "experiments" / f"ekf_monte_carlo_{stamp}"
    else:
        output_root = args.output_dir.resolve()
    _fresh_output_directory(output_root)

    child_env = os.environ.copy()
    package_src = str(PROJECT_ROOT / "orbit_zoo" / "src")
    current_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        package_src
        if not current_pythonpath
        else package_src + os.pathsep + current_pythonpath
    )

    records = []
    for mode in args.modes:
        for seed in args.seeds:
            case_dir = output_root / f"{mode.replace('-', '_')}_seed_{seed}"
            case_dir.mkdir(parents=True, exist_ok=False)
            env = child_env.copy()
            env["ORBITAL_EKF_INITIALIZATION"] = mode
            env["ORBITAL_EKF_SEED"] = str(seed)
            env["ORBITAL_EKF_OUTPUT_DIR"] = str(case_dir)
            env["ORBITAL_FORECAST_EVALUATION_START_SECONDS"] = str(
                args.evaluation_start_seconds
            )
            env["ORBITAL_EKF_METRIC_START_SECONDS"] = str(
                args.evaluation_start_seconds
            )
            env["ORBITAL_EKF_SENSOR_ONLY_WARMUP_SECONDS"] = str(
                args.warmup_seconds
            )

            started = time.perf_counter()
            record = {
                "mode": mode,
                "seed": seed,
                "status": "failed",
                "output_directory": str(case_dir),
            }
            try:
                ekf_output = _run_command(
                    [sys.executable, str(EKF_SCRIPT)],
                    case_dir,
                    env,
                    case_dir / "ekf.log",
                )
                forecast_output = _run_command(
                    [sys.executable, str(FORECAST_SCRIPT)],
                    case_dir,
                    env,
                    case_dir / "forecast.log",
                )
                record.update(_extract_metrics(ekf_output, forecast_output))
                record["status"] = "ok"
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                record["error"] = str(error)
            record["runtime_seconds"] = time.perf_counter() - started
            records.append(record)
            print(
                f"{mode:14s} seed={seed:>3} "
                f"status={record['status']:6s} "
                f"TCA={record.get('tca_mae_s', float('nan')):8.3f}s "
                f"miss={record.get('miss_mae_m', float('nan')):8.3f}m "
                f"elapsed={record['runtime_seconds']:.1f}s"
            )

    summary = _summarize(records)
    report = {
        "configuration": {
            "seeds": args.seeds,
            "modes": args.modes,
            "sensor_only_warmup_seconds": args.warmup_seconds,
            "common_evaluation_start_seconds": args.evaluation_start_seconds,
            "position_measurement_std_m": 20.0,
            "dt_seconds": 2.0,
            "sensor_interval_seconds": 10.0,
        },
        "records": records,
        "summary": summary,
    }
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    with (output_root / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        columns = [
            "mode",
            "seed",
            "status",
            "position_rmse_m",
            "velocity_rmse_mps",
            "mean_nis",
            "tca_mae_s",
            "miss_mae_m",
            "runtime_seconds",
            "error",
            "output_directory",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nDetailed results: {output_root}")
    for mode, metrics in summary.items():
        print(f"\n{mode} (n={metrics['tca_mae_s']['n']} independent seeds)")
        for key in ("tca_mae_s", "miss_mae_m", "position_rmse_m", "velocity_rmse_mps"):
            item = metrics[key]
            if item["standard_deviation"] is None:
                print(f"  {key:20s} {item['mean']:.4f} (n=1; CI unavailable)")
            else:
                print(
                    f"  {key:20s} {item['mean']:.4f} "
                    f"+/- {item['standard_deviation']:.4f} "
                    f"(95% CI {item['ci95_low']:.4f}..{item['ci95_high']:.4f})"
                )

    return 0 if all(record["status"] == "ok" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
