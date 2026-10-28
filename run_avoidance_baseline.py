"""Run no-maneuver and conventional impulse cases in the Orekit simulator."""

import argparse
import json
from pathlib import Path

import numpy as np

from avoidance_controller import (
    plan_nonlinear_avoidance_maneuver,
    thrust_rsw_for_delta_v,
)
from orbital_collision_forecast import predict_orbital_closest_approach
from orbitzoo import OrbitZoo


INITIAL_POSITION_M = np.array([
    -3529923.947865602,
    7042905.715845195,
    0.0,
])
INITIAL_VELOCITY_MPS = np.array([
    6359.116737768876,
    3187.207008809081,
    0.0,
])
RELATIVE_POSITION_M = np.array([0.0, 1000.0, 0.0])
RELATIVE_VELOCITY_MPS = np.array([0.0, -2.0, 0.0])
SPACECRAFT_NAME = "Spacecraft-1"
DT_SECONDS = 2.0
DRY_MASS_KG = 1.0
INITIAL_FUEL_KG = 4.0
ISP_SECONDS = 1000.0


def _new_environment():
    return OrbitZoo(
        dynamics_library="orekit",
        step_size=DT_SECONDS,
        spacecrafts=[{
            "name": SPACECRAFT_NAME,
            "initial_state": [
                *INITIAL_POSITION_M,
                *INITIAL_VELOCITY_MPS,
            ],
            "isp": ISP_SECONDS,
            "dry_mass": DRY_MASS_KG,
            "initial_fuel_mass": INITIAL_FUEL_KG,
        }],
        drifters=[{
            "name": "Debris-1",
            "initial_state": [
                *(INITIAL_POSITION_M + RELATIVE_POSITION_M),
                *(INITIAL_VELOCITY_MPS + RELATIVE_VELOCITY_MPS),
            ],
        }],
        render=False,
    )


def _simulate(epoch_step: int, total_steps: int, thrust_rsw=None) -> dict:
    env = _new_environment()
    spacecraft = env.dynamics.get_body(SPACECRAFT_NAME)
    debris = env.dynamics.get_body("Debris-1")

    relative_positions = []
    fuel_before_burn = None

    for step in range(total_steps + 1):
        if step >= epoch_step:
            relative_positions.append(
                np.asarray(debris.position, dtype=float)
                - np.asarray(spacecraft.position, dtype=float)
            )
        if step == epoch_step:
            fuel_before_burn = float(spacecraft.get_fuel())
        if step < total_steps:
            actions = (
                {SPACECRAFT_NAME: thrust_rsw}
                if step == epoch_step and thrust_rsw is not None
                else None
            )
            env.step(step_size=DT_SECONDS, actions=actions)

    relative_positions = np.asarray(relative_positions)
    distances = np.linalg.norm(relative_positions, axis=1)
    closest_index = int(np.argmin(distances))
    result = {
        "actual_tca_seconds_after_decision": float(
            closest_index * DT_SECONDS
        ),
        "actual_miss_distance_m": float(distances[closest_index]),
        "fuel_before_burn_kg": fuel_before_burn,
        "fuel_after_burn_kg": float(spacecraft.get_fuel()),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing ekf_forecast_history.npz.",
    )
    parser.add_argument("--epoch-seconds", type=float, default=400.0)
    parser.add_argument("--collision-radius-m", type=float, default=10.0)
    parser.add_argument("--minimum-miss-distance-m", type=float, default=100.0)
    parser.add_argument("--sigma-multiplier", type=float, default=3.0)
    parser.add_argument("--max-delta-v-mps", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    history_path = input_dir / "ekf_forecast_history.npz"
    if not history_path.is_file():
        raise FileNotFoundError(f"Forecast history not found: {history_path}")

    with np.load(history_path) as data:
        times = data["times"].copy()
        spacecraft_states = data["spacecraft_states"].copy()
        debris_estimates = data["debris_estimates"].copy()
        debris_covariances = data["debris_covariances"].copy()
        total_time = float(data["total_time"])
        dt = float(data["dt"])
        acceleration_noise_std = (
            float(data["acceleration_noise_std"])
            if "acceleration_noise_std" in data
            else 0.01
        )
        seed = int(data["random_seed"])
        initialization_mode = (
            str(data["initialization_mode"])
            if "initialization_mode" in data
            else "unknown"
        )

    if not np.isfinite(dt) or dt <= 0 or not np.isclose(dt, DT_SECONDS):
        raise ValueError(
            f"Forecast dt must match this simulator's {DT_SECONDS}s step."
        )
    if (
        not np.isfinite(total_time)
        or total_time <= 0
        or not np.isclose(round(total_time / dt) * dt, total_time)
    ):
        raise ValueError("Recorded total_time must be positive and divisible by dt.")
    if not np.isfinite(args.epoch_seconds) or args.epoch_seconds < 0:
        raise ValueError("epoch-seconds must be finite and non-negative.")
    if not np.isfinite(args.collision_radius_m) or args.collision_radius_m <= 0:
        raise ValueError("collision-radius-m must be finite and positive.")
    if (
        not np.isfinite(args.minimum_miss_distance_m)
        or args.minimum_miss_distance_m <= 0
    ):
        raise ValueError("minimum-miss-distance-m must be finite and positive.")
    if not np.isfinite(args.sigma_multiplier) or args.sigma_multiplier < 0:
        raise ValueError("sigma-multiplier must be finite and non-negative.")
    if not np.isfinite(args.max_delta_v_mps) or args.max_delta_v_mps <= 0:
        raise ValueError("max-delta-v-mps must be finite and positive.")
    if not np.isclose(
        round(args.epoch_seconds / dt) * dt,
        args.epoch_seconds,
    ):
        raise ValueError("epoch-seconds must be on the recorded simulation grid.")
    forecast_index = np.flatnonzero(np.isclose(times, args.epoch_seconds))
    if len(forecast_index) != 1:
        raise ValueError(
            f"No unique EKF forecast checkpoint exists at {args.epoch_seconds}s."
        )
    forecast_index = int(forecast_index[0])
    total_steps = int(round(total_time / dt))
    epoch_step = int(round(args.epoch_seconds / dt))
    if epoch_step >= total_steps:
        raise ValueError("The maneuver epoch must precede the simulation end.")

    forecast = predict_orbital_closest_approach(
        spacecraft_state=spacecraft_states[forecast_index],
        debris_estimated_state=debris_estimates[forecast_index],
        debris_covariance=debris_covariances[forecast_index],
        prediction_horizon=total_time - args.epoch_seconds,
        dt=dt,
        acceleration_noise_std=acceleration_noise_std,
    )
    target_miss_distance = max(
        args.minimum_miss_distance_m,
        args.collision_radius_m
        + args.sigma_multiplier * forecast["miss_sigma_m"],
    )
    plan = plan_nonlinear_avoidance_maneuver(
        spacecraft_state=spacecraft_states[forecast_index],
        debris_state=debris_estimates[forecast_index],
        forecast=forecast,
        target_miss_distance_m=target_miss_distance,
        prediction_horizon_s=total_time - args.epoch_seconds,
        propagation_step_s=dt,
        max_delta_v_mps=args.max_delta_v_mps,
    )

    no_maneuver = _simulate(epoch_step, total_steps)
    if np.linalg.norm(plan.delta_v_rsw_mps) > 0:
        spacecraft_mass = DRY_MASS_KG + INITIAL_FUEL_KG
        thrust = thrust_rsw_for_delta_v(
            plan.delta_v_rsw_mps,
            spacecraft_mass_kg=spacecraft_mass,
            specific_impulse_s=ISP_SECONDS,
            burn_duration_s=dt,
        )
        maneuver = _simulate(epoch_step, total_steps, thrust)
    else:
        thrust = np.zeros(3)
        maneuver = _simulate(epoch_step, total_steps)

    achieved_target = (
        maneuver["actual_miss_distance_m"] >= target_miss_distance
    )
    result = {
        "configuration": {
            "seed": seed,
            "initialization_mode": initialization_mode,
            "decision_epoch_seconds": args.epoch_seconds,
            "simulation_end_seconds": total_time,
            "collision_radius_m": args.collision_radius_m,
            "minimum_miss_distance_m": args.minimum_miss_distance_m,
            "sigma_multiplier": args.sigma_multiplier,
            "target_miss_distance_m": target_miss_distance,
            "estimated_range_at_decision_m": float(np.linalg.norm(
                debris_estimates[forecast_index, :3]
                - spacecraft_states[forecast_index, :3]
            )),
            "risk_screening": (
                "target = max(minimum miss distance, collision radius + "
                "sigma multiplier * propagated radial miss sigma); heuristic only"
            ),
            "max_delta_v_mps": args.max_delta_v_mps,
        },
        "forecast_before_maneuver": {
            "tca_seconds": forecast["tca_seconds"],
            "miss_distance_m": forecast["miss_distance_m"],
            "miss_sigma_m": forecast["miss_sigma_m"],
            "risk_indicator": forecast["risk_indicator"],
        },
        "maneuver_plan": {
            "feasible_with_delta_v_limit": plan.feasible,
            "infeasibility_reason": plan.infeasibility_reason,
            "required_delta_v_mps": plan.required_delta_v_mps,
            "commanded_delta_v_mps": plan.commanded_delta_v_mps,
            "planning_model": plan.planning_model,
            "commanded_delta_v_inertial_mps": (
                plan.delta_v_inertial_mps.tolist()
            ),
            "commanded_delta_v_rsw_mps": plan.delta_v_rsw_mps.tolist(),
            "thrust_rsw_newtons": thrust.tolist(),
            "linear_predicted_post_maneuver_miss_m": (
                plan.predicted_miss_after_linear_m
            ),
            "two_body_predicted_post_maneuver_miss_m": (
                plan.predicted_miss_after_model_m
            ),
        },
        "no_maneuver": {
            "actual_tca_seconds_after_decision": (
                no_maneuver["actual_tca_seconds_after_decision"]
            ),
            "actual_miss_distance_m": no_maneuver["actual_miss_distance_m"],
        },
        "maneuver": {
            "actual_tca_seconds_after_decision": (
                maneuver["actual_tca_seconds_after_decision"]
            ),
            "actual_miss_distance_m": maneuver["actual_miss_distance_m"],
            "fuel_consumed_kg": (
                maneuver["fuel_before_burn_kg"]
                - maneuver["fuel_after_burn_kg"]
                if plan.commanded_delta_v_mps > 0
                else 0.0
            ),
            "target_achieved": bool(achieved_target),
            "success": bool(plan.feasible and achieved_target),
        },
    }

    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_dir / "avoidance_results.json"
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nResult saved: {output_path}")
    return 0 if result["maneuver"]["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
