# Sensor-Aware Collision Avoidance Prototype

This repository contains an experimental orbit-estimation and collision-avoidance
pipeline built around OrbitZoo/Orekit. It is a research prototype, not flight
software, and it is **not ready for operational use**. A bounded, experimental
RL training environment is now included; it is intended for research and
validation, not autonomous flight operations.

## Current pipeline

- `orbital_ekf.py` supports truth-informed initialization and a sensor-only
  initialization fitted from relative-position measurements.
- `orbital_collision_forecast.py` propagates the estimate and covariance using a
  two-body model and reports closest-approach and uncertainty-screening values.
- `run_ekf_monte_carlo.py` runs paired seeds and writes per-run artifacts plus
  JSON/CSV summaries.
- `avoidance_controller.py` provides a linear impulse planner, a two-body
  nonlinear miss-distance search, and conversion from RSW impulse to thrust.
- `run_avoidance_baseline.py` compares no maneuver with an Orekit maneuver.
  Ground truth is used only by the simulator-side evaluation, not by the
  sensor-only estimator or maneuver planner.
- `orbit_zoo/src/orbitzoo/scripts/env_col_avoidance.py` provides a
  single-spacecraft, sensor-limited RL experiment with seeded randomized
  encounters, an EKF-style relative-state filter, bounded impulses, and
  no-maneuver/conventional-controller evaluation.

## Reproduce the estimation comparison

Use Python 3.10 or 3.11, as required by the OrbitZoo package metadata. From the
repository root in PowerShell:

```powershell
python -m pip install -e ".\orbit_zoo[test]"
python run_ekf_monte_carlo.py --seeds 7 21 42 99 123 --modes truth-informed sensor-only --warmup-seconds 400 --evaluation-start-seconds 400 --output-dir experiments\ekf_monte_carlo_<new-name>
```

The output directory must be new; the runner refuses to overwrite an existing
directory. The sensor-only run uses a 400 s acquisition interval, 20 m
per-axis relative-position noise, and 10 s measurement spacing. It evaluates
the remaining 200 s of the 600 s scenario.

The matched five-seed results in
`experiments/ekf_monte_carlo_20261027T132705Z/summary.json` are:

| Initialization | TCA MAE (mean ± SD) | Miss-distance MAE (mean ± SD) | Position RMSE | Velocity RMSE |
|---|---:|---:|---:|---:|
| Truth-informed | 1.200 ± 0.347 s | 2.262 ± 2.724 m | 4.506 m | 0.00243 m/s |
| Sensor-only | 6.895 ± 1.981 s | 11.091 ± 2.319 m | 17.326 m | 0.14565 m/s |

These are averages over five measurement-noise seeds on the **same nominal
encounter geometry**, not five independent orbital scenarios. Sensor-only
mean TCA error is below 10 s, but one seed is exactly 10 s, so not every
individual result meets a strict `< 10 s` requirement.

## Run the conventional avoidance comparison

Use a sensor-only run directory containing `ekf_forecast_history.npz`:

```powershell
python run_avoidance_baseline.py --input-dir experiments\ekf_monte_carlo_20261027T132705Z\sensor_only_seed_42
```

The default requested miss distance is the larger of 100 m and the collision
radius plus three propagated radial standard deviations. This is a screening
heuristic, **not** a collision probability. The output is saved as
`avoidance_results.json`; existing files are never overwritten. The command
returns a non-zero status when the requested target is infeasible or not
achieved, while still saving the evaluation.

For the seed-42 sensor-only case at 400 s, the default target is 595.6 m while
the estimated range at the decision epoch is only 277.4 m. The target therefore
cannot be attained as a minimum separation from that late decision point. The
nonlinear planner marks the plan infeasible and commands the best candidate it
finds along the estimated miss-direction search, up to the 5 m/s limit. Orekit
measured a 281.0 m minimum miss distance with that maneuver versus 76.7 m with
no maneuver. This is an improvement, but it does not meet the requested target.

The optional `--sigma-multiplier 0` selects only the configured minimum
distance, useful for lower-target experiments; it does not remove estimation
error. In the same seed-42 case, the two-body planner predicted a 100 m target
with a 0.098 m/s impulse, while Orekit measured 92.7 m. The difference
illustrates why current results do not establish robust maneuver performance.

## RL collision-avoidance experiment

The RL example is deliberately a **single-agent** experiment, aligned with the
single-spacecraft EKF and avoidance baseline above. It samples a new encounter
geometry per seeded episode, synthesizes noisy relative-position measurements
(20 m standard deviation per axis, every 10 s), and gives the policy only a
filtered relative state, predicted closest-approach summaries, uncertainty,
and fuel fraction. The filter also accounts for known self-maneuvers when
propagating relative velocity. Simulator truth is used to generate those
measurements and calculate rewards/evaluation metrics, but is not included in
policy observations or the conventional planner input.

The reward combines change in the filtered predicted miss-distance potential,
a small control/time cost, and a terminal reward based on the simulator's
minimum separation. The collision threshold is 15 m and the requested target
miss distance is 100 m; neither threshold is a formal collision-probability
claim. A policy action is limited to 0.05 m/s per 10-second decision, or at most
3 m/s of accumulated impulse magnitude over the default 600-second episode.
The training loop uses a fast NumPy RK4 two-body simulator with 2-second
integration substeps and 10-second decisions; it does not start Orekit for each
episode. This makes repeated scenario sampling practical, but introduces a
model-fidelity gap: the current policy evaluation and baselines use the same
two-body training simulator. Validate candidate policies against the existing
Orekit-based simulator before drawing conclusions about higher-fidelity
behavior.

The script no longer starts training when imported. From the repository root,
install the package and run the short training smoke test first:

```powershell
python -m pip install -e ".\orbit_zoo[test]"
python -m orbitzoo.scripts.env_col_avoidance --smoke-test
```

Start a training run explicitly (default: 1,000 episodes) and optionally enable
TensorBoard-compatible metrics:

```powershell
python -m orbitzoo.scripts.env_col_avoidance --episodes 1000 --seed 42 --metrics-path rl_collision_avoidance
```

The best policy is saved under
`orbit_zoo\trained_models\collision_avoidance\agent.pt`. Evaluate a saved policy
on held-out seeds against matched no-maneuver and conventional linear-planner
baselines:

```powershell
python -m orbitzoo.scripts.env_col_avoidance --mode evaluate --model-path orbit_zoo\trained_models\collision_avoidance\agent.pt --episodes 100 --seed 10000
```

Evaluation reports the fraction of episodes meeting the 100 m target, the
collision fraction under the 15 m screening radius, mean minimum miss distance,
and mean fuel consumed. Use multiple disjoint held-out seed ranges and compare
against the baselines before interpreting a policy as useful.

## Validation and limitations

Run the package tests with:

```powershell
python -m pytest orbit_zoo\tests -q
```

The package tests cover the orbital estimator, conventional avoidance planner,
RL algorithm smoke cases, fast sensor-limited avoidance environment, and a
one-episode PPO checkpoint smoke test. On Windows, JPype/Orekit may also print
native access-violation diagnostics during test collection even when pytest
exits successfully; investigate that runtime issue before relying on long-running
or parallel Orekit workloads.

Material limitations:

- The sensor models are simplified Gaussian relative-position noise, not
  validated radar or optical sensor models. The RL example's constant-velocity
  relative-state filter is intentionally approximate.
- Estimation and maneuver prediction use two-body dynamics; force-model
  mismatch, realistic navigation errors, and varied encounter geometries have
  not been characterized sufficiently for operational use. The RL scenario
  randomizes encounter parameters but does not model a validated population of
  orbital traffic.
- The risk indicator and three-sigma target are screening heuristics. The
  propagated covariance and reported radial sigma have not been calibrated as
  formal collision probabilities.
- The nonlinear maneuver search is one-dimensional along a miss-direction
  impulse, not a global control optimizer. Fuel, actuator, maneuver timing, and
  multi-burn constraints are not comprehensively modeled.
- The RL environment is a research baseline, not a validated training-ready
  operational system. Reward weights, scenario distributions, filter
  calibration, multi-agent behavior, robustness to model mismatch, and
  large-scale training performance still need systematic study. A passing
  smoke test is not evidence that the policy has learned safe avoidance.
