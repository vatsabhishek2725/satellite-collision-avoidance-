
# import os
# import numpy as np
# from orbitzoo import OrbitZoo

# # ============================================================
# # 1. CONFIGURATION
# # ============================================================

# MU_EARTH = 3.986004418e14  # m^3/s^2

# DT = 2.0
# TOTAL_TIME = 600.0
# SENSOR_INTERVAL = 10.0

# MEASUREMENT_STD = float(
#     os.environ.get("ORBITAL_EKF_MEASUREMENT_STD", "20.0")
# )

# INITIAL_VELOCITY_STD = 5.0

# # Tunable acceleration-noise parameter; not a measured value.
# ACCELERATION_NOISE_STD = 0.01

# SEED = int(os.environ.get("ORBITAL_EKF_SEED", "42"))
# rng = np.random.default_rng(seed=SEED)

# TOTAL_STEPS = int(round(TOTAL_TIME / DT))
# SENSOR_STEPS = int(round(SENSOR_INTERVAL / DT))

# if DT <= 0 or SENSOR_INTERVAL <= 0:
#     raise ValueError("DT and SENSOR_INTERVAL must be positive.")

# if SENSOR_STEPS < 1:
#     raise ValueError("Sensor interval must be at least one timestep.")

# if not np.isclose(TOTAL_STEPS * DT, TOTAL_TIME):
#     raise ValueError("TOTAL_TIME must be divisible by DT.")

# if not np.isclose(SENSOR_STEPS * DT, SENSOR_INTERVAL):
#     raise ValueError("SENSOR_INTERVAL must be divisible by DT.")

# if MEASUREMENT_STD <= 0:
#     raise ValueError("MEASUREMENT_STD must be positive.")


# # ============================================================
# # 2. INITIALIZE ORBITZOO
# # ============================================================

# r0 = np.array([
#     -3529923.947865602,
#     7042905.715845195,
#     0.0
# ])

# v0 = np.array([
#     6359.116737768876,
#     3187.207008809081,
#     0.0
# ])

# spacecrafts = [{
#     "name": "Spacecraft-1",
#     "initial_state": [*r0, *v0],
#     "isp": 1000,
#     "dry_mass": 1,
#     "initial_fuel_mass": 4
# }]

# drifters = [{
#     "name": "Debris-1",
#     "initial_state": [
#         *(r0 + np.array([0.0, 1000.0, 0.0])),
#         *(v0 - np.array([0.0, 2.0, 0.0]))
#     ]
# }]

# env = OrbitZoo(
#     dynamics_library="orekit",
#     step_size=DT,
#     spacecrafts=spacecrafts,
#     drifters=drifters,
#     render=False
# )

# spacecraft = env.dynamics.get_body("Spacecraft-1")
# debris = env.dynamics.get_body("Debris-1")


# # ============================================================
# # 3. TWO-BODY ORBITAL DYNAMICS
# # ============================================================

# # State x = [x, y, z, vx, vy, vz]
# # Position is in metres; velocity is in m/s.

# def orbital_dynamics(x):
#     r = x[:3]
#     v = x[3:]

#     radius = np.linalg.norm(r)

#     if radius < 1.0:
#         raise ValueError("Invalid orbital radius.")

#     acceleration = -MU_EARTH * r / radius**3

#     return np.concatenate([v, acceleration])


# def rk4_step(x, dt):
#     k1 = orbital_dynamics(x)
#     k2 = orbital_dynamics(x + 0.5 * dt * k1)
#     k3 = orbital_dynamics(x + 0.5 * dt * k2)
#     k4 = orbital_dynamics(x + dt * k3)

#     return x + (dt / 6.0) * (
#         k1 + 2.0 * k2 + 2.0 * k3 + k4
#     )


# def numerical_jacobian(x, dt):
#     # Central finite-difference Jacobian of the discrete
#     # RK4 state propagation function.
#     J = np.zeros((6, 6))

#     for i in range(6):
#         epsilon = 1e-3 if i < 3 else 1e-5

#         delta = np.zeros(6)
#         delta[i] = epsilon

#         x_plus = rk4_step(x + delta, dt)
#         x_minus = rk4_step(x - delta, dt)

#         J[:, i] = (
#             x_plus - x_minus
#         ) / (2.0 * epsilon)

#     return J


# # ============================================================
# # 4. COVARIANCE AND MEASUREMENT MODEL
# # ============================================================

# # Sensor measurement:
# # z_relative = r_debris - r_spacecraft + measurement_noise
# #
# # Convert to an absolute debris-position measurement using
# # the known spacecraft position.

# H = np.zeros((3, 6))
# H[:, :3] = np.eye(3)

# R = np.eye(3) * MEASUREMENT_STD**2


# def process_noise(dt):
#     Q = np.zeros((6, 6))

#     block = ACCELERATION_NOISE_STD**2 * np.array([
#         [dt**4 / 4.0, dt**3 / 2.0],
#         [dt**3 / 2.0, dt**2]
#     ])

#     for axis in range(3):
#         Q[axis, axis] = block[0, 0]
#         Q[axis, axis + 3] = block[0, 1]
#         Q[axis + 3, axis] = block[1, 0]
#         Q[axis + 3, axis + 3] = block[1, 1]

#     return Q


# Q = process_noise(DT)


# # ============================================================
# # 5. GROUND TRUTH AND SENSOR SIMULATION
# # ============================================================

# def get_truth():
#     # Ground truth is reserved for sensor simulation and
#     # offline evaluation. It is not supplied to a future agent.
#     r_sc = np.asarray(spacecraft.position, dtype=float)
#     r_debris = np.asarray(debris.position, dtype=float)
#     v_debris = np.asarray(debris.velocity, dtype=float)

#     return r_sc, np.concatenate([r_debris, v_debris])


# def simulate_relative_measurement(r_sc, debris_state):
#     true_relative_position = debris_state[:3] - r_sc

#     noise = rng.normal(
#         loc=0.0,
#         scale=MEASUREMENT_STD,
#         size=3
#     )

#     return true_relative_position + noise


# # ============================================================
# # 6. INITIALIZE FILTER USING ONE NOISY MEASUREMENT
# # ============================================================

# r_sc, initial_truth = get_truth()

# # This measurement initializes the position estimate.
# # Do not process it a second time at simulation time zero.
# first_relative_measurement = simulate_relative_measurement(
#     r_sc, initial_truth
# )

# x = np.concatenate([
#     r_sc + first_relative_measurement,
#     np.asarray(spacecraft.velocity, dtype=float)
# ])

# P = np.diag([
#     MEASUREMENT_STD**2,
#     MEASUREMENT_STD**2,
#     MEASUREMENT_STD**2,
#     INITIAL_VELOCITY_STD**2,
#     INITIAL_VELOCITY_STD**2,
#     INITIAL_VELOCITY_STD**2
# ])


# # ============================================================
# # 7. EVALUATION LOGS
# # ============================================================

# # These logs are populated at EVERY simulation timestep.
# # They do not control the estimator.

# evaluation_times = []

# position_errors = []
# velocity_errors = []

# position_sigmas = []
# velocity_sigmas = []

# # NIS is recorded only when a new measurement is processed.
# nis_values = []
# measurement_update_times = []

# # ============================================================
# # 7.1 HISTORY FOR ORBITAL COLLISION FORECASTING
# # ============================================================

# # EKF estimates and covariance are saved every sensor interval.
# forecast_times = []
# forecast_spacecraft_states = []
# forecast_debris_estimates = []
# forecast_debris_covariances = []

# # Ground truth is saved separately for offline evaluation only.
# # It is NOT supplied to the collision predictor.
# evaluation_truth_times = []
# evaluation_truth_spacecraft_states = []
# evaluation_truth_debris_states = []
# # ============================================================
# # 8. RUN THE ORBITAL EKF
# # ============================================================

# for step in range(TOTAL_STEPS + 1):

#     current_time = step * DT

#     # --------------------------------------------------------
#     # A. GET CURRENT SIMULATOR STATE
#     # --------------------------------------------------------

#     r_sc, truth = get_truth()

#     # Offline evaluation record only.
#     # This data must not be used by the predictor or future RL agent.
#     evaluation_truth_times.append(current_time)

#     evaluation_truth_spacecraft_states.append(
#     np.concatenate([
#         np.asarray(spacecraft.position, dtype=float),
#         np.asarray(spacecraft.velocity, dtype=float),
#     ])
#     )

#     evaluation_truth_debris_states.append(truth.copy())

    

#     # --------------------------------------------------------
#     # B. SENSOR UPDATE
#     # --------------------------------------------------------
#     # The first sensor measurement was already used to
#     # initialize x and P. The first Kalman update therefore
#     # occurs at SENSOR_INTERVAL, not again at t = 0.

#     if step > 0 and step % SENSOR_STEPS == 0:

#         z_relative = simulate_relative_measurement(
#             r_sc, truth
#         )

#         z_absolute = r_sc + z_relative

#         # Innovation
#         innovation = z_absolute - H @ x

#         # Innovation covariance
#         S = H @ P @ H.T + R

#         # Normalized Innovation Squared
#         nis = float(
#             innovation @ np.linalg.solve(S, innovation)
#         )

#         nis_values.append(nis)
#         measurement_update_times.append(current_time)

#         # Kalman gain
#         K = np.linalg.solve(S, H @ P).T

#         # State update
#         x = x + K @ innovation

#         # Joseph-form covariance update
#         I = np.eye(6)
#         A = I - K @ H

#         P = A @ P @ A.T + K @ R @ K.T

#         # Numerical symmetry protection
#         P = 0.5 * (P + P.T)

#     # --------------------------------------------------------
#     # C. EVALUATE AT EVERY TIMESTEP
#     # --------------------------------------------------------
#     # Evaluate after any measurement update at this time,
#     # but before propagating to the next timestep.

#     position_error = x[:3] - truth[:3]
#     velocity_error = x[3:6] - truth[3:6]

#     position_errors.append(position_error.copy())
#     velocity_errors.append(velocity_error.copy())

#     # Coordinate-wise standard deviations from covariance
#     diagonal_P = np.maximum(np.diag(P), 0.0)

#     sigma_position = np.sqrt(diagonal_P[:3])
#     sigma_velocity = np.sqrt(diagonal_P[3:6])

#     position_sigmas.append(sigma_position.copy())
#     velocity_sigmas.append(sigma_velocity.copy())

#     evaluation_times.append(current_time)


#     # Save the EKF state and covariance at sensor intervals.
# # This is after any measurement update at the current time.

# if step % SENSOR_STEPS == 0:
#     forecast_times.append(current_time)
#     forecast_spacecraft_states.append(
#         np.concatenate([
#             np.asarray(spacecraft.position, dtype=float),
#             np.asarray(spacecraft.velocity, dtype=float),
#         ])
#     )
#     forecast_debris_estimates.append(x.copy())
#     forecast_debris_covariances.append(P.copy())

#     # --------------------------------------------------------
#     # D. PROPAGATE FILTER AND SIMULATOR
#     # --------------------------------------------------------

#     if step < TOTAL_STEPS:

#         F = numerical_jacobian(x, DT)

#         x = rk4_step(x, DT)

#         P = F @ P @ F.T + Q
#         P = 0.5 * (P + P.T)

#         env.step(step_size=DT)


# # ============================================================
# # 9. CALCULATE ALL-TIMESTEP METRICS
# # ============================================================

# evaluation_times = np.asarray(evaluation_times)

# position_errors = np.asarray(position_errors)
# velocity_errors = np.asarray(velocity_errors)

# position_sigmas = np.asarray(position_sigmas)
# velocity_sigmas = np.asarray(velocity_sigmas)

# nis_values = np.asarray(nis_values)

# # Expected shapes:
# # position_errors: (301, 3)
# # velocity_errors: (301, 3)
# # position_sigmas: (301, 3)
# # velocity_sigmas: (301, 3)

# if len(evaluation_times) != TOTAL_STEPS + 1:
#     raise RuntimeError(
#         "Unexpected number of evaluation checkpoints."
#     )

# if len(nis_values) != TOTAL_STEPS // SENSOR_STEPS:
#     raise RuntimeError(
#         "Unexpected number of measurement updates."
#     )

# # 3D RMSE, with one 3D position/velocity error per timestep
# position_rmse = np.sqrt(
#     np.mean(np.sum(position_errors**2, axis=1))
# )

# velocity_rmse = np.sqrt(
#     np.mean(np.sum(velocity_errors**2, axis=1))
# )

# # Coordinate-wise position uncertainty coverage
# coverage_1sigma = 100.0 * np.mean(
#     np.abs(position_errors) <= position_sigmas
# )

# coverage_2sigma = 100.0 * np.mean(
#     np.abs(position_errors) <= 2.0 * position_sigmas
# )

# # Coordinate-wise velocity uncertainty coverage
# velocity_coverage_1sigma = 100.0 * np.mean(
#     np.abs(velocity_errors) <= velocity_sigmas
# )

# velocity_coverage_2sigma = 100.0 * np.mean(
#     np.abs(velocity_errors) <= 2.0 * velocity_sigmas
# )

# mean_nis = (
#     float(np.mean(nis_values))
#     if len(nis_values) > 0
#     else float("nan")
# )

# # ============================================================
# # 9.1 EXPORT EKF OUTPUT FOR COLLISION-RISK PREDICTION
# # ============================================================

# # Export only the spacecraft state and EKF estimate.
# # Ground-truth debris state is intentionally NOT exported.

# spacecraft_final_state = np.concatenate([
#     np.asarray(spacecraft.position, dtype=float),
#     np.asarray(spacecraft.velocity, dtype=float)
# ])

# debris_final_estimated_state = x.copy()
# debris_final_covariance = P.copy()

# np.savez(
#     "ekf_collision_input.npz",
#     spacecraft_state=spacecraft_final_state,
#     debris_estimated_state=debris_final_estimated_state,
#     debris_covariance=debris_final_covariance,
#     simulation_time=float(TOTAL_TIME),
#     measurement_std=float(MEASUREMENT_STD),
#     random_seed=int(SEED)
# )

# print("\n[7] COLLISION-PREDICTOR DATA EXPORT")
# print("-" * 72)
# print("  Output file               : ekf_collision_input.npz")
# print("  Spacecraft state shape    :", spacecraft_final_state.shape)
# print("  Debris estimate shape     :", debris_final_estimated_state.shape)
# print("  EKF covariance shape      :", debris_final_covariance.shape)
# print("  Ground truth exported     : No")


# # ============================================================
# # 8.1 SAVE COLLISION-FORECAST INPUTS AND OFFLINE TRUTH
# # ============================================================

# np.savez(
#     "ekf_forecast_history.npz",
#     times=np.asarray(forecast_times),
#     spacecraft_states=np.asarray(forecast_spacecraft_states),
#     debris_estimates=np.asarray(forecast_debris_estimates),
#     debris_covariances=np.asarray(forecast_debris_covariances),
#     total_time=float(TOTAL_TIME),
#     dt=float(DT),
#     sensor_interval=float(SENSOR_INTERVAL),
#     measurement_std=float(MEASUREMENT_STD),
#     random_seed=int(SEED),
# )

# # Separate file: offline evaluation only.
# np.savez(
#     "collision_forecast_truth.npz",
#     times=np.asarray(evaluation_truth_times),
#     spacecraft_states=np.asarray(
#         evaluation_truth_spacecraft_states
#     ),
#     debris_states=np.asarray(
#         evaluation_truth_debris_states
#     ),
# )

# print("\n[8] ORBITAL FORECAST HISTORY")
# print("-" * 72)
# print("  EKF history file          : ekf_forecast_history.npz")
# print("  Offline truth file        : collision_forecast_truth.npz")
# print("  EKF forecast checkpoints :", len(forecast_times))
# print("  Ground-truth checkpoints :", len(evaluation_truth_times))
# print("  Ground truth used by EKF predictor: No")
# # ============================================================
# # 10. PRINT ORGANIZED REPORT
# # ============================================================

# print("\n" + "=" * 72)
# print("             ORBITAL EXTENDED KALMAN FILTER REPORT")
# print("=" * 72)

# print("\n[1] MODEL CONFIGURATION")
# print("-" * 72)
# print(f"  Orbital model              : Earth two-body gravity")
# print(f"  Numerical integrator       : RK4")
# print(f"  Simulation duration        : {TOTAL_TIME:.1f} s")
# print(f"  Dynamics time step         : {DT:.1f} s")
# print(f"  Sensor interval            : {SENSOR_INTERVAL:.1f} s")
# print(f"  Measurement noise std      : {MEASUREMENT_STD:.1f} m/coordinate")
# print(f"  Random seed                : {SEED}")

# print("\n[2] EVALUATION BOOKKEEPING")
# print("-" * 72)
# print(f"  Evaluation interval        : {DT:.1f} s")
# print(f"  Evaluation checkpoints     : {len(evaluation_times)}")
# print(f"  Initializing measurements  : 1")
# print(f"  Kalman measurement updates : {len(nis_values)}")
# print(f"  NIS samples                : {len(nis_values)}")
# print("  Evaluation includes predictions between measurements.")

# print("\n[3] ESTIMATION ACCURACY — ALL TIMESTEPS")
# print("-" * 72)
# print(f"  3D position RMSE           : {position_rmse:.3f} m")
# print(f"  3D velocity RMSE           : {velocity_rmse:.6f} m/s")

# print("\n[4] UNCERTAINTY VALIDATION")
# print("-" * 72)
# print(f"  Mean NIS                   : {mean_nis:.3f}")
# print("  Reference for 3D measurement: approximately 3.0")
# print(f"  Coordinate 1-sigma coverage : {coverage_1sigma:.1f}%")
# print(f"  Coordinate 2-sigma coverage : {coverage_2sigma:.1f}%")
# print(f"  Velocity 1-sigma coverage   : {velocity_coverage_1sigma:.1f}%")
# print(f"  Velocity 2-sigma coverage   : {velocity_coverage_2sigma:.1f}%")

# print("\n[5] FINAL STATE ESTIMATE")
# print("-" * 72)
# print("  Debris position (m):")
# print(" ", np.array2string(x[:3], precision=3))
# print("  Debris velocity (m/s):")
# print(" ", np.array2string(x[3:6], precision=6))

# print("\n[6] IMPORTANT LIMITATIONS")
# print("-" * 72)
# print("  The filter assumes two-body Earth gravity.")
# print("  OrbitZoo may include dynamics not represented in this model.")
# print("  Process noise is a tunable approximation.")
# print("  Ground truth is used for sensor simulation and offline evaluation.")
# print("  NIS is calculated only for actual Kalman measurement updates.")
# print("  Coverage checkpoints are temporally correlated.")
# print("  This experiment alone does not establish filter consistency.")
# print("=" * 72)

# env.close()



import os
from pathlib import Path
import numpy as np
from orbitzoo import OrbitZoo
from orbital_mechanics import (
    fit_initial_orbit_from_relative_positions,
    numerical_jacobian,
    rk4_step,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

MU_EARTH = 3.986004418e14  # m^3/s^2

DT = 2.0
TOTAL_TIME = 600.0
SENSOR_INTERVAL = 10.0
METRIC_START_SECONDS = float(
    os.environ.get("ORBITAL_EKF_METRIC_START_SECONDS", "0.0")
)
SENSOR_ONLY_WARMUP_SECONDS = float(
    os.environ.get("ORBITAL_EKF_SENSOR_ONLY_WARMUP_SECONDS", "400.0")
)

MEASUREMENT_STD = float(
    os.environ.get("ORBITAL_EKF_MEASUREMENT_STD", "20.0")
)

INITIALIZATION_MODE = os.environ.get(
    "ORBITAL_EKF_INITIALIZATION", "truth-informed"
).strip().lower()
if INITIALIZATION_MODE not in {"truth-informed", "sensor-only"}:
    raise ValueError(
        "ORBITAL_EKF_INITIALIZATION must be 'truth-informed' or 'sensor-only'."
    )

DEFAULT_INITIAL_VELOCITY_STD = (
    "0.0001" if INITIALIZATION_MODE == "truth-informed" else "5.0"
)
INITIAL_VELOCITY_STD = float(
    os.environ.get(
        "ORBITAL_EKF_INITIAL_VELOCITY_STD",
        DEFAULT_INITIAL_VELOCITY_STD,
    )
)

# This experiment uses a deterministic two-body model. Set a nonzero value
# when modeling unrepresented or stochastic accelerations.
ACCELERATION_NOISE_STD = float(
    os.environ.get("ORBITAL_EKF_ACCELERATION_NOISE_STD", "0.0")
)

SEED = int(os.environ.get("ORBITAL_EKF_SEED", "42"))
rng = np.random.default_rng(seed=SEED)

TOTAL_STEPS = int(round(TOTAL_TIME / DT))
SENSOR_STEPS = int(round(SENSOR_INTERVAL / DT))

if DT <= 0 or SENSOR_INTERVAL <= 0:
    raise ValueError("DT and SENSOR_INTERVAL must be positive.")

if SENSOR_STEPS < 1:
    raise ValueError("Sensor interval must be at least one timestep.")

if not np.isclose(TOTAL_STEPS * DT, TOTAL_TIME):
    raise ValueError("TOTAL_TIME must be divisible by DT.")

if not np.isclose(SENSOR_STEPS * DT, SENSOR_INTERVAL):
    raise ValueError("SENSOR_INTERVAL must be divisible by DT.")

if (
    not np.isfinite(METRIC_START_SECONDS)
    or METRIC_START_SECONDS < 0
    or METRIC_START_SECONDS > TOTAL_TIME
    or not np.isclose(
        round(METRIC_START_SECONDS / DT) * DT,
        METRIC_START_SECONDS,
    )
):
    raise ValueError(
        "Metric evaluation start must be finite, within the simulation, "
        "and divisible by DT."
    )

if MEASUREMENT_STD <= 0:
    raise ValueError("MEASUREMENT_STD must be positive.")

if INITIAL_VELOCITY_STD <= 0:
    raise ValueError("INITIAL_VELOCITY_STD must be positive.")

if ACCELERATION_NOISE_STD < 0:
    raise ValueError("ACCELERATION_NOISE_STD cannot be negative.")

if (
    INITIALIZATION_MODE == "sensor-only"
    and (
        SENSOR_ONLY_WARMUP_SECONDS < 2.0 * SENSOR_INTERVAL
        or SENSOR_ONLY_WARMUP_SECONDS > TOTAL_TIME
        or not np.isclose(
            round(SENSOR_ONLY_WARMUP_SECONDS / DT) * DT,
            SENSOR_ONLY_WARMUP_SECONDS,
        )
        or not np.isclose(
            round(SENSOR_ONLY_WARMUP_SECONDS / SENSOR_INTERVAL)
            * SENSOR_INTERVAL,
            SENSOR_ONLY_WARMUP_SECONDS,
        )
    )
):
    raise ValueError(
        "Sensor-only warmup must be at least two sensor intervals, no "
        "longer than the simulation, and divisible by both DT and "
        "SENSOR_INTERVAL."
    )

OUTPUT_DIR = Path(os.environ.get("ORBITAL_EKF_OUTPUT_DIR", ".")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if os.environ.get("ORBITAL_EKF_ALLOW_OVERWRITE") != "1":
    existing_outputs = [
        filename
        for filename in (
            "ekf_collision_input.npz",
            "ekf_forecast_history.npz",
            "collision_forecast_truth.npz",
        )
        if (OUTPUT_DIR / filename).exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing experiment output(s): "
            f"{', '.join(existing_outputs)}. Set ORBITAL_EKF_OUTPUT_DIR to "
            "a new directory, or set ORBITAL_EKF_ALLOW_OVERWRITE=1 explicitly."
        )


# ============================================================
# 2. INITIALIZE ORBITZOO
# ============================================================

r0 = np.array([
    -3529923.947865602,
    7042905.715845195,
    0.0
])

v0 = np.array([
    6359.116737768876,
    3187.207008809081,
    0.0
])

spacecrafts = [{
    "name": "Spacecraft-1",
    "initial_state": [*r0, *v0],
    "isp": 1000,
    "dry_mass": 1,
    "initial_fuel_mass": 4
}]

drifters = [{
    "name": "Debris-1",
    "initial_state": [
        *(r0 + np.array([0.0, 1000.0, 0.0])),
        *(v0 - np.array([0.0, 2.0, 0.0]))
    ]
}]

env = OrbitZoo(
    dynamics_library="orekit",
    step_size=DT,
    spacecrafts=spacecrafts,
    drifters=drifters,
    render=False
)

spacecraft = env.dynamics.get_body("Spacecraft-1")
debris = env.dynamics.get_body("Debris-1")


# ============================================================
# 4. COVARIANCE AND MEASUREMENT MODEL
# ============================================================

# Sensor measurement:
# z_relative = r_debris - r_spacecraft + measurement_noise
#
# Convert to an absolute debris-position measurement using
# the known spacecraft position.

H = np.zeros((3, 6))
H[:, :3] = np.eye(3)

R = np.eye(3) * MEASUREMENT_STD**2


def process_noise(dt):
    Q = np.zeros((6, 6))

    block = ACCELERATION_NOISE_STD**2 * np.array([
        [dt**4 / 4.0, dt**3 / 2.0],
        [dt**3 / 2.0, dt**2]
    ])

    for axis in range(3):
        Q[axis, axis] = block[0, 0]
        Q[axis, axis + 3] = block[0, 1]
        Q[axis + 3, axis] = block[1, 0]
        Q[axis + 3, axis + 3] = block[1, 1]

    return Q


Q = process_noise(DT)


# ============================================================
# 5. GROUND TRUTH AND SENSOR SIMULATION
# ============================================================

def get_truth():
    # Ground truth is reserved for sensor simulation and
    # offline evaluation. It is not supplied to a future agent.

    r_sc = np.asarray(spacecraft.position, dtype=float)
    r_debris = np.asarray(debris.position, dtype=float)
    v_debris = np.asarray(debris.velocity, dtype=float)

    return r_sc, np.concatenate([r_debris, v_debris])


def simulate_relative_measurement(r_sc, debris_state):
    true_relative_position = debris_state[:3] - r_sc

    noise = rng.normal(
        loc=0.0,
        scale=MEASUREMENT_STD,
        size=3
    )

    return true_relative_position + noise


# ============================================================
# 6. INITIALIZE FILTER
# ============================================================

r_sc, initial_truth = get_truth()

first_relative_measurement = simulate_relative_measurement(
    r_sc, initial_truth
)

evaluation_truth_times = [0.0]
evaluation_truth_spacecraft_states = [
    np.concatenate([
        np.asarray(spacecraft.position, dtype=float),
        np.asarray(spacecraft.velocity, dtype=float),
    ])
]
evaluation_truth_debris_states = [initial_truth.copy()]

if INITIALIZATION_MODE == "truth-informed":
    initial_relative_velocity_guess = np.array([0.0, -2.0, 0.0])
    x = np.concatenate([
        r_sc + first_relative_measurement,
        np.asarray(spacecraft.velocity, dtype=float)
        + initial_relative_velocity_guess
    ])
    P = np.diag([
        MEASUREMENT_STD**2,
        MEASUREMENT_STD**2,
        MEASUREMENT_STD**2,
        INITIAL_VELOCITY_STD**2,
        INITIAL_VELOCITY_STD**2,
        INITIAL_VELOCITY_STD**2,
    ])
    START_STEP = 0
    initialization_measurement_count = 1
else:
    START_STEP = int(round(SENSOR_ONLY_WARMUP_SECONDS / DT))
    sample_steps = np.arange(
        0,
        START_STEP + 1,
        SENSOR_STEPS,
        dtype=int,
    )
    relative_measurements = [first_relative_measurement]
    spacecraft_samples = [
        np.concatenate([
            np.asarray(spacecraft.position, dtype=float),
            np.asarray(spacecraft.velocity, dtype=float),
        ])
    ]

    for step in range(1, START_STEP + 1):
        env.step(step_size=DT)
        current_time = step * DT
        r_sc, truth = get_truth()
        evaluation_truth_times.append(current_time)
        evaluation_truth_spacecraft_states.append(
            np.concatenate([
                np.asarray(spacecraft.position, dtype=float),
                np.asarray(spacecraft.velocity, dtype=float),
            ])
        )
        evaluation_truth_debris_states.append(truth.copy())

        if step % SENSOR_STEPS == 0:
            relative_measurements.append(
                simulate_relative_measurement(r_sc, truth)
            )
            spacecraft_samples.append(
                np.concatenate([
                    np.asarray(spacecraft.position, dtype=float),
                    np.asarray(spacecraft.velocity, dtype=float),
                ])
            )

    fit = fit_initial_orbit_from_relative_positions(
        relative_position_measurements=np.asarray(relative_measurements),
        spacecraft_states=np.asarray(spacecraft_samples),
        sample_steps=sample_steps,
        dt=DT,
        measurement_std=MEASUREMENT_STD,
    )
    x = fit.state.copy()
    P = fit.covariance.copy()
    for _ in range(START_STEP):
        F = numerical_jacobian(x, DT)
        x = rk4_step(x, DT)
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)
    initialization_measurement_count = fit.measurement_count


# ============================================================
# 7. EVALUATION LOGS
# ============================================================

# These logs are populated at EVERY simulation timestep.
# They do not control the estimator.

evaluation_times = []

position_errors = []
velocity_errors = []

position_sigmas = []
velocity_sigmas = []

# NIS is recorded only when a new measurement is processed.
nis_values = []
measurement_update_times = []


# ============================================================
# 7.1 HISTORY FOR ORBITAL COLLISION FORECASTING
# ============================================================

# EKF estimates and covariance are saved every sensor interval.

forecast_times = []
forecast_spacecraft_states = []
forecast_debris_estimates = []
forecast_debris_covariances = []

# Ground truth is saved separately for offline evaluation only.
# It is NOT supplied to the collision predictor.


# ============================================================
# 8. RUN THE ORBITAL EKF
# ============================================================

for step in range(START_STEP, TOTAL_STEPS + 1):

    current_time = step * DT

    # --------------------------------------------------------
    # A. GET CURRENT SIMULATOR STATE
    # --------------------------------------------------------

    r_sc, truth = get_truth()

    # Offline evaluation record only.
    # This data must not be used by the predictor or future RL agent.

    if not evaluation_truth_times or current_time > evaluation_truth_times[-1]:
        evaluation_truth_times.append(current_time)
        evaluation_truth_spacecraft_states.append(
            np.concatenate([
                np.asarray(spacecraft.position, dtype=float),
                np.asarray(spacecraft.velocity, dtype=float)
            ])
        )
        evaluation_truth_debris_states.append(truth.copy())

    # --------------------------------------------------------
    # B. SENSOR UPDATE
    # --------------------------------------------------------
    # The first sensor measurement was already used to
    # initialize x and P. The first Kalman update therefore
    # occurs at SENSOR_INTERVAL, not again at t = 0.

    if step > START_STEP and step % SENSOR_STEPS == 0:

        z_relative = simulate_relative_measurement(
            r_sc, truth
        )

        z_absolute = r_sc + z_relative

        # Innovation
        innovation = z_absolute - H @ x

        # Innovation covariance
        S = H @ P @ H.T + R

        # Normalized Innovation Squared
        nis = float(
            innovation @ np.linalg.solve(S, innovation)
        )

        nis_values.append(nis)
        measurement_update_times.append(current_time)

        # Kalman gain
        K = np.linalg.solve(S, H @ P).T

        # State update
        x = x + K @ innovation

        # Joseph-form covariance update
        I = np.eye(6)
        A = I - K @ H

        P = A @ P @ A.T + K @ R @ K.T

        # Numerical symmetry protection
        P = 0.5 * (P + P.T)

    # --------------------------------------------------------
    # C. EVALUATE AT EVERY TIMESTEP
    # --------------------------------------------------------
    # Evaluate after any measurement update at this time,
    # but before propagating to the next timestep.

    position_error = x[:3] - truth[:3]
    velocity_error = x[3:6] - truth[3:6]

    position_errors.append(position_error.copy())
    velocity_errors.append(velocity_error.copy())

    # Coordinate-wise standard deviations from covariance
    diagonal_P = np.maximum(np.diag(P), 0.0)

    sigma_position = np.sqrt(diagonal_P[:3])
    sigma_velocity = np.sqrt(diagonal_P[3:6])

    position_sigmas.append(sigma_position.copy())
    velocity_sigmas.append(sigma_velocity.copy())

    evaluation_times.append(current_time)

    # --------------------------------------------------------
    # C.1 SAVE FORECAST HISTORY AT SENSOR INTERVALS
    # --------------------------------------------------------
    # Save after any measurement update at the current time.

    if step % SENSOR_STEPS == 0:

        forecast_times.append(current_time)

        forecast_spacecraft_states.append(
            np.concatenate([
                np.asarray(spacecraft.position, dtype=float),
                np.asarray(spacecraft.velocity, dtype=float)
            ])
        )

        forecast_debris_estimates.append(x.copy())
        forecast_debris_covariances.append(P.copy())

    # --------------------------------------------------------
    # D. PROPAGATE FILTER AND SIMULATOR
    # --------------------------------------------------------
    # The simulator and EKF advance by DT seconds each iteration.

    if step < TOTAL_STEPS:

        F = numerical_jacobian(x, DT)

        x = rk4_step(x, DT)

        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        env.step(step_size=DT)


# ============================================================
# 9. CALCULATE ALL-TIMESTEP METRICS
# ============================================================

evaluation_times = np.asarray(evaluation_times)

position_errors = np.asarray(position_errors)
velocity_errors = np.asarray(velocity_errors)

position_sigmas = np.asarray(position_sigmas)
velocity_sigmas = np.asarray(velocity_sigmas)

nis_values = np.asarray(nis_values)
measurement_update_times = np.asarray(measurement_update_times)

# Expected shapes:
# position_errors: (301, 3)
# velocity_errors: (301, 3)
# position_sigmas: (301, 3)
# velocity_sigmas: (301, 3)

if len(evaluation_times) != TOTAL_STEPS - START_STEP + 1:
    raise RuntimeError(
        "Unexpected number of evaluation checkpoints."
    )

if len(nis_values) != (TOTAL_STEPS - START_STEP) // SENSOR_STEPS:
    raise RuntimeError(
        "Unexpected number of measurement updates."
    )
if len(measurement_update_times) != len(nis_values):
    raise RuntimeError("NIS values and measurement timestamps are misaligned.")

if len(forecast_times) != (
    (TOTAL_STEPS - START_STEP) // SENSOR_STEPS + 1
):
    raise RuntimeError(
        "Unexpected number of forecast checkpoints."
    )

metric_mask = evaluation_times >= METRIC_START_SECONDS
if not metric_mask.any():
    raise ValueError("No EKF state samples fall in the evaluation window.")

metric_position_errors = position_errors[metric_mask]
metric_velocity_errors = velocity_errors[metric_mask]
metric_position_sigmas = position_sigmas[metric_mask]
metric_velocity_sigmas = velocity_sigmas[metric_mask]
metric_nis_values = nis_values[
    measurement_update_times >= METRIC_START_SECONDS
]

# 3D RMSE, with one 3D position/velocity error per timestep in the window.

position_rmse = np.sqrt(
    np.mean(np.sum(metric_position_errors**2, axis=1))
)

velocity_rmse = np.sqrt(
    np.mean(np.sum(metric_velocity_errors**2, axis=1))
)

# Coordinate-wise position uncertainty coverage

coverage_1sigma = 100.0 * np.mean(
    np.abs(metric_position_errors) <= metric_position_sigmas
)

coverage_2sigma = 100.0 * np.mean(
    np.abs(metric_position_errors) <= 2.0 * metric_position_sigmas
)

# Coordinate-wise velocity uncertainty coverage

velocity_coverage_1sigma = 100.0 * np.mean(
    np.abs(metric_velocity_errors) <= metric_velocity_sigmas
)

velocity_coverage_2sigma = 100.0 * np.mean(
    np.abs(metric_velocity_errors) <= 2.0 * metric_velocity_sigmas
)

mean_nis = (
    float(np.mean(metric_nis_values))
    if len(metric_nis_values) > 0
    else float("nan")
)


# ============================================================
# 9.1 EXPORT EKF OUTPUT FOR COLLISION-RISK PREDICTION
# ============================================================

# Export only the spacecraft state and EKF estimate.
# Ground-truth debris state is intentionally NOT exported.

spacecraft_final_state = np.concatenate([
    np.asarray(spacecraft.position, dtype=float),
    np.asarray(spacecraft.velocity, dtype=float)
])

debris_final_estimated_state = x.copy()
debris_final_covariance = P.copy()

np.savez(
    OUTPUT_DIR / "ekf_collision_input.npz",
    spacecraft_state=spacecraft_final_state,
    debris_estimated_state=debris_final_estimated_state,
    debris_covariance=debris_final_covariance,
    simulation_time=float(TOTAL_TIME),
    measurement_std=float(MEASUREMENT_STD),
    random_seed=int(SEED),
    initialization_mode=INITIALIZATION_MODE,
    initial_velocity_std=(
        float(INITIAL_VELOCITY_STD)
        if INITIALIZATION_MODE == "truth-informed"
        else float("nan")
    ),
    acceleration_noise_std=float(ACCELERATION_NOISE_STD),
    initialization_warmup_seconds=float(START_STEP * DT),
    metric_evaluation_start_seconds=float(METRIC_START_SECONDS),
)

print("\n[7] COLLISION-PREDICTOR DATA EXPORT")
print("-" * 72)
print("  Output file               :", OUTPUT_DIR / "ekf_collision_input.npz")
print("  Spacecraft state shape    :", spacecraft_final_state.shape)
print("  Debris estimate shape     :", debris_final_estimated_state.shape)
print("  EKF covariance shape      :", debris_final_covariance.shape)
print("  Ground truth exported     : No")


# ============================================================
# 9.2 SAVE COLLISION-FORECAST INPUTS AND OFFLINE TRUTH
# ============================================================

np.savez(
    OUTPUT_DIR / "ekf_forecast_history.npz",
    times=np.asarray(forecast_times),
    spacecraft_states=np.asarray(forecast_spacecraft_states),
    debris_estimates=np.asarray(forecast_debris_estimates),
    debris_covariances=np.asarray(forecast_debris_covariances),
    total_time=float(TOTAL_TIME),
    dt=float(DT),
    sensor_interval=float(SENSOR_INTERVAL),
    measurement_std=float(MEASUREMENT_STD),
    random_seed=int(SEED),
    initialization_mode=INITIALIZATION_MODE,
    initial_velocity_std=(
        float(INITIAL_VELOCITY_STD)
        if INITIALIZATION_MODE == "truth-informed"
        else float("nan")
    ),
    acceleration_noise_std=float(ACCELERATION_NOISE_STD),
    initialization_warmup_seconds=float(START_STEP * DT),
    metric_evaluation_start_seconds=float(METRIC_START_SECONDS),
)

# Separate file: offline evaluation only.

np.savez(
    OUTPUT_DIR / "collision_forecast_truth.npz",
    times=np.asarray(evaluation_truth_times),
    spacecraft_states=np.asarray(
        evaluation_truth_spacecraft_states
    ),
    debris_states=np.asarray(
        evaluation_truth_debris_states
    )
)

print("\n[8] ORBITAL FORECAST HISTORY")
print("-" * 72)
print("  EKF history file          :", OUTPUT_DIR / "ekf_forecast_history.npz")
print("  Offline truth file        :", OUTPUT_DIR / "collision_forecast_truth.npz")
print("  EKF forecast checkpoints :", len(forecast_times))
print("  Ground-truth checkpoints :", len(evaluation_truth_times))
print("  Ground truth used by EKF predictor: No")


# ============================================================
# 10. PRINT ORGANIZED REPORT
# ============================================================

print("\n" + "=" * 72)
print("             ORBITAL EXTENDED KALMAN FILTER REPORT")
print("=" * 72)

print("\n[1] MODEL CONFIGURATION")
print("-" * 72)
print(f"  Orbital model              : Earth two-body gravity")
print(f"  Numerical integrator       : RK4")
print(f"  Simulation duration        : {TOTAL_TIME:.1f} s")
print(f"  Dynamics time step         : {DT:.1f} s")
print(f"  Sensor interval            : {SENSOR_INTERVAL:.1f} s")
print(f"  Measurement noise std      : {MEASUREMENT_STD:.1f} m/coordinate")
print(f"  Random seed                : {SEED}")
print(f"  Initialization mode       : {INITIALIZATION_MODE}")
if INITIALIZATION_MODE == "truth-informed":
    print(f"  Initial velocity std      : {INITIAL_VELOCITY_STD:.6f} m/s")
else:
    print(f"  Sensor-only fit warmup    : {START_STEP * DT:.1f} s")
    print(f"  Fit measurements          : {initialization_measurement_count}")
    print(f"  Fit residual RMS          : {fit.residual_rms_m:.3f} m")
print(f"  Acceleration noise std    : {ACCELERATION_NOISE_STD:.6f} m/s^2")

print("\n[2] EVALUATION BOOKKEEPING")
print("-" * 72)
print(f"  Evaluation interval        : {DT:.1f} s")
print(f"  Metric window start        : {METRIC_START_SECONDS:.1f} s")
print(f"  Metric state samples       : {int(metric_mask.sum())}")
print(f"  Initializing measurements  : {initialization_measurement_count}")
print(f"  Kalman measurement updates : {len(nis_values)}")
print(f"  NIS samples in metric window: {len(metric_nis_values)}")
print("  Evaluation includes predictions between measurements.")

print("\n[3] ESTIMATION ACCURACY — ALL TIMESTEPS")
print("-" * 72)
print(f"  3D position RMSE           : {position_rmse:.3f} m")
print(f"  3D velocity RMSE           : {velocity_rmse:.6f} m/s")

print("\n[4] UNCERTAINTY VALIDATION")
print("-" * 72)
print(f"  Mean NIS                   : {mean_nis:.3f}")
print("  Reference for 3D measurement: approximately 3.0")
print(f"  Coordinate 1-sigma coverage : {coverage_1sigma:.1f}%")
print(f"  Coordinate 2-sigma coverage : {coverage_2sigma:.1f}%")
print(f"  Velocity 1-sigma coverage   : {velocity_coverage_1sigma:.1f}%")
print(f"  Velocity 2-sigma coverage   : {velocity_coverage_2sigma:.1f}%")

print("\n[5] FINAL STATE ESTIMATE")
print("-" * 72)
print("  Debris position (m):")
print(" ", np.array2string(x[:3], precision=3))
print("  Debris velocity (m/s):")
print(" ", np.array2string(x[3:6], precision=6))

print("\n[6] IMPORTANT LIMITATIONS")
print("-" * 72)
print("  The filter assumes two-body Earth gravity.")
print("  OrbitZoo may include dynamics not represented in this model.")
print("  Process noise is a tunable approximation.")
print("  Ground truth is used for sensor simulation and offline evaluation.")
print("  NIS is calculated only for actual Kalman measurement updates.")
print("  Coverage checkpoints are temporally correlated.")
if INITIALIZATION_MODE == "truth-informed":
    print("  Initial relative velocity is truth-informed in this experiment.")
else:
    print("  Initial state is fit from noisy position-only measurements.")
print("  This experiment alone does not establish filter consistency.")

print("=" * 72)

env.close()