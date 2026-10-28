
# import numpy as np


# # ============================================================
# # 1. CONFIGURATION
# # ============================================================

# MU_EARTH = 3.986004418e14

# INPUT_FILE = "ekf_forecast_history.npz"
# TRUTH_FILE = "collision_forecast_truth.npz"

# COLLISION_RADIUS = 10.0

# # Use the same acceleration-noise parameter as the EKF.
# ACCELERATION_NOISE_STD = 0.01


# # ============================================================
# # 2. ORBITAL DYNAMICS
# # ============================================================

# def orbital_dynamics(state):
#     """Two-body Earth orbital dynamics."""

#     r = state[:3]
#     v = state[3:]

#     radius = np.linalg.norm(r)

#     if radius < 1.0:
#         raise ValueError("Invalid orbital radius.")

#     acceleration = -MU_EARTH * r / radius**3

#     return np.concatenate([v, acceleration])


# def rk4_step(state, dt):
#     """Propagate a six-dimensional orbital state using RK4."""

#     k1 = orbital_dynamics(state)
#     k2 = orbital_dynamics(state + 0.5 * dt * k1)
#     k3 = orbital_dynamics(state + 0.5 * dt * k2)
#     k4 = orbital_dynamics(state + dt * k3)

#     return state + (dt / 6.0) * (
#         k1 + 2.0 * k2 + 2.0 * k3 + k4
#     )


# # ============================================================
# # 3. COVARIANCE PROPAGATION
# # ============================================================

# def numerical_jacobian(state, dt):
#     """Numerical Jacobian of the discrete RK4 propagation."""

#     F = np.zeros((6, 6))

#     for i in range(6):

#         epsilon = 1e-3 if i < 3 else 1e-5

#         delta = np.zeros(6)
#         delta[i] = epsilon

#         state_plus = rk4_step(state + delta, dt)
#         state_minus = rk4_step(state - delta, dt)

#         F[:, i] = (
#             state_plus - state_minus
#         ) / (2.0 * epsilon)

#     return F


# def process_noise(dt):
#     """Same acceleration-noise structure used in the EKF."""

#     Q = np.zeros((6, 6))

#     block = ACCELERATION_NOISE_STD**2 * np.array([
#         [dt**4 / 4.0, dt**3 / 2.0],
#         [dt**3 / 2.0, dt**2],
#     ])

#     for axis in range(3):

#         Q[axis, axis] = block[0, 0]
#         Q[axis, axis + 3] = block[0, 1]

#         Q[axis + 3, axis] = block[1, 0]
#         Q[axis + 3, axis + 3] = block[1, 1]

#     return Q


# # ============================================================
# # 4. FUTURE ORBITAL CLOSEST-APPROACH PREDICTION
# # ============================================================

# def predict_orbital_closest_approach(
#     spacecraft_state,
#     debris_estimated_state,
#     debris_covariance,
#     prediction_horizon,
#     dt,
# ):
#     """
#     Predict future closest approach by propagating both objects
#     under two-body gravity.

#     Debris covariance is propagated using a numerical Jacobian.

#     The spacecraft is treated as having an exact state.
#     """

#     spacecraft_state = np.asarray(
#         spacecraft_state, dtype=float
#     ).copy()

#     debris_state = np.asarray(
#         debris_estimated_state, dtype=float
#     ).copy()

#     P = np.asarray(
#         debris_covariance, dtype=float
#     ).copy()

#     if spacecraft_state.shape != (6,):
#         raise ValueError("Spacecraft state must have shape (6,).")

#     if debris_state.shape != (6,):
#         raise ValueError("Debris state must have shape (6,).")

#     if P.shape != (6, 6):
#         raise ValueError("Covariance must have shape (6, 6).")

#     if prediction_horizon < 0:
#         raise ValueError("Prediction horizon cannot be negative.")

#     if dt <= 0:
#         raise ValueError("dt must be positive.")

#     number_of_steps = int(
#         round(prediction_horizon / dt)
#     )

#     if not np.isclose(
#         number_of_steps * dt,
#         prediction_horizon,
#     ):
#         raise ValueError(
#             "Prediction horizon must be divisible by dt."
#         )

#     Q = process_noise(dt)

#     times = [0.0]
#     miss_distances = []
#     miss_vectors = []
#     position_covariances = []

#     def record_state(current_time):

#         relative_position = (
#             debris_state[:3] - spacecraft_state[:3]
#         )

#         miss_vectors.append(relative_position.copy())

#         miss_distances.append(
#             float(np.linalg.norm(relative_position))
#         )

#         position_covariances.append(
#             P[:3, :3].copy()
#         )

#         times.append(current_time)

#     # Record current epoch.
#     record_state(0.0)

#     # Propagate both orbital states into the future.
#     for step in range(number_of_steps):

#         F = numerical_jacobian(debris_state, dt)

#         debris_state = rk4_step(debris_state, dt)
#         spacecraft_state = rk4_step(
#             spacecraft_state, dt
#         )

#         P = F @ P @ F.T + Q
#         P = 0.5 * (P + P.T)

#         current_time = (step + 1) * dt

#         record_state(current_time)

#     times = np.asarray(times)

#     # The first element in times is an initial record;
#     # distances and covariance arrays align with each recorded state.
#     miss_distances = np.asarray(miss_distances)
#     miss_vectors = np.asarray(miss_vectors)
#     position_covariances = np.asarray(
#         position_covariances
#     )

#     closest_index = int(np.argmin(miss_distances))

#     tca = float(times[closest_index])
#     miss_distance = float(
#         miss_distances[closest_index]
#     )

#     miss_vector = miss_vectors[closest_index]

#     P_position = position_covariances[
#         closest_index
#     ]

#     # Estimate radial miss-distance uncertainty.
#     if miss_distance > 1e-9:

#         direction = miss_vector / miss_distance

#         miss_variance = (
#             direction @ P_position @ direction
#         )

#         miss_sigma = float(
#             np.sqrt(max(miss_variance, 0.0))
#         )

#     else:

#         # At zero miss distance, the radial direction is undefined.
#         eigenvalues = np.linalg.eigvalsh(P_position)

#         miss_sigma = float(
#             np.sqrt(max(eigenvalues[-1], 0.0))
#         )

#     lower_3sigma = max(
#         0.0,
#         miss_distance - 3.0 * miss_sigma,
#     )

#     upper_3sigma = (
#         miss_distance + 3.0 * miss_sigma
#     )

#     if lower_3sigma > COLLISION_RADIUS:

#         risk_indicator = (
#             "LOWER UNCERTAINTY BOUND ABOVE RADIUS"
#         )

#     elif upper_3sigma < COLLISION_RADIUS:

#         risk_indicator = (
#             "UNCERTAINTY REGION WITHIN RADIUS"
#         )

#     else:

#         risk_indicator = (
#             "POTENTIAL ENCOUNTER — NEEDS FURTHER ANALYSIS"
#         )

#     return {
#         "tca_seconds": tca,
#         "miss_distance_m": miss_distance,
#         "miss_sigma_m": miss_sigma,
#         "lower_3sigma_m": lower_3sigma,
#         "upper_3sigma_m": upper_3sigma,
#         "risk_indicator": risk_indicator,
#         "miss_distances": miss_distances,
#         "times": times,
#     }


# # ============================================================
# # 5. MAIN PROGRAM
# # ============================================================

# def main():

#     with np.load(INPUT_FILE) as data:

#         forecast_times = data["times"].copy()

#         spacecraft_states = (
#             data["spacecraft_states"].copy()
#         )

#         debris_estimates = (
#             data["debris_estimates"].copy()
#         )

#         debris_covariances = (
#             data["debris_covariances"].copy()
#         )

#         total_time = float(data["total_time"])
#         dt = float(data["dt"])

#     results = []

#     print("\n" + "=" * 100)
#     print("       ORBITAL COLLISION FORECAST — EKF ESTIMATE + TWO-BODY PROPAGATION")
#     print("=" * 100)

#     print(
#         f"\nSimulation duration : {total_time:.1f} s"
#     )

#     print(
#         f"Propagation step    : {dt:.1f} s"
#     )

#     print(
#         f"Forecast checkpoints: {len(forecast_times)}"
#     )

#     print(
#         f"Collision radius    : {COLLISION_RADIUS:.1f} m"
#     )

#     print("\n[1] FORWARD PREDICTIONS")
#     print("-" * 100)

#     print(
#         f"{'Epoch (s)':>10} | "
#         f"{'Horizon (s)':>11} | "
#         f"{'TCA (s)':>10} | "
#         f"{'Miss (m)':>12} | "
#         f"{'Sigma (m)':>11} | "
#         f"{'Screening':<45}"
#     )

#     print("-" * 100)

#     for i, epoch in enumerate(forecast_times):

#         horizon = total_time - float(epoch)

#         result = predict_orbital_closest_approach(
#             spacecraft_state=spacecraft_states[i],
#             debris_estimated_state=debris_estimates[i],
#             debris_covariance=debris_covariances[i],
#             prediction_horizon=horizon,
#             dt=dt,
#         )

#         results.append(result)

#         print(
#             f"{epoch:10.1f} | "
#             f"{horizon:11.1f} | "
#             f"{result['tca_seconds']:10.1f} | "
#             f"{result['miss_distance_m']:12.3f} | "
#             f"{result['miss_sigma_m']:11.3f} | "
#             f"{result['risk_indicator']:<45}"
#         )

#     # --------------------------------------------------------
#     # OFFLINE EVALUATION
#     # --------------------------------------------------------
#     # Ground truth is loaded only after predictions have been
#     # generated. It is never used by the predictor.
#     # --------------------------------------------------------

#     with np.load(TRUTH_FILE) as data:

#         truth_times = data["times"].copy()

#         truth_spacecraft_states = (
#             data["spacecraft_states"].copy()
#         )

#         truth_debris_states = (
#             data["debris_states"].copy()
#         )

#     actual_tca_errors = []
#     actual_miss_errors = []

#     print("\n[2] OFFLINE EVALUATION AGAINST ORBITZOO")
#     print("-" * 100)

#     print(
#         f"{'Epoch (s)':>10} | "
#         f"{'Pred TCA (s)':>12} | "
#         f"{'Actual TCA (s)':>14} | "
#         f"{'Pred Miss (m)':>13} | "
#         f"{'Actual Miss (m)':>14} | "
#         f"{'TCA Error (s)':>13}"
#     )

#     print("-" * 100)

#     for i, epoch in enumerate(forecast_times):

#         # Checkpoint times align with the truth sampling grid.
#         start_index = int(
#             round(float(epoch) / dt)
#         )

#         future_sc = truth_spacecraft_states[
#             start_index:
#         ]

#         future_debris = truth_debris_states[
#             start_index:
#         ]

#         actual_relative_positions = (
#             future_debris[:, :3]
#             - future_sc[:, :3]
#         )

#         actual_distances = np.linalg.norm(
#             actual_relative_positions,
#             axis=1,
#         )

#         actual_index = int(
#             np.argmin(actual_distances)
#         )

#         actual_tca = actual_index * dt

#         actual_miss = float(
#             actual_distances[actual_index]
#         )

#         predicted_tca = results[i][
#             "tca_seconds"
#         ]

#         predicted_miss = results[i][
#             "miss_distance_m"
#         ]

#         tca_error = abs(
#             predicted_tca - actual_tca
#         )

#         miss_error = abs(
#             predicted_miss - actual_miss
#         )

#         actual_tca_errors.append(tca_error)
#         actual_miss_errors.append(miss_error)

#         print(
#             f"{epoch:10.1f} | "
#             f"{predicted_tca:12.1f} | "
#             f"{actual_tca:14.1f} | "
#             f"{predicted_miss:13.3f} | "
#             f"{actual_miss:14.3f} | "
#             f"{tca_error:13.1f}"
#         )

#     actual_tca_errors = np.asarray(
#         actual_tca_errors
#     )

#     actual_miss_errors = np.asarray(
#         actual_miss_errors
#     )

#     print("\n[3] AGGREGATED OFFLINE ERRORS")
#     print("-" * 100)

#     print(
#         "  Mean absolute TCA error : "
#         f"{np.mean(actual_tca_errors):.3f} s"
#     )

#     print(
#         "  Mean absolute miss error: "
#         f"{np.mean(actual_miss_errors):.3f} m"
#     )

#     print("\n" + "=" * 100)
#     print("NOTE: Screening is not a formal collision probability.")
#     print("Ground truth is used only for offline evaluation.")
#     print("=" * 100)


# if __name__ == "__main__":
#     main()



import os
import numpy as np
from orbital_mechanics import numerical_jacobian, rk4_step


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "ekf_forecast_history.npz"
TRUTH_FILE = "collision_forecast_truth.npz"

COLLISION_RADIUS = 10.0

EVALUATION_START_SECONDS = float(
    os.environ.get("ORBITAL_FORECAST_EVALUATION_START_SECONDS", "0.0")
)

if not np.isfinite(EVALUATION_START_SECONDS) or EVALUATION_START_SECONDS < 0:
    raise ValueError("Evaluation start time must be finite and non-negative.")


# ============================================================
# 2. ORBITAL DYNAMICS
# ============================================================

# ============================================================
# 3. COVARIANCE PROPAGATION
# ============================================================

def process_noise(dt, acceleration_noise_std=0.0):
    """Same acceleration-noise structure used in the EKF."""

    Q = np.zeros((6, 6))

    block = acceleration_noise_std**2 * np.array([
        [dt**4 / 4.0, dt**3 / 2.0],
        [dt**3 / 2.0, dt**2]
    ])

    for axis in range(3):

        Q[axis, axis] = block[0, 0]
        Q[axis, axis + 3] = block[0, 1]

        Q[axis + 3, axis] = block[1, 0]
        Q[axis + 3, axis + 3] = block[1, 1]

    return Q


# ============================================================
# 4. FUTURE ORBITAL CLOSEST-APPROACH PREDICTION
# ============================================================

def predict_orbital_closest_approach(
    spacecraft_state,
    debris_estimated_state,
    debris_covariance,
    prediction_horizon,
    dt,
    acceleration_noise_std=0.0,
):
    """
    Predict future closest approach by propagating both objects
    under two-body gravity.

    Debris covariance is propagated using a numerical Jacobian.

    The spacecraft is treated as having an exact state.
    """

    spacecraft_state = np.asarray(
        spacecraft_state, dtype=float
    ).copy()

    debris_state = np.asarray(
        debris_estimated_state, dtype=float
    ).copy()

    P = np.asarray(
        debris_covariance, dtype=float
    ).copy()

    if spacecraft_state.shape != (6,):
        raise ValueError("Spacecraft state must have shape (6,).")

    if debris_state.shape != (6,):
        raise ValueError("Debris state must have shape (6,).")

    if P.shape != (6, 6):
        raise ValueError("Covariance must have shape (6, 6).")

    if prediction_horizon < 0:
        raise ValueError("Prediction horizon cannot be negative.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    number_of_steps = int(
        round(prediction_horizon / dt)
    )

    if not np.isclose(
        number_of_steps * dt,
        prediction_horizon,
    ):
        raise ValueError(
            "Prediction horizon must be divisible by dt."
        )

    if not np.isfinite(acceleration_noise_std) or acceleration_noise_std < 0:
        raise ValueError("acceleration_noise_std must be finite and non-negative.")
    Q = process_noise(dt, acceleration_noise_std)

    # CORRECTION:
    # Start with an empty time array. record_state() appends
    # the time corresponding to each recorded state.
    #
    # Previously, times started as [0.0] and record_state(0.0)
    # appended another 0.0, creating a one-entry misalignment.

    times = []
    miss_distances = []
    miss_vectors = []
    position_covariances = []

    def record_state(current_time):

        relative_position = (
            debris_state[:3] - spacecraft_state[:3]
        )

        miss_vectors.append(relative_position.copy())

        miss_distances.append(
            float(np.linalg.norm(relative_position))
        )

        position_covariances.append(
            P[:3, :3].copy()
        )

        times.append(float(current_time))

    # Record current epoch exactly once.
    record_state(0.0)

    # Propagate both orbital states into the future.
    for step in range(number_of_steps):

        F = numerical_jacobian(debris_state, dt)

        debris_state = rk4_step(debris_state, dt)

        spacecraft_state = rk4_step(
            spacecraft_state, dt
        )

        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        current_time = (step + 1) * dt

        record_state(current_time)

    times = np.asarray(times)

    miss_distances = np.asarray(miss_distances)
    miss_vectors = np.asarray(miss_vectors)

    position_covariances = np.asarray(
        position_covariances
    )

    # All recorded arrays now have the same number of entries.
    if not (
        len(times)
        == len(miss_distances)
        == len(miss_vectors)
        == len(position_covariances)
    ):
        raise RuntimeError(
            "Time, distance, vector, and covariance arrays "
            "are not aligned."
        )

    closest_index = int(np.argmin(miss_distances))

    tca = float(times[closest_index])

    miss_distance = float(
        miss_distances[closest_index]
    )

    miss_vector = miss_vectors[closest_index]

    P_position = position_covariances[
        closest_index
    ]

    # --------------------------------------------------------
    # ESTIMATE RADIAL MISS-DISTANCE UNCERTAINTY
    # --------------------------------------------------------

    if miss_distance > 1e-9:

        direction = miss_vector / miss_distance

        miss_variance = (
            direction @ P_position @ direction
        )

        miss_sigma = float(
            np.sqrt(max(miss_variance, 0.0))
        )

    else:

        # At zero miss distance, the radial direction is undefined.
        eigenvalues = np.linalg.eigvalsh(P_position)

        miss_sigma = float(
            np.sqrt(max(eigenvalues[-1], 0.0))
        )

    lower_3sigma = max(
        0.0,
        miss_distance - 3.0 * miss_sigma,
    )

    upper_3sigma = (
        miss_distance + 3.0 * miss_sigma
    )

    # --------------------------------------------------------
    # SIMPLE UNCERTAINTY SCREENING
    # --------------------------------------------------------

    if lower_3sigma > COLLISION_RADIUS:

        risk_indicator = (
            "LOWER UNCERTAINTY BOUND ABOVE RADIUS"
        )

    elif upper_3sigma < COLLISION_RADIUS:

        risk_indicator = (
            "UNCERTAINTY REGION WITHIN RADIUS"
        )

    else:

        risk_indicator = (
            "POTENTIAL ENCOUNTER — NEEDS FURTHER ANALYSIS"
        )

    return {
        "tca_seconds": tca,
        "miss_distance_m": miss_distance,
        "miss_sigma_m": miss_sigma,
        "lower_3sigma_m": lower_3sigma,
        "upper_3sigma_m": upper_3sigma,
        "risk_indicator": risk_indicator,
        "miss_vector_m": miss_vector.copy(),
        "miss_distances": miss_distances,
        "times": times,
    }


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD EKF FORECAST HISTORY
    # --------------------------------------------------------

    with np.load(INPUT_FILE) as data:

        forecast_times = data["times"].copy()

        spacecraft_states = (
            data["spacecraft_states"].copy()
        )

        debris_estimates = (
            data["debris_estimates"].copy()
        )

        debris_covariances = (
            data["debris_covariances"].copy()
        )

        total_time = float(data["total_time"])
        dt = float(data["dt"])
        acceleration_noise_std = (
            float(data["acceleration_noise_std"])
            if "acceleration_noise_std" in data
            else 0.01
        )

    if EVALUATION_START_SECONDS > total_time:
        raise ValueError("Evaluation start time exceeds the simulation duration.")

    results = []

    print("\n" + "=" * 100)
    print(
        "       ORBITAL COLLISION FORECAST — "
        "EKF ESTIMATE + TWO-BODY PROPAGATION"
    )
    print("=" * 100)

    print(
        f"\nSimulation duration : {total_time:.1f} s"
    )

    print(
        f"Propagation step    : {dt:.1f} s"
    )

    print(
        f"Forecast checkpoints: {len(forecast_times)}"
    )

    print(
        f"Collision radius    : {COLLISION_RADIUS:.1f} m"
    )

    # --------------------------------------------------------
    # FORWARD PREDICTIONS
    # --------------------------------------------------------

    print("\n[1] FORWARD PREDICTIONS")
    print("-" * 100)

    print(
        f"{'Epoch (s)':>10} | "
        f"{'Horizon (s)':>11} | "
        f"{'TCA (s)':>10} | "
        f"{'Miss (m)':>12} | "
        f"{'Sigma (m)':>11} | "
        f"{'Screening':<45}"
    )

    print("-" * 100)

    for i, epoch in enumerate(forecast_times):

        horizon = total_time - float(epoch)

        result = predict_orbital_closest_approach(
            spacecraft_state=spacecraft_states[i],
            debris_estimated_state=debris_estimates[i],
            debris_covariance=debris_covariances[i],
            prediction_horizon=horizon,
            dt=dt,
            acceleration_noise_std=acceleration_noise_std,
        )

        results.append(result)

        print(
            f"{epoch:10.1f} | "
            f"{horizon:11.1f} | "
            f"{result['tca_seconds']:10.1f} | "
            f"{result['miss_distance_m']:12.3f} | "
            f"{result['miss_sigma_m']:11.3f} | "
            f"{result['risk_indicator']:<45}"
        )

    # --------------------------------------------------------
    # OFFLINE EVALUATION
    # --------------------------------------------------------
    # Ground truth is loaded only after predictions have been
    # generated. It is never used by the predictor.
    # --------------------------------------------------------

    with np.load(TRUTH_FILE) as data:

        truth_times = data["times"].copy()

        truth_spacecraft_states = (
            data["spacecraft_states"].copy()
        )

        truth_debris_states = (
            data["debris_states"].copy()
        )

    # --------------------------------------------------------
    # CHECK INPUT ALIGNMENT
    # --------------------------------------------------------

    if len(truth_times) != len(truth_spacecraft_states):
        raise RuntimeError(
            "Truth times and spacecraft states have different lengths."
        )

    if len(truth_times) != len(truth_debris_states):
        raise RuntimeError(
            "Truth times and debris states have different lengths."
        )

    if not np.allclose(
        truth_times,
        np.arange(len(truth_times)) * dt,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Truth timestamps do not match the expected timestep grid."
        )

    actual_tca_errors = []
    actual_miss_errors = []

    print("\n[2] OFFLINE EVALUATION AGAINST ORBITZOO")
    print("-" * 100)

    print(
        f"{'Epoch (s)':>10} | "
        f"{'Pred TCA (s)':>12} | "
        f"{'Actual TCA (s)':>14} | "
        f"{'Pred Miss (m)':>13} | "
        f"{'Actual Miss (m)':>14} | "
        f"{'TCA Error (s)':>13}"
    )

    print("-" * 100)

    for i, epoch in enumerate(forecast_times):

        # Checkpoint times align with the truth sampling grid.
        start_index = int(
            round(float(epoch) / dt)
        )

        if start_index < 0 or start_index >= len(truth_times):
            raise IndexError(
                f"Forecast epoch {epoch} is outside truth data."
            )

        # Verify forecast checkpoint aligns with truth timestamp.
        if not np.isclose(
            truth_times[start_index],
            epoch,
            atol=1e-8,
        ):
            raise RuntimeError(
                f"Forecast epoch {epoch} does not align "
                "with its truth timestamp."
            )

        future_sc = truth_spacecraft_states[
            start_index:
        ]

        future_debris = truth_debris_states[
            start_index:
        ]

        actual_relative_positions = (
            future_debris[:, :3]
            - future_sc[:, :3]
        )

        actual_distances = np.linalg.norm(
            actual_relative_positions,
            axis=1,
        )

        actual_index = int(
            np.argmin(actual_distances)
        )

        actual_tca = actual_index * dt

        actual_miss = float(
            actual_distances[actual_index]
        )

        predicted_tca = results[i][
            "tca_seconds"
        ]

        predicted_miss = results[i][
            "miss_distance_m"
        ]

        tca_error = abs(
            predicted_tca - actual_tca
        )

        miss_error = abs(
            predicted_miss - actual_miss
        )

        if epoch >= EVALUATION_START_SECONDS:
            actual_tca_errors.append(tca_error)
            actual_miss_errors.append(miss_error)

        print(
            f"{epoch:10.1f} | "
            f"{predicted_tca:12.1f} | "
            f"{actual_tca:14.1f} | "
            f"{predicted_miss:13.3f} | "
            f"{actual_miss:14.3f} | "
            f"{tca_error:13.1f}"
        )

    actual_tca_errors = np.asarray(
        actual_tca_errors
    )

    actual_miss_errors = np.asarray(
        actual_miss_errors
    )
    if len(actual_tca_errors) == 0:
        raise ValueError("No forecast checkpoints fall in the evaluation window.")

    # --------------------------------------------------------
    # AGGREGATED OFFLINE ERRORS
    # --------------------------------------------------------

    print("\n[3] AGGREGATED OFFLINE ERRORS")
    print("-" * 100)
    print(
        f"  Evaluation start time    : {EVALUATION_START_SECONDS:.1f} s"
    )
    print(
        f"  Evaluated checkpoints    : {len(actual_tca_errors)}"
    )

    print(
        "  Mean absolute TCA error : "
        f"{np.mean(actual_tca_errors):.3f} s"
    )

    print(
        "  Mean absolute miss error: "
        f"{np.mean(actual_miss_errors):.3f} m"
    )

    print("\n" + "=" * 100)
    print("NOTE: Screening is not a formal collision probability.")
    print("Ground truth is used only for offline evaluation.")
    print("=" * 100)


if __name__ == "__main__":
    main()