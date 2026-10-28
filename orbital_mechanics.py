"""Shared two-body orbital propagation and measurement-only orbit fitting."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


MU_EARTH = 3.986004418e14


def two_body_dynamics(state: np.ndarray) -> np.ndarray:
    position = state[:3]
    velocity = state[3:]
    radius = np.linalg.norm(position)

    if not np.isfinite(radius) or radius < 1.0:
        raise ValueError("Invalid orbital radius.")

    acceleration = -MU_EARTH * position / radius**3
    return np.concatenate([velocity, acceleration])


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("Orbital state must be a finite vector with shape (6,).")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")

    k1 = two_body_dynamics(state)
    k2 = two_body_dynamics(state + 0.5 * dt * k1)
    k3 = two_body_dynamics(state + 0.5 * dt * k2)
    k4 = two_body_dynamics(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def numerical_jacobian(state: np.ndarray, dt: float) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("Orbital state must be a finite vector with shape (6,).")

    jacobian = np.zeros((6, 6), dtype=float)
    for index in range(6):
        epsilon = 1e-3 if index < 3 else 1e-5
        delta = np.zeros(6)
        delta[index] = epsilon
        jacobian[:, index] = (
            rk4_step(state + delta, dt) - rk4_step(state - delta, dt)
        ) / (2.0 * epsilon)
    return jacobian


@dataclass(frozen=True)
class OrbitFit:
    state: np.ndarray
    covariance: np.ndarray
    residual_rms_m: float
    measurement_count: int
    optimality: float


def fit_initial_orbit_from_relative_positions(
    relative_position_measurements: np.ndarray,
    spacecraft_states: np.ndarray,
    sample_steps: np.ndarray,
    dt: float,
    measurement_std: float,
) -> OrbitFit:
    """Estimate an initial debris state from noisy relative-position samples.

    The measurements are propagated against a two-body debris model. Spacecraft
    states are treated as known onboard navigation data; no debris truth states
    are consumed by the estimator.
    """
    measurements = np.asarray(relative_position_measurements, dtype=float)
    spacecraft_states = np.asarray(spacecraft_states, dtype=float)
    sample_steps = np.asarray(sample_steps, dtype=int)

    if measurements.ndim != 2 or measurements.shape[1] != 3:
        raise ValueError("Measurements must have shape (n, 3).")
    if spacecraft_states.shape != (len(measurements), 6):
        raise ValueError("Spacecraft states must have shape (n, 6).")
    if sample_steps.shape != (len(measurements),):
        raise ValueError("sample_steps must have one entry per measurement.")
    if len(measurements) < 3:
        raise ValueError("At least three measurements are required for orbit fitting.")
    if not np.isfinite(measurements).all() or not np.isfinite(spacecraft_states).all():
        raise ValueError("Measurements and spacecraft states must be finite.")
    if np.any(sample_steps < 0) or np.any(np.diff(sample_steps) <= 0):
        raise ValueError("sample_steps must be non-negative and strictly increasing.")
    if sample_steps[0] != 0:
        raise ValueError("The first measurement must be at the fit epoch (step 0).")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if not np.isfinite(measurement_std) or measurement_std <= 0:
        raise ValueError("measurement_std must be finite and positive.")

    fit_horizon = int(sample_steps[-1])

    def residual(relative_initial_state: np.ndarray) -> np.ndarray:
        initial_spacecraft_state = spacecraft_states[0]
        debris_state = np.concatenate([
            initial_spacecraft_state[:3] + relative_initial_state[:3],
            initial_spacecraft_state[3:] + relative_initial_state[3:],
        ])
        predicted_relative_positions = []
        sample_index = 0

        for step in range(fit_horizon + 1):
            if step == sample_steps[sample_index]:
                predicted_relative_positions.append(
                    debris_state[:3] - spacecraft_states[sample_index, :3]
                )
                sample_index += 1
                if sample_index == len(sample_steps):
                    break
            debris_state = rk4_step(debris_state, dt)

        return (
            np.asarray(predicted_relative_positions) - measurements
        ).ravel() / measurement_std

    initial_guess = np.concatenate([measurements[0], np.zeros(3)])
    result = least_squares(
        residual,
        initial_guess,
        x_scale=np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
        max_nfev=200,
    )
    if not result.success:
        raise RuntimeError(f"Sensor-only orbital fit failed: {result.message}")
    if np.linalg.matrix_rank(result.jac) < 6:
        raise RuntimeError("Sensor-only orbital fit is underdetermined.")

    information = result.jac.T @ result.jac
    covariance = measurement_std**2 * np.linalg.inv(information)
    covariance = 0.5 * (covariance + covariance.T)
    if np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise RuntimeError("Sensor-only fit produced a non-positive covariance.")

    initial_spacecraft_state = spacecraft_states[0]
    state = np.concatenate([
        initial_spacecraft_state[:3] + result.x[:3],
        initial_spacecraft_state[3:] + result.x[3:],
    ])
    residuals_m = result.fun * measurement_std
    return OrbitFit(
        state=state,
        covariance=covariance,
        residual_rms_m=float(np.sqrt(np.mean(residuals_m**2))),
        measurement_count=len(measurements),
        optimality=float(result.optimality),
    )
