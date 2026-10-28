"""Scenario-level impulse planner and thrust conversion for Orekit RSW actions."""

from dataclasses import dataclass

import numpy as np

from orbital_mechanics import rk4_step


STANDARD_GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class ManeuverPlan:
    delta_v_inertial_mps: np.ndarray
    delta_v_rsw_mps: np.ndarray
    predicted_miss_before_m: float
    predicted_miss_after_linear_m: float
    target_miss_distance_m: float
    required_delta_v_mps: float | None
    feasible: bool
    predicted_miss_after_model_m: float | None = None
    planning_model: str = "linear"
    infeasibility_reason: str | None = None
    commanded_delta_v_mps: float | None = None


def inertial_to_rsw(vector: np.ndarray, spacecraft_state: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    state = np.asarray(spacecraft_state, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("vector must be a finite three-element vector.")
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("spacecraft_state must be a finite six-element vector.")

    radius = np.linalg.norm(state[:3])
    if radius < 1e-12:
        raise ValueError("Spacecraft position cannot be zero.")
    radial = state[:3] / radius
    angular_momentum = np.cross(state[:3], state[3:])
    momentum_norm = np.linalg.norm(angular_momentum)
    if momentum_norm < 1e-12:
        raise ValueError("Spacecraft state does not define an orbital plane.")
    cross_track = angular_momentum / momentum_norm
    along_track = np.cross(cross_track, radial)
    basis = np.column_stack((radial, along_track, cross_track))
    return basis.T @ vector


def plan_avoidance_maneuver(
    spacecraft_state: np.ndarray,
    forecast: dict,
    target_miss_distance_m: float,
    max_delta_v_mps: float | None = None,
) -> ManeuverPlan:
    state = np.asarray(spacecraft_state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("spacecraft_state must be a finite six-element vector.")
    if not np.isfinite(target_miss_distance_m) or target_miss_distance_m <= 0:
        raise ValueError("target_miss_distance_m must be finite and positive.")
    if max_delta_v_mps is not None and (
        not np.isfinite(max_delta_v_mps) or max_delta_v_mps <= 0
    ):
        raise ValueError("max_delta_v_mps must be finite and positive.")

    tca = float(forecast["tca_seconds"])
    miss_distance = float(forecast["miss_distance_m"])
    miss_vector = np.asarray(forecast["miss_vector_m"], dtype=float)
    if (
        not np.isfinite(tca)
        or not np.isfinite(miss_distance)
        or miss_distance < 0
        or miss_vector.shape != (3,)
        or not np.isfinite(miss_vector).all()
    ):
        raise ValueError("Forecast contains invalid closest-approach values.")
    if not np.isclose(np.linalg.norm(miss_vector), miss_distance, rtol=1e-5, atol=1e-6):
        raise ValueError("Forecast miss vector and distance are inconsistent.")

    if miss_distance >= target_miss_distance_m:
        zero = np.zeros(3)
        return ManeuverPlan(
            delta_v_inertial_mps=zero,
            delta_v_rsw_mps=zero.copy(),
            predicted_miss_before_m=miss_distance,
            predicted_miss_after_linear_m=miss_distance,
            target_miss_distance_m=target_miss_distance_m,
            required_delta_v_mps=0.0,
            feasible=True,
            commanded_delta_v_mps=0.0,
        )
    if tca <= 0:
        raise ValueError("Cannot plan a future avoidance maneuver at or after TCA.")
    if miss_distance <= 1e-9:
        raise ValueError(
            "A miss direction is undefined at zero distance; a 3D optimizer is required."
        )

    miss_direction = miss_vector / miss_distance
    required_delta_v = (
        (miss_distance - target_miss_distance_m) / tca
    ) * miss_direction
    required_magnitude = float(np.linalg.norm(required_delta_v))
    feasible = max_delta_v_mps is None or required_magnitude <= max_delta_v_mps
    commanded_delta_v = required_delta_v.copy()
    if not feasible:
        commanded_delta_v *= max_delta_v_mps / required_magnitude

    predicted_vector = miss_vector - commanded_delta_v * tca
    return ManeuverPlan(
        delta_v_inertial_mps=commanded_delta_v,
        delta_v_rsw_mps=inertial_to_rsw(commanded_delta_v, state),
        predicted_miss_before_m=miss_distance,
        predicted_miss_after_linear_m=float(np.linalg.norm(predicted_vector)),
        target_miss_distance_m=target_miss_distance_m,
        required_delta_v_mps=required_magnitude,
        feasible=feasible,
        commanded_delta_v_mps=float(np.linalg.norm(commanded_delta_v)),
    )


def predict_minimum_miss_distance(
    spacecraft_state: np.ndarray,
    debris_state: np.ndarray,
    delta_v_inertial_mps: np.ndarray,
    prediction_horizon_s: float,
    propagation_step_s: float,
) -> float:
    spacecraft = np.asarray(spacecraft_state, dtype=float).copy()
    debris = np.asarray(debris_state, dtype=float).copy()
    delta_v = np.asarray(delta_v_inertial_mps, dtype=float)
    if spacecraft.shape != (6,) or not np.isfinite(spacecraft).all():
        raise ValueError("spacecraft_state must be a finite six-element vector.")
    if debris.shape != (6,) or not np.isfinite(debris).all():
        raise ValueError("debris_state must be a finite six-element vector.")
    if delta_v.shape != (3,) or not np.isfinite(delta_v).all():
        raise ValueError("delta_v_inertial_mps must be a finite three-element vector.")
    if not np.isfinite(prediction_horizon_s) or prediction_horizon_s < 0:
        raise ValueError("prediction_horizon_s must be finite and non-negative.")
    if not np.isfinite(propagation_step_s) or propagation_step_s <= 0:
        raise ValueError("propagation_step_s must be finite and positive.")
    number_of_steps = int(round(prediction_horizon_s / propagation_step_s))
    if not np.isclose(number_of_steps * propagation_step_s, prediction_horizon_s):
        raise ValueError("prediction_horizon_s must be divisible by propagation_step_s.")

    spacecraft[3:] += delta_v
    minimum_miss_distance = float(np.linalg.norm(debris[:3] - spacecraft[:3]))
    for _ in range(number_of_steps):
        spacecraft = rk4_step(spacecraft, propagation_step_s)
        debris = rk4_step(debris, propagation_step_s)
        minimum_miss_distance = min(
            minimum_miss_distance,
            float(np.linalg.norm(debris[:3] - spacecraft[:3])),
        )
    return minimum_miss_distance


def plan_nonlinear_avoidance_maneuver(
    spacecraft_state: np.ndarray,
    debris_state: np.ndarray,
    forecast: dict,
    target_miss_distance_m: float,
    prediction_horizon_s: float,
    propagation_step_s: float,
    max_delta_v_mps: float,
    search_samples: int = 101,
) -> ManeuverPlan:
    if not isinstance(search_samples, int) or search_samples < 3:
        raise ValueError("search_samples must be an integer of at least three.")
    if not np.isfinite(max_delta_v_mps) or max_delta_v_mps <= 0:
        raise ValueError("max_delta_v_mps must be finite and positive.")

    plan_avoidance_maneuver(
        spacecraft_state=spacecraft_state,
        forecast=forecast,
        target_miss_distance_m=target_miss_distance_m,
        max_delta_v_mps=None,
    )
    state = np.asarray(spacecraft_state, dtype=float)
    debris = np.asarray(debris_state, dtype=float)
    if debris.shape != (6,) or not np.isfinite(debris).all():
        raise ValueError("debris_state must be a finite six-element vector.")

    baseline_miss = predict_minimum_miss_distance(
        state,
        debris,
        np.zeros(3),
        prediction_horizon_s,
        propagation_step_s,
    )
    if baseline_miss >= target_miss_distance_m:
        zero = np.zeros(3)
        return ManeuverPlan(
            delta_v_inertial_mps=zero,
            delta_v_rsw_mps=zero.copy(),
            predicted_miss_before_m=float(forecast["miss_distance_m"]),
            predicted_miss_after_linear_m=float(forecast["miss_distance_m"]),
            target_miss_distance_m=target_miss_distance_m,
            required_delta_v_mps=0.0,
            feasible=True,
            predicted_miss_after_model_m=baseline_miss,
            planning_model="two-body grid search",
            commanded_delta_v_mps=0.0,
        )

    miss_distance = float(forecast["miss_distance_m"])
    if miss_distance <= 1e-9:
        raise ValueError("A miss direction is undefined at zero distance.")
    burn_direction = -np.asarray(forecast["miss_vector_m"], dtype=float) / miss_distance
    magnitudes = np.linspace(0.0, max_delta_v_mps, search_samples)
    predicted_distances = np.asarray([
        predict_minimum_miss_distance(
            state,
            debris,
            burn_direction * magnitude,
            prediction_horizon_s,
            propagation_step_s,
        )
        for magnitude in magnitudes
    ])

    candidate_indices = np.flatnonzero(
        predicted_distances >= target_miss_distance_m
    )
    if len(candidate_indices):
        upper_index = int(candidate_indices[0])
        lower = float(magnitudes[max(0, upper_index - 1)])
        upper = float(magnitudes[upper_index])
        for _ in range(20):
            midpoint = 0.5 * (lower + upper)
            predicted = predict_minimum_miss_distance(
                state,
                debris,
                burn_direction * midpoint,
                prediction_horizon_s,
                propagation_step_s,
            )
            if predicted >= target_miss_distance_m:
                upper = midpoint
            else:
                lower = midpoint
        commanded_magnitude = upper
        achieved_miss = predict_minimum_miss_distance(
            state,
            debris,
            burn_direction * commanded_magnitude,
            prediction_horizon_s,
            propagation_step_s,
        )
        feasible = True
    else:
        best_index = int(np.argmax(predicted_distances))
        commanded_magnitude = float(magnitudes[best_index])
        achieved_miss = float(predicted_distances[best_index])
        feasible = False
    estimated_range = float(np.linalg.norm(debris[:3] - state[:3]))
    if feasible:
        infeasibility_reason = None
    elif target_miss_distance_m > estimated_range:
        infeasibility_reason = (
            "Target exceeds estimated range at decision epoch; the future "
            "minimum includes the current separation."
        )
    else:
        infeasibility_reason = (
            "Target was not reached along the searched miss-direction impulse "
            "within the delta-v limit."
        )

    commanded_delta_v = burn_direction * commanded_magnitude
    predicted_linear_miss = float(np.linalg.norm(
        np.asarray(forecast["miss_vector_m"], dtype=float)
        - commanded_delta_v * float(forecast["tca_seconds"])
    ))
    return ManeuverPlan(
        delta_v_inertial_mps=commanded_delta_v,
        delta_v_rsw_mps=inertial_to_rsw(commanded_delta_v, state),
        predicted_miss_before_m=miss_distance,
        predicted_miss_after_linear_m=predicted_linear_miss,
        target_miss_distance_m=target_miss_distance_m,
        required_delta_v_mps=commanded_magnitude if feasible else None,
        feasible=feasible,
        predicted_miss_after_model_m=achieved_miss,
        planning_model="two-body grid search",
        infeasibility_reason=infeasibility_reason,
        commanded_delta_v_mps=commanded_magnitude,
    )


def thrust_rsw_for_delta_v(
    delta_v_rsw_mps: np.ndarray,
    spacecraft_mass_kg: float,
    specific_impulse_s: float,
    burn_duration_s: float,
) -> np.ndarray:
    delta_v = np.asarray(delta_v_rsw_mps, dtype=float)
    if delta_v.shape != (3,) or not np.isfinite(delta_v).all():
        raise ValueError("delta_v_rsw_mps must be a finite three-element vector.")
    if not np.isfinite(spacecraft_mass_kg) or spacecraft_mass_kg <= 0:
        raise ValueError("spacecraft_mass_kg must be finite and positive.")
    if not np.isfinite(specific_impulse_s) or specific_impulse_s <= 0:
        raise ValueError("specific_impulse_s must be finite and positive.")
    if not np.isfinite(burn_duration_s) or burn_duration_s <= 0:
        raise ValueError("burn_duration_s must be finite and positive.")

    delta_v_magnitude = float(np.linalg.norm(delta_v))
    if delta_v_magnitude == 0:
        return np.zeros(3)

    exhaust_velocity = specific_impulse_s * STANDARD_GRAVITY_MPS2
    thrust_magnitude = (
        spacecraft_mass_kg
        * exhaust_velocity
        / burn_duration_s
        * (-np.expm1(-delta_v_magnitude / exhaust_velocity))
    )
    return delta_v / delta_v_magnitude * thrust_magnitude
