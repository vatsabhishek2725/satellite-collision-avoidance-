import numpy as np

from avoidance_controller import (
    inertial_to_rsw,
    plan_avoidance_maneuver,
    plan_nonlinear_avoidance_maneuver,
    predict_minimum_miss_distance,
    thrust_rsw_for_delta_v,
)


SPACECRAFT_STATE = np.array([
    7000000.0,
    0.0,
    0.0,
    0.0,
    7500.0,
    0.0,
])


def _forecast(miss_vector, tca=50.0):
    miss_vector = np.asarray(miss_vector, dtype=float)
    return {
        "tca_seconds": tca,
        "miss_distance_m": float(np.linalg.norm(miss_vector)),
        "miss_vector_m": miss_vector,
    }


def test_controller_does_not_maneuver_when_target_is_met():
    plan = plan_avoidance_maneuver(
        SPACECRAFT_STATE,
        _forecast([150.0, 0.0, 0.0]),
        target_miss_distance_m=100.0,
    )

    assert plan.feasible
    assert plan.required_delta_v_mps == 0.0
    assert np.array_equal(plan.delta_v_inertial_mps, np.zeros(3))


def test_inertial_to_rsw_uses_radial_along_track_cross_track_axes():
    converted = inertial_to_rsw(np.array([1.0, 2.0, 3.0]), SPACECRAFT_STATE)

    assert np.allclose(converted, [1.0, 2.0, 3.0])


def test_inertial_to_rsw_rejects_zero_position():
    with np.testing.assert_raises_regex(ValueError, "position cannot be zero"):
        inertial_to_rsw(np.ones(3), np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_controller_plans_delta_v_for_target_miss_distance():
    plan = plan_avoidance_maneuver(
        SPACECRAFT_STATE,
        _forecast([50.0, 0.0, 0.0]),
        target_miss_distance_m=150.0,
    )

    assert plan.feasible
    assert np.isclose(plan.predicted_miss_after_linear_m, 150.0)
    assert np.isclose(plan.required_delta_v_mps, 2.0)


def test_insufficient_delta_v_is_flagged_and_capped():
    plan = plan_avoidance_maneuver(
        SPACECRAFT_STATE,
        _forecast([50.0, 0.0, 0.0]),
        target_miss_distance_m=150.0,
        max_delta_v_mps=0.5,
    )

    assert not plan.feasible
    assert np.isclose(np.linalg.norm(plan.delta_v_inertial_mps), 0.5)
    assert plan.predicted_miss_after_linear_m < plan.target_miss_distance_m


def test_two_body_prediction_includes_decision_epoch_range():
    debris_state = SPACECRAFT_STATE.copy()
    debris_state[1] += 50.0
    predicted = predict_minimum_miss_distance(
        SPACECRAFT_STATE,
        debris_state,
        np.array([0.0, 0.0, 0.0]),
        prediction_horizon_s=0.0,
        propagation_step_s=2.0,
    )

    assert predicted == 50.0


def test_nonlinear_controller_flags_target_above_current_range_unreachable():
    debris_state = SPACECRAFT_STATE.copy()
    debris_state[1] += 50.0
    plan = plan_nonlinear_avoidance_maneuver(
        spacecraft_state=SPACECRAFT_STATE,
        debris_state=debris_state,
        forecast=_forecast([10.0, 0.0, 0.0], tca=10.0),
        target_miss_distance_m=60.0,
        prediction_horizon_s=10.0,
        propagation_step_s=2.0,
        max_delta_v_mps=5.0,
        search_samples=5,
    )

    assert not plan.feasible
    assert plan.required_delta_v_mps is None
    assert 0.0 < plan.commanded_delta_v_mps <= 5.0
    assert plan.predicted_miss_after_model_m <= 50.0
    assert "exceeds estimated range" in plan.infeasibility_reason


def test_adverse_burn_reduces_miss_distance():
    nominal = _forecast([100.0, 0.0, 0.0])
    plan = plan_avoidance_maneuver(
        SPACECRAFT_STATE,
        nominal,
        target_miss_distance_m=200.0,
    )
    adverse_relative_position = (
        nominal["miss_vector_m"] + plan.delta_v_inertial_mps * nominal["tca_seconds"]
    )

    assert np.linalg.norm(adverse_relative_position) < nominal["miss_distance_m"]


def test_thrust_conversion_matches_rocket_equation_impulse():
    desired_delta_v = np.array([0.0, 0.2, 0.0])
    mass = 5.0
    isp = 1000.0
    duration = 2.0
    thrust = thrust_rsw_for_delta_v(desired_delta_v, mass, isp, duration)
    force_magnitude = np.linalg.norm(thrust)
    final_mass = mass - force_magnitude * duration / (isp * 9.80665)
    achieved_delta_v = isp * 9.80665 * np.log(mass / final_mass)

    assert np.isclose(achieved_delta_v, np.linalg.norm(desired_delta_v))
    assert np.allclose(thrust / force_magnitude, desired_delta_v / 0.2)
