import numpy as np
import pytest

from orbital_mechanics import (
    fit_initial_orbit_from_relative_positions,
    rk4_step,
)


def _sensor_only_sample():
    dt = 2.0
    spacecraft_state = np.array([
        -3529923.947865602,
        7042905.715845195,
        0.0,
        6359.116737768876,
        3187.207008809081,
        0.0,
    ])
    debris_state = spacecraft_state + np.array(
        [0.0, 1000.0, 0.0, 0.0, -2.0, 0.0]
    )
    sample_steps = np.arange(0, 101, 5)
    spacecraft_samples = []
    relative_positions = []

    for step in range(101):
        if step % 5 == 0:
            spacecraft_samples.append(spacecraft_state.copy())
            relative_positions.append(
                debris_state[:3] - spacecraft_state[:3]
            )
        if step < 100:
            spacecraft_state = rk4_step(spacecraft_state, dt)
            debris_state = rk4_step(debris_state, dt)

    noise = np.random.default_rng(42).normal(
        0.0, 20.0, size=(len(sample_steps), 3)
    )
    return (
        np.asarray(relative_positions) + noise,
        np.asarray(spacecraft_samples),
        sample_steps,
        dt,
    )


def test_sensor_only_fit_recovers_state_and_psd_covariance():
    measurements, spacecraft, sample_steps, dt = _sensor_only_sample()

    fit = fit_initial_orbit_from_relative_positions(
        measurements,
        spacecraft,
        sample_steps,
        dt,
        measurement_std=20.0,
    )

    true_state = np.array([
        -3529923.947865602,
        7043905.715845195,
        0.0,
        6359.116737768876,
        3185.207008809081,
        0.0,
    ])
    assert np.linalg.norm(fit.state[:3] - true_state[:3]) < 5.0
    assert np.linalg.norm(fit.state[3:] - true_state[3:]) < 0.06
    assert fit.covariance.shape == (6, 6)
    assert np.allclose(fit.covariance, fit.covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(fit.covariance).min() >= -1e-9
    assert fit.measurement_count == len(sample_steps)


def test_sensor_only_fit_rejects_too_few_measurements():
    with pytest.raises(ValueError, match="At least three measurements"):
        fit_initial_orbit_from_relative_positions(
            np.zeros((2, 3)),
            np.zeros((2, 6)),
            np.array([0, 5]),
            dt=2.0,
            measurement_std=20.0,
        )


def test_sensor_only_fit_rejects_non_increasing_sample_times():
    measurements, spacecraft, sample_steps, dt = _sensor_only_sample()
    sample_steps[1] = sample_steps[0]

    with pytest.raises(ValueError, match="strictly increasing"):
        fit_initial_orbit_from_relative_positions(
            measurements,
            spacecraft,
            sample_steps,
            dt,
            measurement_std=20.0,
        )
