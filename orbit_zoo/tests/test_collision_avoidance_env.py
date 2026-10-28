import subprocess
import sys

import numpy as np
import pytest

import orbitzoo.scripts.env_col_avoidance as collision_avoidance
from orbitzoo.scripts.env_col_avoidance import (
    MAX_DELTA_V_PER_STEP_MPS,
    ColAvoidanceEnv,
    _encounter_states,
    _measurement_update,
    train_policy,
)


def test_importing_orbitzoo_does_not_start_the_orekit_environment():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, orbitzoo; "
            "assert 'orbitzoo.env' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_encounter_generator_produces_distinct_approaching_scenarios():
    first_spacecraft, first_debris = _encounter_states(np.random.default_rng(7))
    second_spacecraft, second_debris = _encounter_states(np.random.default_rng(8))

    assert first_spacecraft.shape == (6,)
    assert first_debris.shape == (6,)
    assert not np.array_equal(first_debris, second_debris)

    relative_position = first_debris[:3] - first_spacecraft[:3]
    relative_velocity = first_debris[3:] - first_spacecraft[3:]
    assert relative_position @ relative_velocity < 0


def test_measurement_update_returns_finite_symmetric_covariance():
    state = np.array([100.0, -20.0, 5.0, -1.0, 0.2, 0.0])
    covariance = np.diag([400.0] * 3 + [4.0] * 3)
    measurement = np.array([90.0, -18.0, 3.0])

    updated_state, updated_covariance = _measurement_update(
        state,
        covariance,
        measurement,
        measurement_std=20.0,
    )

    assert np.isfinite(updated_state).all()
    assert np.isfinite(updated_covariance).all()
    assert np.allclose(updated_covariance, updated_covariance.T)
    assert np.linalg.eigvalsh(updated_covariance).min() >= -1e-10


@pytest.fixture(scope="module")
def env():
    return ColAvoidanceEnv(episode_steps=1)


def test_reset_is_reproducible_and_observation_is_sensor_limited(env):
    first = env.reset(seed=123)
    first_relative_truth = env._true_relative_state()
    second = env.reset(seed=123)
    second_relative_truth = env._true_relative_state()

    assert np.array_equal(first, second)
    assert np.allclose(first_relative_truth, second_relative_truth)
    assert first.shape == (10,)
    assert np.isfinite(first).all()
    assert not np.allclose(first[:6], np.concatenate([
        second_relative_truth[:3] / 1000.0,
        second_relative_truth[3:] / 5.0,
    ]))


def test_action_is_bounded_and_episode_returns_terminal_reward(env, monkeypatch):
    env.reset(seed=42)
    applied_delta_v = []
    apply_delta_v = env._apply_delta_v_rsw

    def record_delta_v(delta_v):
        applied_delta_v.append(delta_v.copy())
        return apply_delta_v(delta_v)

    monkeypatch.setattr(env, "_apply_delta_v_rsw", record_delta_v)
    observation, reward, terminated, info = env.step(np.array([1.0, 1.0, 1.0]))

    assert observation.shape == (10,)
    assert np.isfinite(observation).all()
    assert np.isfinite(reward)
    assert terminated
    assert "minimum_miss_distance_m" in info
    assert 0.0 <= env.fuel_remaining_kg < 4.0
    assert np.linalg.norm(applied_delta_v[0]) == pytest.approx(
        MAX_DELTA_V_PER_STEP_MPS
    )


def test_filter_accounts_for_known_maneuver(monkeypatch, env):
    env.reset(seed=17)
    initial_estimate = env._estimate.copy()
    initial_state = env.spacecraft_state.copy()
    delta_v_rsw = np.array([MAX_DELTA_V_PER_STEP_MPS, 0.0, 0.0])
    expected_delta_v_inertial = (
        collision_avoidance._rsw_basis(initial_state) @ delta_v_rsw
    )
    monkeypatch.setattr(
        collision_avoidance,
        "_measurement_update",
        lambda state, covariance, measurement, measurement_std: (
            state,
            covariance,
        ),
    )

    env.step(np.array([1.0, 0.0, 0.0]))

    assert np.allclose(
        env._estimate[3:],
        initial_estimate[3:] - expected_delta_v_inertial,
    )


def test_segment_collision_detection_catches_crossing_between_steps(env):
    env.reset(seed=42)
    env.debris_state[:3] = env.spacecraft_state[:3] + [-20.0, 0.0, 0.0]
    env.debris_state[3:] = env.spacecraft_state[3:] + [4.0, 0.0, 0.0]
    env._minimum_miss_distance_m = 20.0

    _, reward, terminated, info = env.step(np.zeros(3))

    assert terminated
    assert info["collision"]
    assert info["minimum_miss_distance_m"] <= 15.0
    assert reward < -9.0


@pytest.mark.parametrize("invalid_action", ([1.0, 2.0], [1.0, 2.0, float("nan")]))
def test_invalid_policy_actions_are_rejected(env, invalid_action):
    env.reset(seed=0)
    with pytest.raises(ValueError, match="finite three-element vector"):
        env.step(invalid_action)


def test_policy_training_smoke_saves_a_checkpoint(tmp_path):
    environment = ColAvoidanceEnv(episode_steps=2)
    mean_reward = train_policy(
        environment,
        episodes=1,
        seed=7,
        save_path=tmp_path,
    )

    assert np.isfinite(mean_reward)
    assert (tmp_path / "agent.pt").is_file()


def test_maximum_policy_impulse_is_small_and_bounded():
    assert MAX_DELTA_V_PER_STEP_MPS == pytest.approx(0.05)
