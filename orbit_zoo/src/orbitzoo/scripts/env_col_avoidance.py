"""Fast sensor-limited single-spacecraft collision-avoidance RL experiment."""

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch


AGENT_NAME = "agent"
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
MU_EARTH_M3PS2 = 3.986004418e14
STEP_SECONDS = 10.0
HORIZON_SECONDS = 600.0
DEFAULT_STEPS_PER_EPISODE = int(HORIZON_SECONDS / STEP_SECONDS)
INTEGRATION_STEP_SECONDS = 2.0
MEASUREMENT_STD_M = 20.0
ACCELERATION_STD_MPS2 = 0.05
INITIAL_VELOCITY_STD_MPS = 2.0
COLLISION_RADIUS_M = 15.0
TARGET_MISS_DISTANCE_M = 100.0
MAX_DELTA_V_PER_STEP_MPS = 0.05
MAX_CONVENTIONAL_DELTA_V_MPS = 5.0
DRY_MASS_KG = 1.0
INITIAL_FUEL_KG = 4.0
SPECIFIC_IMPULSE_S = 1000.0
STANDARD_GRAVITY_MPS2 = 9.80665
OBSERVATION_SIZE = 10


def _encounter_states(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create distinct, bounded close-approach states around a common orbit."""
    relative_velocity = rng.normal(size=3)
    relative_velocity_norm = float(np.linalg.norm(relative_velocity))
    if relative_velocity_norm < 1e-12:
        relative_velocity = np.array([1.0, 0.0, 0.0])
        relative_velocity_norm = 1.0
    relative_velocity *= rng.uniform(1.0, 3.0) / relative_velocity_norm

    perpendicular = rng.normal(size=3)
    perpendicular -= (
        perpendicular @ relative_velocity
        / (relative_velocity @ relative_velocity)
    ) * relative_velocity
    perpendicular_norm = float(np.linalg.norm(perpendicular))
    if perpendicular_norm < 1e-8:
        perpendicular = np.cross(relative_velocity, [1.0, 0.0, 0.0])
        perpendicular_norm = float(np.linalg.norm(perpendicular))
    perpendicular /= perpendicular_norm

    time_to_closest_approach = rng.uniform(120.0, HORIZON_SECONDS)
    impact_parameter = rng.uniform(0.0, 120.0)
    relative_position = (
        -relative_velocity * time_to_closest_approach
        + perpendicular * impact_parameter
    )
    spacecraft_state = np.concatenate([
        INITIAL_POSITION_M,
        INITIAL_VELOCITY_MPS,
    ])
    debris_state = np.concatenate([
        spacecraft_state[:3] + relative_position,
        spacecraft_state[3:] + relative_velocity,
    ])
    return spacecraft_state, debris_state


def _two_body_derivative(state: np.ndarray) -> np.ndarray:
    position = state[:3]
    radius = float(np.linalg.norm(position))
    if not np.isfinite(radius) or radius < 1.0:
        raise ValueError("Orbital state has an invalid radius.")
    acceleration = -MU_EARTH_M3PS2 * position / radius**3
    return np.concatenate([state[3:], acceleration])


def _rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = _two_body_derivative(state)
    k2 = _two_body_derivative(state + 0.5 * dt * k1)
    k3 = _two_body_derivative(state + 0.5 * dt * k2)
    k4 = _two_body_derivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _transition_matrix(dt: float) -> np.ndarray:
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt
    return transition


def _process_noise(dt: float, acceleration_std: float) -> np.ndarray:
    covariance = np.zeros((6, 6), dtype=float)
    block = acceleration_std**2 * np.array([
        [dt**4 / 4.0, dt**3 / 2.0],
        [dt**3 / 2.0, dt**2],
    ])
    for axis in range(3):
        covariance[axis, axis] = block[0, 0]
        covariance[axis, axis + 3] = block[0, 1]
        covariance[axis + 3, axis] = block[1, 0]
        covariance[axis + 3, axis + 3] = block[1, 1]
    return covariance


def _measurement_update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    measurement_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    observation_matrix = np.zeros((3, 6))
    observation_matrix[:, :3] = np.eye(3)
    measurement_covariance = np.eye(3) * measurement_std**2
    innovation_covariance = (
        observation_matrix @ covariance @ observation_matrix.T
        + measurement_covariance
    )
    kalman_gain = np.linalg.solve(
        innovation_covariance,
        observation_matrix @ covariance,
    ).T
    updated_state = state + kalman_gain @ (
        measurement - observation_matrix @ state
    )
    residual_transform = np.eye(6) - kalman_gain @ observation_matrix
    updated_covariance = (
        residual_transform @ covariance @ residual_transform.T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return updated_state, updated_covariance


def _rsw_basis(spacecraft_state: np.ndarray) -> np.ndarray:
    radial = spacecraft_state[:3] / np.linalg.norm(spacecraft_state[:3])
    normal = np.cross(spacecraft_state[:3], spacecraft_state[3:])
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        raise ValueError("Spacecraft state does not define an orbital plane.")
    normal /= normal_norm
    along_track = np.cross(normal, radial)
    return np.column_stack((radial, along_track, normal))


class ColAvoidanceEnv:
    """Fast two-body simulator exposing only noisy, filtered relative state."""

    def __init__(
        self,
        step_size: float = STEP_SECONDS,
        episode_steps: int = DEFAULT_STEPS_PER_EPISODE,
        measurement_std_m: float = MEASUREMENT_STD_M,
    ):
        if not np.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive.")
        if not isinstance(episode_steps, int) or episode_steps <= 0:
            raise ValueError("episode_steps must be a positive integer.")
        if not np.isfinite(measurement_std_m) or measurement_std_m <= 0:
            raise ValueError("measurement_std_m must be finite and positive.")
        self.step_size = float(step_size)
        self.episode_steps = episode_steps
        self.horizon_seconds = self.step_size * episode_steps
        self.measurement_std_m = float(measurement_std_m)
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        self.spacecraft_state, self.debris_state = _encounter_states(self._rng)
        self._mass_kg = DRY_MASS_KG + INITIAL_FUEL_KG
        self._steps = 0
        self._collided = False
        self._minimum_miss_distance_m = float(np.linalg.norm(
            self.debris_state[:3] - self.spacecraft_state[:3]
        ))
        self._estimate = np.concatenate([
            self._measure_relative_position(),
            np.zeros(3),
        ])
        self._estimate_covariance = np.diag([
            self.measurement_std_m**2,
            self.measurement_std_m**2,
            self.measurement_std_m**2,
            INITIAL_VELOCITY_STD_MPS**2,
            INITIAL_VELOCITY_STD_MPS**2,
            INITIAL_VELOCITY_STD_MPS**2,
        ])
        observation = self.observations()
        self._previous_observation = observation.copy()
        return observation

    def _true_relative_state(self) -> np.ndarray:
        return np.concatenate([
            self.debris_state[:3] - self.spacecraft_state[:3],
            self.debris_state[3:] - self.spacecraft_state[3:],
        ])

    def _measure_relative_position(self) -> np.ndarray:
        return self._true_relative_state()[:3] + self._rng.normal(
            0.0,
            self.measurement_std_m,
            size=3,
        )

    def _estimated_closest_approach(self) -> tuple[float, float]:
        position = self._estimate[:3]
        velocity = self._estimate[3:]
        speed_squared = float(velocity @ velocity)
        remaining_seconds = max(
            0.0,
            (self.episode_steps - self._steps) * self.step_size,
        )
        if speed_squared < 1e-12:
            tca = 0.0
        else:
            tca = float(np.clip(
                -(position @ velocity) / speed_squared,
                0.0,
                remaining_seconds,
            ))
        miss_distance = float(np.linalg.norm(position + velocity * tca))
        return tca, miss_distance

    def observations(self) -> np.ndarray:
        tca, miss_distance = self._estimated_closest_approach()
        position_sigma = float(np.sqrt(max(
            np.linalg.eigvalsh(self._estimate_covariance[:3, :3]).max(),
            0.0,
        )))
        observation = np.concatenate([
            self._estimate[:3] / 1000.0,
            self._estimate[3:] / 5.0,
            [miss_distance / 1000.0],
            [tca / self.horizon_seconds],
            [position_sigma / 100.0],
            [self.fuel_remaining_kg / INITIAL_FUEL_KG],
        ])
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise RuntimeError("The collision-avoidance observation is invalid.")
        return observation.astype(np.float32)

    @property
    def fuel_remaining_kg(self) -> float:
        return max(0.0, self._mass_kg - DRY_MASS_KG)

    def _apply_delta_v_rsw(
        self,
        delta_v_rsw: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        magnitude = float(np.linalg.norm(delta_v_rsw))
        if magnitude == 0:
            return 0.0, np.zeros(3)
        exhaust_velocity = SPECIFIC_IMPULSE_S * STANDARD_GRAVITY_MPS2
        available_delta_v = exhaust_velocity * math.log(
            self._mass_kg / DRY_MASS_KG
        )
        if magnitude > available_delta_v:
            delta_v_rsw = delta_v_rsw * (available_delta_v / magnitude)
            magnitude = available_delta_v
        mass_after_burn = self._mass_kg * math.exp(
            -magnitude / exhaust_velocity
        )
        delta_v_inertial = _rsw_basis(self.spacecraft_state) @ delta_v_rsw
        self.spacecraft_state[3:] += delta_v_inertial
        fuel_used = self._mass_kg - mass_after_burn
        self._mass_kg = max(DRY_MASS_KG, mass_after_burn)
        return float(fuel_used), delta_v_inertial

    def step(
        self,
        action: np.ndarray,
        delta_v_override_rsw: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool, dict[str, float | bool]]:
        policy_action = np.asarray(action, dtype=float)
        if policy_action.shape != (3,) or not np.isfinite(policy_action).all():
            raise ValueError("Each policy action must be a finite three-element vector.")
        if np.any(np.abs(policy_action) > 1.0 + 1e-8):
            raise ValueError("Policy action components must be in [-1, 1].")
        if delta_v_override_rsw is None:
            action_norm = float(np.linalg.norm(policy_action))
            bounded_action = (
                policy_action / action_norm if action_norm > 1.0 else policy_action
            )
            delta_v_rsw = bounded_action * MAX_DELTA_V_PER_STEP_MPS
        else:
            delta_v_rsw = np.asarray(delta_v_override_rsw, dtype=float)
            if delta_v_rsw.shape != (3,) or not np.isfinite(delta_v_rsw).all():
                raise ValueError("delta_v_override_rsw must be a finite three-vector.")
            if np.linalg.norm(delta_v_rsw) > MAX_CONVENTIONAL_DELTA_V_MPS + 1e-8:
                raise ValueError("Conventional impulse exceeds the 5 m/s maneuver cap.")

        previous_relative_position = self._true_relative_state()[:3]
        fuel_used, delta_v_inertial = self._apply_delta_v_rsw(delta_v_rsw)
        number_of_substeps = max(
            1,
            int(math.ceil(self.step_size / INTEGRATION_STEP_SECONDS)),
        )
        substep = self.step_size / number_of_substeps
        for _ in range(number_of_substeps):
            self.spacecraft_state = _rk4_step(self.spacecraft_state, substep)
            self.debris_state = _rk4_step(self.debris_state, substep)

        transition = _transition_matrix(self.step_size)
        self._estimate = transition @ self._estimate
        self._estimate[3:] -= delta_v_inertial
        self._estimate_covariance = (
            transition @ self._estimate_covariance @ transition.T
            + _process_noise(self.step_size, ACCELERATION_STD_MPS2)
        )
        self._estimate, self._estimate_covariance = _measurement_update(
            self._estimate,
            self._estimate_covariance,
            self._measure_relative_position(),
            self.measurement_std_m,
        )

        relative_position = self._true_relative_state()[:3]
        segment = relative_position - previous_relative_position
        segment_length_squared = float(segment @ segment)
        segment_fraction = 0.0
        if segment_length_squared > 1e-12:
            segment_fraction = float(np.clip(
                -(previous_relative_position @ segment) / segment_length_squared,
                0.0,
                1.0,
            ))
        segment_minimum = float(np.linalg.norm(
            previous_relative_position + segment_fraction * segment
        ))
        self._minimum_miss_distance_m = min(
            self._minimum_miss_distance_m,
            segment_minimum,
        )
        self._collided = segment_minimum <= COLLISION_RADIUS_M
        self._steps += 1

        observation = self.observations()
        previous_miss = float(self._previous_observation[6]) * 1000.0
        current_miss = float(observation[6]) * 1000.0
        previous_potential = np.clip(
            (previous_miss - COLLISION_RADIUS_M)
            / (TARGET_MISS_DISTANCE_M - COLLISION_RADIUS_M),
            0.0,
            1.0,
        )
        current_potential = np.clip(
            (current_miss - COLLISION_RADIUS_M)
            / (TARGET_MISS_DISTANCE_M - COLLISION_RADIUS_M),
            0.0,
            1.0,
        )
        reward = (
            0.1 * (current_potential - previous_potential)
            - 0.002
            - 0.01 * min(float(np.linalg.norm(policy_action)), 1.0) ** 2
        )
        terminated = self._collided or self._steps >= self.episode_steps
        if self._collided:
            reward -= 10.0
        elif self._steps >= self.episode_steps:
            safe_fraction = np.clip(
                (self._minimum_miss_distance_m - COLLISION_RADIUS_M)
                / (TARGET_MISS_DISTANCE_M - COLLISION_RADIUS_M),
                0.0,
                1.0,
            )
            reward += 5.0 * float(safe_fraction)

        self._previous_observation = observation.copy()
        info = {
            "minimum_miss_distance_m": self._minimum_miss_distance_m,
            "fuel_used_kg": fuel_used,
            "collision": self._collided,
            "success": (
                terminated
                and self._minimum_miss_distance_m >= TARGET_MISS_DISTANCE_M
            ),
        }
        return observation, float(reward), terminated, info


def _inertial_to_rsw(vector: np.ndarray, spacecraft_state: np.ndarray) -> np.ndarray:
    return _rsw_basis(spacecraft_state).T @ vector


def _conventional_action(env: ColAvoidanceEnv) -> np.ndarray:
    """Return one bounded linear-planner impulse in the spacecraft RSW frame."""
    position = env._estimate[:3]
    velocity = env._estimate[3:]
    speed_squared = float(velocity @ velocity)
    if speed_squared < 1e-12:
        return np.zeros(3)
    tca = float(np.clip(
        -(position @ velocity) / speed_squared,
        0.0,
        env.horizon_seconds,
    ))
    if tca <= 0:
        return np.zeros(3)
    miss_vector = position + velocity * tca
    miss_distance = float(np.linalg.norm(miss_vector))
    if miss_distance >= TARGET_MISS_DISTANCE_M or miss_distance <= 1e-9:
        return np.zeros(3)
    delta_v_inertial = (
        (miss_distance - TARGET_MISS_DISTANCE_M)
        / tca
        * miss_vector
        / miss_distance
    )
    magnitude = float(np.linalg.norm(delta_v_inertial))
    if magnitude > MAX_CONVENTIONAL_DELTA_V_MPS:
        delta_v_inertial *= MAX_CONVENTIONAL_DELTA_V_MPS / magnitude
    return _inertial_to_rsw(delta_v_inertial, env.spacecraft_state)


def _run_episode(
    env: ColAvoidanceEnv,
    seed: int,
    strategy: str,
    policy=None,
) -> dict[str, float]:
    observation = env.reset(seed)
    conventional_impulse = (
        _conventional_action(env) if strategy == "conventional" else None
    )
    for step in range(env.episode_steps):
        if strategy == "policy":
            action = np.asarray(policy(observation), dtype=float)
            observation, _, terminated, _ = env.step(action)
        elif strategy == "conventional" and step == 0:
            observation, _, terminated, _ = env.step(
                np.zeros(3),
                delta_v_override_rsw=conventional_impulse,
            )
        else:
            observation, _, terminated, _ = env.step(np.zeros(3))
        if terminated:
            break
    return {
        "minimum_miss_distance_m": env._minimum_miss_distance_m,
        "fuel_used_kg": INITIAL_FUEL_KG - env.fuel_remaining_kg,
        "collision": float(env._collided),
        "success": float(
            env._minimum_miss_distance_m >= TARGET_MISS_DISTANCE_M
        ),
    }


def evaluate_strategies(
    env: ColAvoidanceEnv,
    episodes: int,
    seed: int,
    policy=None,
) -> dict[str, dict[str, float]]:
    strategies = ["no_maneuver", "conventional"]
    if policy is not None:
        strategies.append("policy")
    results = {}
    for strategy in strategies:
        records = [
            _run_episode(env, seed + episode, strategy, policy=policy)
            for episode in range(episodes)
        ]
        results[strategy] = {
            "episodes": float(episodes),
            "success_rate": float(np.mean([item["success"] for item in records])),
            "collision_rate": float(np.mean([item["collision"] for item in records])),
            "mean_minimum_miss_distance_m": float(np.mean([
                item["minimum_miss_distance_m"] for item in records
            ])),
            "mean_fuel_used_kg": float(np.mean([
                item["fuel_used_kg"] for item in records
            ])),
        }
    return results


def _new_ppo_policy(
    *,
    device: torch.device,
    learn_epsilon: bool = False,
):
    from orbitzoo.rl_algorithms.ppo import PPO

    return PPO(
        device=device,
        has_continuous_action_space=True,
        action_space=[1.0, 1.0, 1.0],
        state_dim=OBSERVATION_SIZE,
        action_dim=3,
        action_to_thrust_fn=None,
        lr_actor=1e-4,
        lr_critic=1e-3,
        K_epochs=1,
        epsilon=0.2,
        learn_epsilon=learn_epsilon,
    )


def train_policy(
    env: ColAvoidanceEnv,
    *,
    episodes: int,
    seed: int,
    save_path: Path,
    metrics_path: str | None = None,
) -> float:
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer.")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    algorithm = _new_ppo_policy(device=device)
    save_path.mkdir(parents=True, exist_ok=True)

    writer = None
    if metrics_path:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(log_dir=str(Path("runs") / metrics_path))

    best_return = -float("inf")
    episode_return_sum = 0.0
    try:
        for episode in range(episodes):
            observation = env.reset(seed + episode)
            episode_return = 0.0
            for _ in range(env.episode_steps):
                action = algorithm.select_action(observation)
                action = np.asarray(action, dtype=float)
                next_observation, reward, terminated, _ = env.step(action)
                algorithm.memory.add((
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                ))
                episode_return += reward
                observation = next_observation
                if terminated:
                    break

            if algorithm.memory.rewards:
                algorithm.update()
            episode_return_sum += episode_return
            if episode_return > best_return:
                best_return = episode_return
                algorithm.save(str(save_path / "agent.pt"))
            if (episode + 1) % 50 == 0:
                algorithm.save(str(save_path / "agent_latest.pt"))
            if writer is not None:
                writer.add_scalar("Metrics/reward", episode_return, episode + 1)
                writer.add_scalar(
                    "Metrics/minimum_miss_distance_m",
                    env._minimum_miss_distance_m,
                    episode + 1,
                )
                writer.add_scalar(
                    "Metrics/fuel_remaining_kg",
                    env.fuel_remaining_kg,
                    episode + 1,
                )
            print(
                f"episode {episode + 1}: reward={episode_return:.4f}, "
                f"minimum_miss={env._minimum_miss_distance_m:.2f} m, "
                f"fuel={env.fuel_remaining_kg:.4f} kg"
            )
    finally:
        if writer is not None:
            writer.close()
    return episode_return_sum / episodes


def _load_policy(model_path: Path):
    device = torch.device("cpu")
    policy = _new_ppo_policy(device=device)
    policy.load(str(model_path))
    return lambda observation: policy.select_action(
        observation,
        deterministic=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=DEFAULT_STEPS_PER_EPISODE,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "trained_models"
            / "collision_avoidance"
        ),
    )
    parser.add_argument("--metrics-path")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    if args.episodes <= 0 or args.steps_per_episode <= 0:
        parser.error("--episodes and --steps-per-episode must be positive.")
    if args.mode == "evaluate" and args.model_path is None:
        parser.error("--model-path is required with --mode evaluate.")
    if args.smoke_test and args.mode != "train":
        parser.error("--smoke-test is available only in train mode.")

    if args.smoke_test:
        args.episodes = 1
        args.steps_per_episode = 2
        args.metrics_path = None

    env = ColAvoidanceEnv(episode_steps=args.steps_per_episode)
    if args.mode == "train":
        if args.smoke_test:
            with tempfile.TemporaryDirectory(prefix="orbitzoo-rl-smoke-") as directory:
                average_reward = train_policy(
                    env,
                    episodes=args.episodes,
                    seed=args.seed,
                    save_path=Path(directory),
                )
            print(f"Smoke training completed; mean episode reward={average_reward:.4f}")
        else:
            average_reward = train_policy(
                env,
                episodes=args.episodes,
                seed=args.seed,
                save_path=args.save_path,
                metrics_path=args.metrics_path,
            )
            print(f"Training completed; mean episode reward={average_reward:.4f}")
    else:
        if not args.model_path.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {args.model_path}")
        policy = _load_policy(args.model_path)
        results = evaluate_strategies(
            env,
            episodes=args.episodes,
            seed=args.seed,
            policy=policy,
        )
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
