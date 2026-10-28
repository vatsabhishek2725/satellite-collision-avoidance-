
import numpy as np
from orbitzoo import OrbitZoo
import os

# ============================================================
# 1. CONFIGURATION
# ============================================================

MU_EARTH = 3.986004418e14  # m^3/s^2

DT = 2.0
TOTAL_TIME = 600.0
SENSOR_INTERVAL = 10.0

# MEASUREMENT_STD = 20.0
MEASUREMENT_STD = float(
    os.environ.get("ORBITAL_EKF_MEASUREMENT_STD", "20.0")
)

INITIAL_VELOCITY_STD = 5.0

# Tunable acceleration-noise parameter, not a measured value.
ACCELERATION_NOISE_STD = 0.01

# rng = np.random.default_rng(seed=42)
SEED = int(os.environ.get("ORBITAL_EKF_SEED", "42"))
rng = np.random.default_rng(seed=SEED)

TOTAL_STEPS = int(TOTAL_TIME / DT)
SENSOR_STEPS = int(SENSOR_INTERVAL / DT)


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
# 3. TWO-BODY ORBITAL DYNAMICS
# ============================================================

# State x = [x, y, z, vx, vy, vz] in inertial coordinates.

def orbital_dynamics(x):
    r = x[:3]
    v = x[3:]

    radius = np.linalg.norm(r)

    if radius < 1.0:
        raise ValueError("Invalid orbital radius.")

    acceleration = -MU_EARTH * r / radius**3

    return np.concatenate([v, acceleration])


def rk4_step(x, dt):
    k1 = orbital_dynamics(x)
    k2 = orbital_dynamics(x + 0.5 * dt * k1)
    k3 = orbital_dynamics(x + 0.5 * dt * k2)
    k4 = orbital_dynamics(x + dt * k3)

    return x + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )


def numerical_jacobian(x, dt):
    # Central finite-difference approximation of the
    # Jacobian of the discrete RK4 propagation function.
    J = np.zeros((6, 6))

    for i in range(6):
        epsilon = 1e-3 if i < 3 else 1e-5

        delta = np.zeros(6)
        delta[i] = epsilon

        x_plus = rk4_step(x + delta, dt)
        x_minus = rk4_step(x - delta, dt)

        J[:, i] = (x_plus - x_minus) / (2.0 * epsilon)

    return J


# ============================================================
# 4. COVARIANCE AND MEASUREMENT MODEL
# ============================================================

# Sensor reports relative position:
# z_relative = r_debris - r_spacecraft + noise
#
# Convert it to an inertial debris-position measurement
# using the known spacecraft position.

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
    r_sc = np.asarray(spacecraft.position, dtype=float)
    r_debris = np.asarray(debris.position, dtype=float)

    v_debris = np.asarray(debris.velocity, dtype=float)

    return r_sc, np.concatenate([r_debris, v_debris])


def simulate_relative_measurement(r_sc, debris_state):
    true_relative_position = debris_state[:3] - r_sc

    return true_relative_position + rng.normal(
        0.0, MEASUREMENT_STD, size=3
    )


# ============================================================
# 6. INITIALIZE FILTER FROM NOISY MEASUREMENT
# ============================================================

r_sc, truth = get_truth()

first_relative_measurement = simulate_relative_measurement(
    r_sc, truth
)

# Convert the noisy relative measurement into an absolute
# debris-position estimate using the known spacecraft position.
x = np.concatenate([
    r_sc + first_relative_measurement,
    np.asarray(spacecraft.velocity, dtype=float)
])

P = np.diag([
    MEASUREMENT_STD**2,
    MEASUREMENT_STD**2,
    MEASUREMENT_STD**2,
    INITIAL_VELOCITY_STD**2,
    INITIAL_VELOCITY_STD**2,
    INITIAL_VELOCITY_STD**2
])


# ============================================================
# 7. RUN ORBITAL EKF
# ============================================================

position_errors = []
velocity_errors = []
nis_values = []
coverage_1sigma = []
coverage_2sigma = []

for step in range(TOTAL_STEPS + 1):

    # Evaluate at the current simulation time.
    r_sc, truth = get_truth()

    if step % SENSOR_STEPS == 0:

        z_relative = simulate_relative_measurement(r_sc, truth)
        z_absolute = r_sc + z_relative

        innovation = z_absolute - H @ x
        S = H @ P @ H.T + R

        nis = float(
            innovation @ np.linalg.solve(S, innovation)
        )
        nis_values.append(nis)

        K = np.linalg.solve(S, H @ P).T
        x = x + K @ innovation

        I = np.eye(6)
        A = I - K @ H

        P = A @ P @ A.T + K @ R @ K.T
        P = 0.5 * (P + P.T)

        # Ground truth is used only for offline evaluation.
        position_error = x[:3] - truth[:3]
        velocity_error = x[3:] - truth[3:]

        position_errors.append(position_error.copy())
        velocity_errors.append(velocity_error.copy())

        sigma_position = np.sqrt(
            np.maximum(np.diag(P)[:3], 0.0)
        )

        coverage_1sigma.extend(
            (np.abs(position_error) <= sigma_position).tolist()
        )

        coverage_2sigma.extend(
            (np.abs(position_error) <= 2.0 * sigma_position).tolist()
        )

    if step < TOTAL_STEPS:

        # Predict the debris state through two-body dynamics.
        F = numerical_jacobian(x, DT)

        x = rk4_step(x, DT)
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        # Advance OrbitZoo by the same interval.
        env.step(step_size=DT)


# ============================================================
# 8. CALCULATE METRICS
# ============================================================

position_errors = np.asarray(position_errors)
velocity_errors = np.asarray(velocity_errors)
nis_values = np.asarray(nis_values)

position_rmse = np.sqrt(
    np.mean(np.sum(position_errors**2, axis=1))
)

velocity_rmse = np.sqrt(
    np.mean(np.sum(velocity_errors**2, axis=1))
)


# ============================================================
# 9. PRINT ORGANIZED REPORT
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

print("\n[2] ESTIMATION ACCURACY")
print("-" * 72)
print(f"  3D position RMSE           : {position_rmse:.3f} m")
print(f"  3D velocity RMSE           : {velocity_rmse:.6f} m/s")

print("\n[3] UNCERTAINTY VALIDATION")
print("-" * 72)
print(f"  Mean NIS                   : {np.mean(nis_values):.3f}")
print("  Reference for 3D measurement: approximately 3.0")
print(f"  Coordinate 1-sigma coverage : {np.mean(coverage_1sigma):.1%}")
print(f"  Coordinate 2-sigma coverage : {np.mean(coverage_2sigma):.1%}")

print("\n[4] FINAL STATE ESTIMATE")
print("-" * 72)
print("  Debris position (m):")
print(" ", np.array2string(x[:3], precision=3))
print("  Debris velocity (m/s):")
print(" ", np.array2string(x[3:], precision=6))

print("\n[5] IMPORTANT LIMITATIONS")
print("-" * 72)
print("  The filter assumes two-body Earth gravity.")
print("  OrbitZoo may include dynamics not represented in this model.")
print("  Process noise is a tunable approximation.")
print("  One run is not enough to establish filter consistency.")
print("=" * 72)

env.close()
