
import numpy as np
from orbitzoo import OrbitZoo


# ============================================================
# 1. CONFIGURATION
# ============================================================

DT = 2.0
TOTAL_TIME = 600.0
SENSOR_INTERVAL = 10.0
SENSOR_NOISE_STD = 20.0

# Initial velocity uncertainty: standard deviation in m/s.
INITIAL_VELOCITY_STD = 2.0

# Assumed acceleration-noise parameter for the motion model.
ACCELERATION_NOISE_STD = 0.05

rng = np.random.default_rng(seed=42)


# ============================================================
# 2. ORBITZOO SCENARIO
# ============================================================

SC_POSITION = np.array([
    -3529923.947865602,
    7042905.715845195,
    0.0
])

SC_VELOCITY = np.array([
    6359.116737768876,
    3187.207008809081,
    0.0
])

spacecrafts = [{
    "name": "Spacecraft-1",
    "initial_state": [
        *SC_POSITION,
        *SC_VELOCITY
    ],
    "isp": 1000,
    "dry_mass": 1,
    "initial_fuel_mass": 4
}]

drifters = [{
    "name": "Debris-1",
    "initial_state": [
        *(SC_POSITION + np.array([0.0, 1000.0, 0.0])),
        *(SC_VELOCITY - np.array([0.0, 2.0, 0.0]))
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
# 3. KALMAN FILTER MODEL
# ============================================================

# State vector:
# [relative_x, relative_y, relative_z,
#  relative_vx, relative_vy, relative_vz]

def transition_matrix(dt):
    F = np.eye(6)
    F[:3, 3:] = np.eye(3) * dt
    return F


# The sensor measures relative position only.
H = np.zeros((3, 6))
H[:3, :3] = np.eye(3)

R = np.eye(3) * SENSOR_NOISE_STD**2


def process_noise(dt, acceleration_std):
    Q = np.zeros((6, 6))

    block = acceleration_std**2 * np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2]
    ])

    for axis in range(3):
        Q[axis, axis] = block[0, 0]
        Q[axis, axis + 3] = block[0, 1]
        Q[axis + 3, axis] = block[1, 0]
        Q[axis + 3, axis + 3] = block[1, 1]

    return Q


def get_ground_truth():
    """Used only for synthetic sensing and offline evaluation."""
    r_sc = np.asarray(spacecraft.position, dtype=float)
    r_debris = np.asarray(debris.position, dtype=float)

    v_sc = np.asarray(spacecraft.velocity, dtype=float)
    v_debris = np.asarray(debris.velocity, dtype=float)

    return np.concatenate([
        r_debris - r_sc,
        v_debris - v_sc
    ])


def simulate_measurement(true_state):
    """Generate a noisy relative-position observation."""
    return (
        true_state[:3]
        + rng.normal(0.0, SENSOR_NOISE_STD, size=3)
    )


def measurement_update(x, P, z):
    innovation = z - H @ x
    S = H @ P @ H.T + R

    # K = P H^T S^-1, without explicitly inverting S.
    K = np.linalg.solve(S, H @ P).T

    x_updated = x + K @ innovation

    # Joseph form helps preserve a valid covariance matrix.
    I = np.eye(6)
    A = I - K @ H
    P_updated = A @ P @ A.T + K @ R @ K.T
    P_updated = 0.5 * (P_updated + P_updated.T)

    return x_updated, P_updated


# ============================================================
# 4. INITIALIZE FROM THE FIRST NOISY MEASUREMENT
# ============================================================

truth = get_ground_truth()
first_measurement = simulate_measurement(truth)

# No ground-truth position or velocity is copied into the estimate.
x = np.concatenate([
    first_measurement,
    np.zeros(3)
])

P = np.diag([
    SENSOR_NOISE_STD**2,
    SENSOR_NOISE_STD**2,
    SENSOR_NOISE_STD**2,
    INITIAL_VELOCITY_STD**2,
    INITIAL_VELOCITY_STD**2,
    INITIAL_VELOCITY_STD**2
])

F = transition_matrix(DT)
Q = process_noise(DT, ACCELERATION_NOISE_STD)

sensor_steps = int(round(SENSOR_INTERVAL / DT))
total_steps = int(round(TOTAL_TIME / DT))

position_errors = []
velocity_errors = []
position_covariances = []
velocity_covariances = []

# Record the initial estimate for evaluation.
position_errors.append(x[:3] - truth[:3])
velocity_errors.append(x[3:] - truth[3:])
position_covariances.append(np.diag(P)[:3].copy())
velocity_covariances.append(np.diag(P)[3:].copy())


# ============================================================
# 5. PROPAGATE, PREDICT AND UPDATE
# ============================================================

for step in range(1, total_steps + 1):

    # Advance the orbital simulation.
    env.step(step_size=DT)

    # Kalman prediction using the constant-velocity model.
    x = F @ x
    P = F @ P @ F.T + Q
    P = 0.5 * (P + P.T)

    # Process a sensor measurement at the configured interval.
    if step % sensor_steps == 0:

        truth = get_ground_truth()
        z = simulate_measurement(truth)

        x, P = measurement_update(x, P, z)

        # Ground truth is used only for evaluation here.
        position_errors.append(x[:3] - truth[:3])
        velocity_errors.append(x[3:] - truth[3:])

        position_covariances.append(np.diag(P)[:3].copy())
        velocity_covariances.append(np.diag(P)[3:].copy())


# ============================================================
# 6. EVALUATION
# ============================================================

position_errors = np.asarray(position_errors)
velocity_errors = np.asarray(velocity_errors)

position_rmse = np.sqrt(
    np.mean(np.sum(position_errors**2, axis=1))
)

velocity_rmse = np.sqrt(
    np.mean(np.sum(velocity_errors**2, axis=1))
)

position_covariances = np.asarray(position_covariances)
velocity_covariances = np.asarray(velocity_covariances)


# ============================================================
# 7. ORGANIZED REPORT
# ============================================================

print("\n" + "=" * 70)
print("        KALMAN FILTER BASELINE — NO GROUND-TRUTH INITIALIZATION")
print("=" * 70)

print("\n[1] CONFIGURATION")
print("-" * 70)
print(f"  Simulation duration        : {TOTAL_TIME:.1f} s")
print(f"  Simulation time step        : {DT:.1f} s")
print(f"  Sensor interval             : {SENSOR_INTERVAL:.1f} s")
print(f"  Measurement noise std      : {SENSOR_NOISE_STD:.1f} m/coordinate")
print(f"  Measurements after startup : {total_steps // sensor_steps}")
print(f"  Evaluation checkpoints     : {len(position_errors)}")

print("\n[2] ESTIMATION ACCURACY")
print("-" * 70)
print(f"  3D relative-position RMSE   : {position_rmse:.3f} m")
print(f"  3D relative-velocity RMSE   : {velocity_rmse:.6f} m/s")

print("\n[3] FINAL STATE ESTIMATE")
print("-" * 70)
print("  Relative position (m)      :")
print(" ", np.array2string(x[:3], precision=3))
print("  Relative velocity (m/s)    :")
print(" ", np.array2string(x[3:], precision=6))

print("\n[4] FINAL ESTIMATED UNCERTAINTY")
print("-" * 70)
print("  Position variance (m^2)    :")
print(" ", np.array2string(np.diag(P)[:3], precision=3))
print("  Velocity variance ((m/s)^2):")
print(" ", np.array2string(np.diag(P)[3:], precision=6))

print("\n[5] INTERPRETATION")
print("-" * 70)
print("  Initialization uses the first noisy position measurement.")
print("  Velocity begins at zero with explicitly specified uncertainty.")
print("  Ground truth is not used by the filter's prediction or update.")
print("  The constant-velocity model is only a baseline for orbital motion.")
print("  Covariance calibration still requires further validation.")
print("=" * 70)

env.close()

