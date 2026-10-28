
import numpy as np
from orbitzoo import OrbitZoo


# ============================================================
# 1. CONFIGURATION
# ============================================================

DT = 2.0
TOTAL_TIME = 600.0
SENSOR_INTERVAL = 10.0
MEASUREMENT_STD = 20.0

INITIAL_VELOCITY_STD = 2.0
ACCELERATION_NOISE_STD = 0.05

rng = np.random.default_rng(seed=42)


# ============================================================
# 2. INITIALIZE THE ORBITZOO SCENARIO
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
# 3. KALMAN FILTER MATRICES
# ============================================================

F = np.eye(6)
F[:3, 3:] = np.eye(3) * DT

H = np.zeros((3, 6))
H[:3, :3] = np.eye(3)

R = np.eye(3) * MEASUREMENT_STD**2

Q = np.zeros((6, 6))
noise_block = ACCELERATION_NOISE_STD**2 * np.array([
    [DT**4 / 4, DT**3 / 2],
    [DT**3 / 2, DT**2]
])

for axis in range(3):
    Q[axis, axis] = noise_block[0, 0]
    Q[axis, axis + 3] = noise_block[0, 1]
    Q[axis + 3, axis] = noise_block[1, 0]
    Q[axis + 3, axis + 3] = noise_block[1, 1]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_truth():
    r_sc = np.asarray(spacecraft.position, dtype=float)
    r_debris = np.asarray(debris.position, dtype=float)

    v_sc = np.asarray(spacecraft.velocity, dtype=float)
    v_debris = np.asarray(debris.velocity, dtype=float)

    return np.concatenate([
        r_debris - r_sc,
        v_debris - v_sc
    ])


def get_measurement(truth):
    return truth[:3] + rng.normal(0.0, MEASUREMENT_STD, 3)


def predict(x, P):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, 0.5 * (P_pred + P_pred.T)


def update(x, P, z):
    innovation = z - H @ x
    S = H @ P @ H.T + R

    # NIS is calculated before the measurement update.
    nis = float(innovation @ np.linalg.solve(S, innovation))

    K = np.linalg.solve(S, H @ P).T
    x_new = x + K @ innovation

    I = np.eye(6)
    A = I - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    return x_new, P_new, nis


# ============================================================
# 5. INITIALIZE USING ONLY A NOISY MEASUREMENT
# ============================================================

truth = get_truth()
first_measurement = get_measurement(truth)

x = np.concatenate([
    first_measurement,
    np.zeros(3)
])

P = np.diag([
    MEASUREMENT_STD**2,
    MEASUREMENT_STD**2,
    MEASUREMENT_STD**2,
    INITIAL_VELOCITY_STD**2,
    INITIAL_VELOCITY_STD**2,
    INITIAL_VELOCITY_STD**2
])

# Store initial error for coverage analysis.
errors = [x[:3] - truth[:3]]
variances = [np.diag(P)[:3].copy()]

nis_values = []

sensor_steps = int(round(SENSOR_INTERVAL / DT))
total_steps = int(round(TOTAL_TIME / DT))


# ============================================================
# 6. RUN THE FILTER AND RECORD VALIDATION METRICS
# ============================================================

for step in range(1, total_steps + 1):

    env.step(step_size=DT)

    x, P = predict(x, P)

    if step % sensor_steps == 0:
        truth = get_truth()
        z = get_measurement(truth)

        x, P, nis = update(x, P, z)

        nis_values.append(nis)
        errors.append(x[:3] - truth[:3])
        variances.append(np.diag(P)[:3].copy())


# ============================================================
# 7. CALCULATE VALIDATION METRICS
# ============================================================

errors = np.asarray(errors)
variances = np.asarray(variances)
nis_values = np.asarray(nis_values)

sigma = np.sqrt(variances)
absolute_errors = np.abs(errors)

position_rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

coverage_1sigma = np.mean(absolute_errors <= sigma)
coverage_2sigma = np.mean(absolute_errors <= 2.0 * sigma)

mean_nis = float(np.mean(nis_values))
median_nis = float(np.median(nis_values))


# ============================================================
# 8. PRINT A CLEAR VALIDATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("             KALMAN FILTER CONSISTENCY REPORT")
print("=" * 70)

print("\n[1] TEST CONFIGURATION")
print("-" * 70)
print(f"  Simulation duration       : {TOTAL_TIME:.1f} s")
print(f"  Sensor interval            : {SENSOR_INTERVAL:.1f} s")
print(f"  Measurement noise std     : {MEASUREMENT_STD:.1f} m/coordinate")
print(f"  Measurement updates       : {len(nis_values)}")

print("\n[2] ESTIMATION ACCURACY")
print("-" * 70)
print(f"  3D position RMSE           : {position_rmse:.3f} m")

print("\n[3] INNOVATION CONSISTENCY")
print("-" * 70)
print(f"  Mean NIS                   : {mean_nis:.3f}")
print(f"  Median NIS                 : {median_nis:.3f}")
print("  Reference mean for 3D data : approximately 3.0")
print("  Note: this reference assumes a consistent model.")

print("\n[4] UNCERTAINTY COVERAGE")
print("-" * 70)
print(f"  Coordinate errors within 1 sigma : {coverage_1sigma:.1%}")
print(f"  Coordinate errors within 2 sigma : {coverage_2sigma:.1%}")
print("  Ideal Gaussian references        : approximately 68% and 95%")

print("\n[5] INTERPRETATION")
print("-" * 70)

if mean_nis > 3.0:
    print("  Innovations are larger than the nominal reference on average.")
    print("  The model may underestimate uncertainty or miss orbital effects.")
else:
    print("  Mean NIS is not above the nominal reference.")
    print("  This alone does not prove that the filter is calibrated.")

print("  Coverage is an empirical diagnostic, not proof of calibration.")
print("  More independent runs are needed for stronger conclusions.")
print("=" * 70)

env.close()
