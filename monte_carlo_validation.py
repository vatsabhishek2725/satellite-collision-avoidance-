
import numpy as np
from orbitzoo import OrbitZoo


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

NUM_RUNS = 10
DT = 2.0
TOTAL_TIME = 600.0
SENSOR_INTERVAL = 10.0
MEASUREMENT_STD = 20.0

INITIAL_VELOCITY_STD = 2.0
ACCELERATION_NOISE_STD = 0.05

SENSOR_STEPS = int(SENSOR_INTERVAL / DT)
TOTAL_STEPS = int(TOTAL_TIME / DT)


# ============================================================
# 2. FILTER MATRICES
# ============================================================

F = np.eye(6)
F[:3, 3:] = np.eye(3) * DT

H = np.zeros((3, 6))
H[:3, :3] = np.eye(3)

R = np.eye(3) * MEASUREMENT_STD**2

Q = np.zeros((6, 6))
block = ACCELERATION_NOISE_STD**2 * np.array([
    [DT**4 / 4, DT**3 / 2],
    [DT**3 / 2, DT**2]
])

for axis in range(3):
    Q[axis, axis] = block[0, 0]
    Q[axis, axis + 3] = block[0, 1]
    Q[axis + 3, axis] = block[1, 0]
    Q[axis + 3, axis + 3] = block[1, 1]


# ============================================================
# 3. RUN ONE INDEPENDENT EXPERIMENT
# ============================================================

def run_experiment(seed):

    rng = np.random.default_rng(seed)

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

    env = OrbitZoo(
        dynamics_library="orekit",
        step_size=DT,
        spacecrafts=[{
            "name": "Spacecraft-1",
            "initial_state": [*r0, *v0],
            "isp": 1000,
            "dry_mass": 1,
            "initial_fuel_mass": 4
        }],
        drifters=[{
            "name": "Debris-1",
            "initial_state": [
                *(r0 + np.array([0.0, 1000.0, 0.0])),
                *(v0 - np.array([0.0, 2.0, 0.0]))
            ]
        }],
        render=False
    )

    spacecraft = env.dynamics.get_body("Spacecraft-1")
    debris = env.dynamics.get_body("Debris-1")

    def truth_state():
        return np.concatenate([
            np.asarray(debris.position) - np.asarray(spacecraft.position),
            np.asarray(debris.velocity) - np.asarray(spacecraft.velocity)
        ])

    def measurement(truth):
        return truth[:3] + rng.normal(0, MEASUREMENT_STD, 3)

    # Initialize from the first noisy observation.
    truth = truth_state()
    x = np.concatenate([measurement(truth), np.zeros(3)])

    P = np.diag([
        MEASUREMENT_STD**2,
        MEASUREMENT_STD**2,
        MEASUREMENT_STD**2,
        INITIAL_VELOCITY_STD**2,
        INITIAL_VELOCITY_STD**2,
        INITIAL_VELOCITY_STD**2
    ])

    errors = [x[:3] - truth[:3]]
    variances = [np.diag(P)[:3].copy()]
    nis_values = []

    for step in range(1, TOTAL_STEPS + 1):

        env.step(step_size=DT)

        # Constant-velocity filter prediction.
        x = F @ x
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        if step % SENSOR_STEPS == 0:
            truth = truth_state()
            z = measurement(truth)

            innovation = z - H @ x
            S = H @ P @ H.T + R

            nis = float(
                innovation @ np.linalg.solve(S, innovation)
            )

            K = np.linalg.solve(S, H @ P).T
            x = x + K @ innovation

            I = np.eye(6)
            A = I - K @ H
            P = A @ P @ A.T + K @ R @ K.T
            P = 0.5 * (P + P.T)

            errors.append(x[:3] - truth[:3])
            variances.append(np.diag(P)[:3].copy())
            nis_values.append(nis)

    env.close()

    errors = np.asarray(errors)
    variances = np.asarray(variances)

    position_rmse = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
    coverage_1sigma = np.mean(np.abs(errors) <= np.sqrt(variances))
    coverage_2sigma = np.mean(np.abs(errors) <= 2 * np.sqrt(variances))

    return {
        "seed": seed,
        "position_rmse": position_rmse,
        "mean_nis": float(np.mean(nis_values)),
        "coverage_1sigma": coverage_1sigma,
        "coverage_2sigma": coverage_2sigma
    }


# ============================================================
# 4. RUN ALL EXPERIMENTS
# ============================================================

results = []

print("\n" + "=" * 72)
print("           MONTE CARLO KALMAN FILTER VALIDATION")
print("=" * 72)

for run in range(NUM_RUNS):
    result = run_experiment(seed=100 + run)
    results.append(result)

    print(
        f"Run {run + 1:02d}/{NUM_RUNS} | "
        f"Position RMSE: {result['position_rmse']:7.3f} m | "
        f"Mean NIS: {result['mean_nis']:6.3f}"
    )


# ============================================================
# 5. AGGREGATE RESULTS
# ============================================================

rmse = np.array([r["position_rmse"] for r in results])
nis = np.array([r["mean_nis"] for r in results])
coverage1 = np.array([r["coverage_1sigma"] for r in results])
coverage2 = np.array([r["coverage_2sigma"] for r in results])


# ============================================================
# 6. FINAL REPORT
# ============================================================

print("\n" + "=" * 72)
print("                    AGGREGATE RESULTS")
print("=" * 72)

print("\n[1] POSITION ACCURACY")
print("-" * 72)
print(f"  Mean RMSE          : {np.mean(rmse):.3f} m")
print(f"  Standard deviation : {np.std(rmse, ddof=1):.3f} m")
print(f"  Worst run          : {np.max(rmse):.3f} m")
print(f"  Best run           : {np.min(rmse):.3f} m")

print("\n[2] INNOVATION CONSISTENCY")
print("-" * 72)
print(f"  Average run NIS    : {np.mean(nis):.3f}")
print("  Reference          : approximately 3 for a consistent 3D model")

print("\n[3] UNCERTAINTY COVERAGE")
print("-" * 72)
print(f"  Mean 1-sigma coverage : {np.mean(coverage1):.1%}")
print(f"  Mean 2-sigma coverage : {np.mean(coverage2):.1%}")
print("  Ideal Gaussian references: approximately 68% and 95%")

print("\n[4] CONCLUSION")
print("-" * 72)
print("  These results measure sensitivity to sensor noise.")
print("  They do not validate the orbital motion model by themselves.")
print("  The current filter still assumes constant relative velocity.")
print("=" * 72)
