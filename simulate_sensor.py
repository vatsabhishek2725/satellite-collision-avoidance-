
import numpy as np
from orbitzoo import OrbitZoo


# ============================================================
# 1. SENSOR AND SIMULATION SETTINGS
# ============================================================

STEP_SIZE = 2.0
PREDICTION_DURATION = 600.0
SENSOR_INTERVAL = 10.0
SENSOR_NOISE_STD = 20.0  # metres, per coordinate

rng = np.random.default_rng(seed=42)


# ============================================================
# 2. DEFINE THE CLOSE-APPROACH SCENARIO
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

INITIAL_SEPARATION = 1000.0
CLOSING_SPEED = 2.0

spacecrafts = [
    {
        "name": "Spacecraft-1",
        "initial_state": [
            *SC_POSITION,
            *SC_VELOCITY
        ],
        "isp": 1000,
        "dry_mass": 1,
        "initial_fuel_mass": 4
    }
]

drifters = [
    {
        "name": "Debris-1",
        "initial_state": [
            *(SC_POSITION + np.array([
                0.0, INITIAL_SEPARATION, 0.0
            ])),
            *(SC_VELOCITY - np.array([
                0.0, CLOSING_SPEED, 0.0
            ]))
        ]
    }
]


# ============================================================
# 3. INITIALIZE ORBITZOO
# ============================================================

env = OrbitZoo(
    dynamics_library="orekit",
    step_size=STEP_SIZE,
    spacecrafts=spacecrafts,
    drifters=drifters,
    render=False
)

spacecraft = env.dynamics.get_body("Spacecraft-1")
debris = env.dynamics.get_body("Debris-1")


# ============================================================
# 4. SIMULATE NOISY RELATIVE-POSITION MEASUREMENTS
# ============================================================

measurements = []
num_steps = int(PREDICTION_DURATION / STEP_SIZE)
measurement_every = int(SENSOR_INTERVAL / STEP_SIZE)

for step in range(num_steps + 1):

    # Record a measurement at t=0 and every SENSOR_INTERVAL.
    if step % measurement_every == 0:

        r_sc = np.asarray(spacecraft.position, dtype=float)
        r_debris = np.asarray(debris.position, dtype=float)

        # Ground truth is used to simulate and evaluate the sensor.
        true_relative_position = r_debris - r_sc

        noise = rng.normal(
            loc=0.0,
            scale=SENSOR_NOISE_STD,
            size=3
        )

        measured_relative_position = (
            true_relative_position + noise
        )

        measurements.append({
            "time": step * STEP_SIZE,
            "truth": true_relative_position.copy(),
            "measurement": measured_relative_position.copy(),
            "error": noise.copy()
        })

    if step < num_steps:
        env.step(step_size=STEP_SIZE)


# ============================================================
# 5. EVALUATE SENSOR ERROR
# ============================================================

errors = np.array([item["error"] for item in measurements])

coordinate_rmse = np.sqrt(np.mean(errors ** 2))
three_dimensional_error = np.linalg.norm(errors, axis=1)
position_error_rmse = np.sqrt(
    np.mean(three_dimensional_error ** 2)
)


# ============================================================
# 6. PRINT ORGANIZED REPORT
# ============================================================

print("\n" + "=" * 72)
print("                 SIMULATED RADAR SENSOR REPORT")
print("=" * 72)

print("\n[1] SENSOR CONFIGURATION")
print("-" * 72)
print(f"  Measurement interval       : {SENSOR_INTERVAL:.1f} seconds")
print(f"  Noise standard deviation   : {SENSOR_NOISE_STD:.1f} m per coordinate")
print(f"  Random seed                : 42")
print(f"  Measurements collected     : {len(measurements)}")

print("\n[2] SAMPLE OBSERVATIONS")
print("-" * 72)
print(
    f"{'Time (s)':>8} | {'True range (m)':>15} | "
    f"{'Measured range (m)':>19} | {'Range error (m)':>15}"
)
print("-" * 72)

for item in measurements[:8]:
    truth_range = np.linalg.norm(item["truth"])
    measured_range = np.linalg.norm(item["measurement"])
    range_error = measured_range - truth_range

    print(
        f"{item['time']:8.1f} | "
        f"{truth_range:15.3f} | "
        f"{measured_range:19.3f} | "
        f"{range_error:+15.3f}"
    )

print("\n[3] SENSOR ERROR EVALUATION")
print("-" * 72)
print(f"  Coordinate RMSE             : {coordinate_rmse:.3f} m")
print(f"  3D position-error RMSE      : {position_error_rmse:.3f} m")

print("\n[4] IMPORTANT NOTES")
print("-" * 72)
print("  The sensor observes relative position, not velocity.")
print("  Ground truth is used only to generate and evaluate measurements.")
print("  The future estimator must use measurements, not ground truth.")
print("  This is a simplified sensor model, not a validated radar model.")
print("=" * 72)

env.close()
