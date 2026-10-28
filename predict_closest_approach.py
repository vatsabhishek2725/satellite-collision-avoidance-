
import numpy as np
from orbitzoo import OrbitZoo


# ============================================================
# 1. CONFIGURATION
# ============================================================

STEP_SIZE = 5.0
PREDICTION_DURATION = 600.0
NUM_STEPS = int(PREDICTION_DURATION / STEP_SIZE)

spacecrafts = [
    {
        "name": "Spacecraft-1",
        "initial_state": [
            -3529923.947865602,
            7042905.715845195,
            0.0,
            6359.116737768876,
            3187.207008809081,
            0.0
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
            -3529923.947865602,
            7043905.715845195,
            0.0,
            6359.116737768876,
            3187.207008809081,
            0.0
        ]
    }
]


# ============================================================
# 2. INITIALIZE SIMULATION
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
# 3. MEASURE INITIAL SEPARATION
# ============================================================

def calculate_separation():
    position_sc = np.asarray(spacecraft.position, dtype=float)
    position_debris = np.asarray(debris.position, dtype=float)

    return float(np.linalg.norm(position_debris - position_sc))


initial_separation = calculate_separation()

minimum_separation = initial_separation
time_of_minimum = 0.0

# Store measurements for later analysis.
history = [(0.0, initial_separation)]


# ============================================================
# 4. PROPAGATE BOTH OBJECTS
# ============================================================

for step in range(1, NUM_STEPS + 1):

    env.step(step_size=STEP_SIZE)

    current_time = step * STEP_SIZE
    current_separation = calculate_separation()

    history.append((current_time, current_separation))

    if current_separation < minimum_separation:
        minimum_separation = current_separation
        time_of_minimum = current_time


# ============================================================
# 5. GENERATE ORGANIZED REPORT
# ============================================================

final_separation = history[-1][1]

print("\n" + "=" * 64)
print("          ORBITAL CLOSEST-APPROACH REPORT")
print("=" * 64)

print("\n[1] SIMULATION SETTINGS")
print("-" * 64)
print(f"  Prediction duration       : {PREDICTION_DURATION:.1f} s")
print(f"  Time step                 : {STEP_SIZE:.1f} s")
print(f"  Number of intervals       : {NUM_STEPS}")

print("\n[2] SEPARATION MEASUREMENTS")
print("-" * 64)
print(f"  Initial separation        : {initial_separation:,.3f} m")
print(f"  Final separation          : {final_separation:,.3f} m")
print(f"  Minimum sampled separation: {minimum_separation:,.3f} m")
print(f"  Time of minimum           : {time_of_minimum:.1f} s")

print("\n[3] CHANGE IN SEPARATION")
print("-" * 64)
print(
    f"  Net change                : "
    f"{final_separation - initial_separation:+,.3f} m"
)

if final_separation < initial_separation:
    print("  Trend at final time       : Objects are closer than initially.")
elif final_separation > initial_separation:
    print("  Trend at final time       : Objects are farther apart than initially.")
else:
    print("  Trend at final time       : Initial and final distances are equal.")

print("\n[4] INTERPRETATION")
print("-" * 64)
print("  The minimum is the closest separation sampled every 5 seconds.")
print("  It may miss a smaller separation between samples.")
print("  No maneuver was applied during this prediction.")
print("  This report alone does not establish a collision probability.")

print("=" * 64)

env.close()
