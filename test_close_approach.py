
import numpy as np
from orbitzoo import OrbitZoo


# ============================================================
# 1. CONFIGURATION
# ============================================================

STEP_SIZE = 2.0
PREDICTION_DURATION = 600.0
NUM_STEPS = int(PREDICTION_DURATION / STEP_SIZE)

INITIAL_SEPARATION = 1_000.0
CLOSING_SPEED = 2.0  # Initial relative speed toward spacecraft, m/s

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


# ============================================================
# 2. DEFINE SPACECRAFT AND DEBRIS
# ============================================================

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

# Debris begins 1,000 m away along the y-axis.
# Its y-velocity is 2 m/s lower than the spacecraft's.
debris_position = SC_POSITION + np.array([
    0.0, INITIAL_SEPARATION, 0.0
])

debris_velocity = SC_VELOCITY - np.array([
    0.0, CLOSING_SPEED, 0.0
])

drifters = [
    {
        "name": "Debris-1",
        "initial_state": [
            *debris_position,
            *debris_velocity
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
# 4. SEPARATION CALCULATION
# ============================================================

def get_separation():
    r_sc = np.asarray(spacecraft.position, dtype=float)
    r_debris = np.asarray(debris.position, dtype=float)

    return float(np.linalg.norm(r_debris - r_sc))


initial_distance = get_separation()
minimum_distance = initial_distance
time_of_minimum = 0.0
time_of_first_closing_step = None

history = [(0.0, initial_distance)]


# ============================================================
# 5. PROPAGATE THE ORBITS
# ============================================================

for step in range(1, NUM_STEPS + 1):
    env.step(step_size=STEP_SIZE)

    current_time = step * STEP_SIZE
    current_distance = get_separation()

    history.append((current_time, current_distance))

    if current_distance < minimum_distance:
        minimum_distance = current_distance
        time_of_minimum = current_time

    if (
        time_of_first_closing_step is None
        and current_distance < history[-2][1]
    ):
        time_of_first_closing_step = current_time


# ============================================================
# 6. PRINT A CLEAR REPORT
# ============================================================

final_distance = history[-1][1]
distance_change = final_distance - initial_distance

print("\n" + "=" * 64)
print("             CONTROLLED CLOSE-APPROACH TEST")
print("=" * 64)

print("\n[1] SCENARIO")
print("-" * 64)
print(f"  Initial separation       : {initial_distance:,.3f} m")
print(f"  Initial closing speed    : {CLOSING_SPEED:.3f} m/s")
print(f"  Prediction duration      : {PREDICTION_DURATION:.1f} s")
print(f"  Simulation time step     : {STEP_SIZE:.1f} s")
print(f"  Number of intervals      : {NUM_STEPS}")

print("\n[2] SIMULATION RESULTS")
print("-" * 64)
print(f"  Final separation         : {final_distance:,.3f} m")
print(f"  Minimum sampled distance : {minimum_distance:,.3f} m")
print(f"  Time of minimum          : {time_of_minimum:.1f} s")
print(f"  Net distance change      : {distance_change:+,.3f} m")

print("\n[3] INTERPRETATION")
print("-" * 64)

if minimum_distance < initial_distance:
    print("  The objects came closer during the simulation.")
else:
    print("  The sampled minimum was not below the initial separation.")

if time_of_first_closing_step is not None:
    print(
        "  First sampled decrease   : "
        f"{time_of_first_closing_step:.1f} s"
    )
else:
    print("  No decrease was observed at the sampled time steps.")

print("\n  NOTE: This is a synthetic test scenario.")
print("  The sampled minimum may miss a closer approach between steps.")
print("  No collision or maneuver is assumed.")
print("=" * 64)

env.close()
