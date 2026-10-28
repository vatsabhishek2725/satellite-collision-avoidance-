
import numpy as np
from orbitzoo import OrbitZoo


# ============================================================
# 1. SIMULATION CONFIGURATION
# ============================================================

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

env = OrbitZoo(
    dynamics_library="orekit",
    step_size=5,
    spacecrafts=spacecrafts,
    drifters=drifters,
    render=False
)


# ============================================================
# 2. READ CURRENT STATES
# ============================================================

spacecraft = env.dynamics.get_body("Spacecraft-1")
debris = env.dynamics.get_body("Debris-1")

r_sc = np.asarray(spacecraft.position, dtype=float)
v_sc = np.asarray(spacecraft.velocity, dtype=float)

r_debris = np.asarray(debris.position, dtype=float)
v_debris = np.asarray(debris.velocity, dtype=float)


# ============================================================
# 3. RELATIVE MOTION
# ============================================================

relative_position = r_debris - r_sc
relative_velocity = v_debris - v_sc

initial_distance = np.linalg.norm(relative_position)
relative_speed = np.linalg.norm(relative_velocity)


# ============================================================
# 4. LINEAR CLOSEST-APPROACH PREDICTION
# ============================================================

# Time of closest approach under constant relative velocity.
if np.dot(relative_velocity, relative_velocity) > 1e-12:
    t_ca = -np.dot(
        relative_position, relative_velocity
    ) / np.dot(relative_velocity, relative_velocity)
else:
    t_ca = 0.0

# Restrict prediction to the next 600 seconds.
t_ca = np.clip(t_ca, 0.0, 600.0)

predicted_relative_position = (
    relative_position + relative_velocity * t_ca
)

predicted_distance = np.linalg.norm(
    predicted_relative_position
)


# ============================================================
# 5. ORGANIZED REPORT
# ============================================================

print("\n" + "=" * 58)
print("          SATELLITE COLLISION RISK REPORT")
print("=" * 58)

print("\n[1] SIMULATION")
print("-" * 58)
print(f"  Spacecraft             : {spacecraft.name}")
print(f"  Debris                 : {debris.name}")
print(f"  Prediction horizon     : 600 seconds")

print("\n[2] CURRENT RELATIVE MOTION")
print("-" * 58)
print(f"  Initial separation     : {initial_distance:,.3f} m")
print(f"  Relative speed         : {relative_speed:.6f} m/s")

print("\n[3] PREDICTED CLOSEST APPROACH")
print("-" * 58)
print(f"  Predicted time         : {t_ca:,.3f} seconds")
print(f"  Predicted separation   : {predicted_distance:,.3f} m")

print("\n[4] INTERPRETATION")
print("-" * 58)

if t_ca >= 600.0:
    print("  Closest approach is at the end of the prediction window.")
elif relative_speed < 1e-6:
    print("  Relative motion is nearly zero.")
else:
    print("  Closest approach was estimated using linear motion.")

print("  This is a simplified prediction, not a collision guarantee.")
print("=" * 58)
