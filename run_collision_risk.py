
import os
import numpy as np

from collision_risk_predictor import (
    predict_closest_approach,
    print_report,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "ekf_collision_input.npz"

PREDICTION_HORIZON = 600.0

# Illustrative combined collision radius.
# Replace with a justified spacecraft/debris size assumption.
COLLISION_RADIUS = 10.0


# ============================================================
# 2. LOAD EKF OUTPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"{INPUT_FILE} was not found.\n"
        "Run orbital_ekf.py first to generate the EKF output."
    )

with np.load(INPUT_FILE) as data:
    spacecraft_state = data["spacecraft_state"].copy()

    debris_estimated_state = (
        data["debris_estimated_state"].copy()
    )

    debris_covariance = data["debris_covariance"].copy()

    simulation_time = float(data["simulation_time"])
    measurement_std = float(data["measurement_std"])
    random_seed = int(data["random_seed"])


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

if spacecraft_state.shape != (6,):
    raise ValueError(
        "Invalid spacecraft state shape. Expected (6,)."
    )

if debris_estimated_state.shape != (6,):
    raise ValueError(
        "Invalid estimated debris state shape. Expected (6,)."
    )

if debris_covariance.shape != (6, 6):
    raise ValueError(
        "Invalid EKF covariance shape. Expected (6, 6)."
    )

if not np.all(np.isfinite(debris_covariance)):
    raise ValueError(
        "EKF covariance contains non-finite values."
    )

if not np.allclose(
    debris_covariance,
    debris_covariance.T,
    atol=1e-8,
):
    raise ValueError(
        "EKF covariance is not symmetric."
    )


# ============================================================
# 4. COLLISION-RISK PREDICTION
# ============================================================

result = predict_closest_approach(
    spacecraft_state=spacecraft_state,
    debris_estimated_state=debris_estimated_state,
    debris_covariance=debris_covariance,
    prediction_horizon=PREDICTION_HORIZON,
    collision_radius=COLLISION_RADIUS,
)


# ============================================================
# 5. PRINT INTEGRATED REPORT
# ============================================================

print("\n" + "=" * 72)
print("       EKF-INTEGRATED COLLISION-RISK PREDICTION")
print("=" * 72)

print("\n[1] EKF SOURCE")
print("-" * 72)
print(f"  Input file                : {INPUT_FILE}")
print(f"  EKF simulation time       : {simulation_time:.1f} s")
print(f"  Measurement noise std     : {measurement_std:.1f} m")
print(f"  EKF random seed           : {random_seed}")

print("\n[2] ESTIMATED STATE")
print("-" * 72)
print(
    "  Spacecraft position (m)   :",
    np.array2string(spacecraft_state[:3], precision=3),
)
print(
    "  Spacecraft velocity (m/s) :",
    np.array2string(spacecraft_state[3:], precision=6),
)
print(
    "  Debris estimate position (m):",
    np.array2string(debris_estimated_state[:3], precision=3),
)
print(
    "  Debris estimate velocity (m/s):",
    np.array2string(debris_estimated_state[3:], precision=6),
)

print_report(result)
