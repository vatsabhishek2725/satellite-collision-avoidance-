
import numpy as np


def predict_closest_approach(
    spacecraft_state,
    debris_estimated_state,
    debris_covariance,
    prediction_horizon=600.0,
    collision_radius=10.0,
):
    """
    Predict closest approach using a constant-relative-velocity model.

    Parameters
    ----------
    spacecraft_state : array-like, shape (6,)
        Spacecraft position and velocity [x, y, z, vx, vy, vz].
        Position in meters; velocity in m/s.

    debris_estimated_state : array-like, shape (6,)
        EKF-estimated debris position and velocity.

    debris_covariance : array-like, shape (6, 6)
        EKF covariance matrix for the estimated debris state.

    prediction_horizon : float
        Maximum prediction duration in seconds.

    collision_radius : float
        Assumed combined collision radius in meters.

    Returns
    -------
    dict
        Closest-approach time, miss distance, and uncertainty metrics.
    """

    spacecraft_state = np.asarray(spacecraft_state, dtype=float)
    debris_estimated_state = np.asarray(
        debris_estimated_state, dtype=float
    )
    debris_covariance = np.asarray(debris_covariance, dtype=float)

    if spacecraft_state.shape != (6,):
        raise ValueError("spacecraft_state must have shape (6,)")

    if debris_estimated_state.shape != (6,):
        raise ValueError("debris_estimated_state must have shape (6,)")

    if debris_covariance.shape != (6, 6):
        raise ValueError("debris_covariance must have shape (6, 6)")

    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")

    if collision_radius <= 0:
        raise ValueError("collision_radius must be positive")

    # Relative position and velocity: debris minus spacecraft
    relative_position = (
        debris_estimated_state[:3] - spacecraft_state[:3]
    )

    relative_velocity = (
        debris_estimated_state[3:] - spacecraft_state[3:]
    )

    relative_speed_squared = np.dot(
        relative_velocity, relative_velocity
    )

    # Calculate time of closest approach.
    # If relative velocity is zero, closest approach is at t = 0.
    if relative_speed_squared < 1e-12:
        tca = 0.0
    else:
        tca = -np.dot(
            relative_position, relative_velocity
        ) / relative_speed_squared

        # Restrict prediction to the specified future horizon.
        tca = float(np.clip(tca, 0.0, prediction_horizon))

    # Relative position at predicted closest approach
    miss_vector = (
        relative_position + relative_velocity * tca
    )

    miss_distance = float(np.linalg.norm(miss_vector))

    # Constant-velocity state transition matrix:
    #
    # position(t) = position(0) + velocity(0) * t
    # velocity(t) = velocity(0)
    #
    # Propagate the debris covariance to TCA.
    phi = np.eye(6)
    phi[:3, 3:] = np.eye(3) * tca

    covariance_at_tca = (
        phi @ debris_covariance @ phi.T
    )

    # Extract position covariance at TCA.
    position_covariance_at_tca = (
        covariance_at_tca[:3, :3]
    )

    # Estimate uncertainty in the direction of the miss vector.
    if miss_distance > 1e-9:
        miss_direction = miss_vector / miss_distance

        miss_variance = (
            miss_direction
            @ position_covariance_at_tca
            @ miss_direction
        )

        miss_sigma = float(
            np.sqrt(max(miss_variance, 0.0))
        )

    else:
        # At nearly zero miss distance, there is no unique
        # miss direction. Use the largest position uncertainty.
        eigenvalues = np.linalg.eigvalsh(
            position_covariance_at_tca
        )

        miss_sigma = float(
            np.sqrt(max(eigenvalues[-1], 0.0))
        )

    # Simple screening indicator, NOT a formal collision probability.
    lower_bound_3sigma = max(
        0.0, miss_distance - 3.0 * miss_sigma
    )

    upper_bound_3sigma = (
        miss_distance + 3.0 * miss_sigma
    )

    if lower_bound_3sigma > collision_radius:
        risk_indicator = "LOWER UNCERTAINTY BOUND ABOVE RADIUS"
    elif upper_bound_3sigma < collision_radius:
        risk_indicator = "UNCERTAINTY REGION WITHIN RADIUS"
    else:
        risk_indicator = "POTENTIAL ENCOUNTER — NEEDS FURTHER ANALYSIS"

    return {
        "tca_seconds": tca,
        "miss_distance_m": miss_distance,
        "miss_sigma_m": miss_sigma,
        "collision_radius_m": collision_radius,
        "lower_3sigma_m": lower_bound_3sigma,
        "upper_3sigma_m": upper_bound_3sigma,
        "risk_indicator": risk_indicator,
        "relative_position_m": relative_position,
        "relative_velocity_mps": relative_velocity,
        "covariance_at_tca": covariance_at_tca,
    }


def print_report(result):
    """Print a categorized collision-risk report."""

    print("=" * 72)
    print("             COLLISION-RISK PREDICTION REPORT")
    print("=" * 72)

    print("\n[1] CLOSEST APPROACH")
    print("-" * 72)
    print(
        f"  Time of closest approach : "
        f"{result['tca_seconds']:.3f} s"
    )
    print(
        f"  Predicted miss distance : "
        f"{result['miss_distance_m']:.3f} m"
    )

    print("\n[2] UNCERTAINTY")
    print("-" * 72)
    print(
        f"  Miss-distance sigma     : "
        f"{result['miss_sigma_m']:.3f} m"
    )
    print(
        f"  Approx. lower 3-sigma   : "
        f"{result['lower_3sigma_m']:.3f} m"
    )
    print(
        f"  Approx. upper 3-sigma   : "
        f"{result['upper_3sigma_m']:.3f} m"
    )

    print("\n[3] SCREENING")
    print("-" * 72)
    print(
        f"  Assumed collision radius: "
        f"{result['collision_radius_m']:.3f} m"
    )
    print(
        f"  Indicator              : "
        f"{result['risk_indicator']}"
    )

    print("\n[4] RELATIVE STATE")
    print("-" * 72)
    print(
        "  Relative position (m)   :",
        np.array2string(
            result["relative_position_m"],
            precision=3,
        ),
    )
    print(
        "  Relative velocity (m/s) :",
        np.array2string(
            result["relative_velocity_mps"],
            precision=6,
        ),
    )

    print("\n" + "=" * 72)
    print("NOTE: This is a constant-velocity screening model.")
    print("The indicator is not a formal collision probability.")
    print("=" * 72)


if __name__ == "__main__":

    # ---------------------------------------------------------
    # DEMONSTRATION INPUTS
    # ---------------------------------------------------------
    # These are illustrative inputs for testing the mathematics.
    # Later, replace debris_state and debris_covariance with
    # the EKF estimate and covariance.
    #
    # Do not feed ground-truth debris state to the RL agent.
    # ---------------------------------------------------------

    spacecraft_state = np.array([
        -3529923.947865602,
        7042905.715845195,
        0.0,
        6359.116737768876,
        3187.207008809081,
        0.0,
    ])

    debris_estimated_state = np.array([
        -3529923.947865602,
        7043905.715845195,
        0.0,
        6359.116737768876,
        3185.207008809081,
        0.0,
    ])

    # Illustrative initial covariance:
    # position uncertainty = 20 m per coordinate
    # velocity uncertainty = 5 m/s per coordinate
    #
    # Replace this with the actual EKF covariance matrix.
    debris_covariance = np.diag([
        20.0**2,
        20.0**2,
        20.0**2,
        5.0**2,
        5.0**2,
        5.0**2,
    ])

    result = predict_closest_approach(
        spacecraft_state=spacecraft_state,
        debris_estimated_state=debris_estimated_state,
        debris_covariance=debris_covariance,
        prediction_horizon=600.0,
        collision_radius=10.0,
    )

    print_report(result)
