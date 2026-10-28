
import numpy as np
from orbitzoo import OrbitZoo

# One simulated debris object
drifters = [
    {
        "name": "Debris-1",
        "initial_state": [
            6_878_000,       # Semi-major axis: Earth radius + 500 km
            0.001,           # Eccentricity
            np.radians(50),  # Inclination
            np.pi,           # Argument of periapsis
            np.pi,           # Longitude of ascending node
            0.0              # Mean anomaly
        ]
    }
]

# Initialize the orbital simulation
env = OrbitZoo(
    dynamics_library="tensorgator",
    step_size=5,
    drifters=drifters,
    render=False
)

print("Simulation initialized successfully!")

# Advance the simulation and inspect the actual state
for step in range(5):
    env.step()

    debris = env.dynamics.get_body("Debris-1")

    print(f"\nStep: {step + 1}")
    print("Position (m):", debris.position)
    print("Velocity (m/s):", debris.velocity)

print("\nBasic OrbitZoo test completed.")
