
from orbitzoo import OrbitZoo

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

env = OrbitZoo(
    dynamics_library="orekit",
    step_size=5,
    spacecrafts=spacecrafts,
    render=False
)

print("Orekit environment initialized successfully!")

spacecraft = env.dynamics.get_body("Spacecraft-1")

print("\nInitial position (m):", spacecraft.position)
print("Initial velocity (m/s):", spacecraft.velocity)

print("\nApplying a small along-track thrust...")

env.step(
    step_size=5,
    actions={"Spacecraft-1": [0, 0.1, 0]}
)

print("\nPosition after maneuver (m):", spacecraft.position)
print("Velocity after maneuver (m/s):", spacecraft.velocity)

print("\nOrekit maneuver test completed.")
