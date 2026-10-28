
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

print("Environment initialized.")

for name in ["Spacecraft-1", "Debris-1"]:
    body = env.dynamics.get_body(name)

    print(f"\n{name}")
    print("Position (m):", body.position)
    print("Velocity (m/s):", body.velocity)

spacecraft = env.dynamics.get_body("Spacecraft-1")
debris = env.dynamics.get_body("Debris-1")

distance = (
    (sum((spacecraft.position - debris.position) ** 2)) ** 0.5
)

print("\nInitial separation (m):", distance)

print("\nAdvancing simulation by 5 seconds...")
env.step(step_size=5)

print("Spacecraft position:", spacecraft.position)
print("Debris position:", debris.position)

distance_after = (
    (sum((spacecraft.position - debris.position) ** 2)) ** 0.5
)

print("Separation after 5 seconds (m):", distance_after)
