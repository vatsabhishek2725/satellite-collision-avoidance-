
import inspect
from orbitzoo import OrbitZoo

print("OrbitZoo constructor:")
print(inspect.signature(OrbitZoo))

print("\nOrbitZoo class source:")
print(inspect.getfile(OrbitZoo))

env = OrbitZoo(
    dynamics_library="tensorgator",
    step_size=5,
    drifters=[
        {
            "name": "Debris-1",
            "initial_state": [
                6_878_000, 0.001, 0.872665,
                3.141593, 3.141593, 0.0
            ]
        }
    ],
    render=False
)

print("\nEnvironment methods:")
print([m for m in dir(env) if not m.startswith("_")])

print("\nDynamics methods:")
print([m for m in dir(env.dynamics) if not m.startswith("_")])
