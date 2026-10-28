
import inspect
from orbitzoo import OrbitZoo

env = OrbitZoo(
    dynamics_library="tensorgator",
    step_size=5,
    drifters=[
        {
            "name": "Debris-1",
            "initial_state": [
                6_878_000,
                0.001,
                0.872665,
                3.141593,
                3.141593,
                0.0
            ]
        }
    ],
    render=False
)

print("=== Method signatures ===")

for name in ["reset", "action_space", "observation_space", "step"]:
    method = getattr(env, name)
    print(f"{name}: {inspect.signature(method)}")

print("\n=== Resetting environment ===")
result = env.reset()
print("Reset result:", result)

print("\n=== Agent information ===")

for name in ["agents", "possible_agents"]:
    print(f"{name}: {getattr(env, name, 'Not available')}")

agents = getattr(env, "agents", [])

if agents:
    print("\n=== Action and observation spaces ===")

    for agent in agents:
        print(f"\nAgent: {agent}")
        print("Action space:", env.action_space(agent))
        print("Observation space:", env.observation_space(agent))
else:
    print("\nNo agents initialized yet.")
    print("We need to inspect how OrbitZoo creates its agents.")

print("\n=== Environment state ===")
print("State:", env.state)

print("\nInspection completed.")
