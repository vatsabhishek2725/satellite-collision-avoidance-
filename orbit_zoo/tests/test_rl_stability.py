import math

import torch

from orbitzoo.rl_algorithms.ddpg import DDPG
from orbitzoo.rl_algorithms.dqn import DQN
from orbitzoo.rl_algorithms.ippo import IPPO
from orbitzoo.rl_algorithms.main import RLAlgorithm
from orbitzoo.rl_algorithms.mappo import MAPPO
from orbitzoo.rl_algorithms.ppo import PPO
from orbitzoo.rl_algorithms.td3 import TD3
from orbitzoo.rl_algorithms.utils import PopArt, PrioritizedReplayBuffer, SumTree


def test_sum_tree_handles_non_power_of_two_capacity():
    tree = SumTree(3)
    for priority, data in enumerate(("first", "second", "third"), start=1):
        tree.add(priority, data)

    assert tree.total == 6
    assert tree.get(0.5)[2] == "first"
    assert tree.get(1.5)[2] == "second"
    assert tree.get(4.5)[2] == "third"


def test_prioritized_replay_buffer_samples_non_power_of_two_capacity():
    buffer = PrioritizedReplayBuffer(
        torch.device("cpu"), state_size=2, action_size=1, buffer_size=5
    )
    for index in range(5):
        buffer.add(([index, index + 1], [0], float(index), [index + 1, index + 2], False))

    (states, actions, rewards, next_states, dones), weights, _ = buffer.sample(4)

    assert states.shape == (4, 2)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 2)
    assert dones.shape == (4,)
    assert torch.isfinite(weights).all()


def test_continuous_action_is_scaled_per_axis():
    algorithm = RLAlgorithm(
        torch.device("cpu"), True, [2.0, 3.0, 4.0], action_to_thrust_fn=None
    )

    assert algorithm.action_to_thrust([0.5, -1.0, 0.25]) == [1.0, -3.0, 1.0]


def test_ddpg_can_select_and_update_with_single_transition():
    algorithm = DDPG(
        device=torch.device("cpu"),
        has_continuous_action_space=True,
        action_space=[1.0, 2.0],
        state_dim=3,
        action_dim=2,
        action_to_thrust_fn=None,
        update_after=1,
        batch_size=1,
        memory_capacity=4,
    )
    action = algorithm.select_action([0.1, 0.2, 0.3])
    assert action.shape == (2,)
    algorithm.memory.add(([0.1, 0.2, 0.3], action, 1.0, [0.2, 0.3, 0.4], True))

    metrics = algorithm.update()

    assert all(math.isfinite(value) for value in metrics.values())


def test_dqn_can_update_with_single_transition_and_tree():
    algorithm = DQN(
        device=torch.device("cpu"),
        has_continuous_action_space=False,
        action_space=[1.0],
        state_dim=2,
        action_dim=3,
        action_to_thrust_fn=None,
        update_after=1,
        batch_size=1,
        memory_capacity=5,
    )
    algorithm.memory.add(([0.1, 0.2], 1, 1.0, [0.2, 0.3], True))

    metrics = algorithm.update()

    assert all(math.isfinite(value) for value in metrics.values())


def test_td3_can_update_with_single_transition():
    algorithm = TD3(
        device=torch.device("cpu"),
        has_continuous_action_space=True,
        action_space=[1.0, 1.0],
        state_dim=2,
        action_dim=2,
        action_to_thrust_fn=None,
        update_after=1,
        batch_size=1,
        memory_capacity=5,
        policy_delay=1,
    )
    action = algorithm.select_action([0.1, 0.2], deterministic=True)
    algorithm.memory.add(([0.1, 0.2], action, 1.0, [0.2, 0.3], True))

    metrics = algorithm.update()

    assert all(math.isfinite(value) for value in metrics.values())


def test_discrete_ppo_can_update_from_single_transition():
    algorithm = PPO(
        device=torch.device("cpu"),
        has_continuous_action_space=False,
        action_space=[1.0],
        state_dim=2,
        action_dim=3,
        action_to_thrust_fn=None,
        K_epochs=1,
    )
    algorithm.memory.add(([0.1, 0.2], 1, 1.0, [0.2, 0.3], True))

    metrics = algorithm.update()

    assert all(math.isfinite(value) for value in metrics.values())


def test_continuous_ppo_actions_are_bounded_and_update():
    algorithm = PPO(
        device=torch.device("cpu"),
        has_continuous_action_space=True,
        action_space=[1.0, 1.0],
        state_dim=2,
        action_dim=2,
        action_to_thrust_fn=None,
        K_epochs=1,
    )
    action = algorithm.select_action([0.1, 0.2])
    assert (action >= -1).all() and (action <= 1).all()
    algorithm.memory.add(([0.1, 0.2], action, 1.0, [0.2, 0.3], True))

    metrics = algorithm.update()

    assert all(math.isfinite(value) for value in metrics.values())


def test_discrete_ippo_and_mappo_initialization():
    common = {
        "device": torch.device("cpu"),
        "num_agents": 2,
        "state_dim": 3,
        "action_dim": 2,
        "lr_actor": 1e-3,
        "lr_critic": 1e-3,
        "lr_std": 1e-3,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "K_epochs": 1,
        "batch_size": 2,
        "clip": 0.2,
        "has_continuous_action_space": False,
        "action_space": [1.0],
        "action_to_thrust_fn": None,
    }

    ippo = IPPO(**common)
    mappo = MAPPO(**common, central_state_dim=6)

    assert next(ippo.actor.parameters()).device.type == "cpu"
    assert next(mappo.actor.parameters()).device.type == "cpu"


def test_popart_updates_registered_parameters():
    popart = PopArt(torch.device("cpu"), input_shape=3, output_shape=1)
    before = popart(torch.ones(2, 3)).detach()
    popart.update(torch.tensor([[1.0], [3.0]]))
    after = popart(torch.ones(2, 3)).detach()

    assert before.shape == after.shape
    assert torch.isfinite(after).all()
