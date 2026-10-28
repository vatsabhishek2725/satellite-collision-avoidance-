import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions import Categorical
from torch.distributions.transforms import TanhTransform
from torch.utils.data import DataLoader, TensorDataset

from orbitzoo.rl_algorithms.utils import RolloutBuffer
from orbitzoo.rl_algorithms.main import RLAlgorithm

class ActorCritic(nn.Module):
    def __init__(self, device, state_dim_actor, state_dim_critic, action_dim, has_continuous_action_space, action_std_init, learn_std):
        super(ActorCritic, self).__init__()

        self.device = device
        self.has_continuous_action_space = has_continuous_action_space
        self.learn_std = learn_std
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            # nn.BatchNorm1d(state_dim_actor),
                            nn.Linear(state_dim_actor, 500),
                            nn.Tanh(),
                            nn.Linear(500, 450),
                            nn.Tanh(),
                            nn.Linear(450, action_dim),
                            nn.Tanh(),
                        )
            if self.learn_std:
                self.log_std = nn.Parameter(torch.full((action_dim,), torch.log(torch.tensor(action_std_init))))
            # with torch.no_grad():
            #     self.actor[-2].weight.data *= 0.01
            #     self.actor[-2].bias.data *= 0.01
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim_actor, 500),
                            nn.Tanh(),
                            nn.Linear(500, 450),
                            nn.Tanh(),
                            nn.Linear(450, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # lower weights of last layer
        # in discrete spaces, this creates a near uniform distribution across actions
        # in continuous spaces (where actions vary between [-1, 1]), this creates outputs around 0
        with torch.no_grad():
            self.actor[-2].weight.data *= 0.01
            self.actor[-2].bias.data *= 0.01

        # critic
        self.critic = nn.Sequential(
                        # nn.BatchNorm1d(state_dim_critic),
                        nn.Linear(state_dim_critic, 500),
                        nn.Tanh(),
                        nn.Linear(500, 450),
                        nn.Tanh(),
                        nn.Linear(450, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
            # print(self.action_var)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def action_distribution(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            if self.learn_std:
                action_std = torch.exp(self.log_std)
                covariance = torch.diag_embed(action_std**2).to(self.device)
            else:
                action_var = self.action_var.expand_as(action_mean)
                covariance = torch.diag_embed(action_var).to(self.device)
            return TransformedDistribution(
                MultivariateNormal(action_mean, covariance),
                [TanhTransform(cache_size=1)],
            )
        return Categorical(self.actor(state))
    
    def act(self, state):
        dist = self.action_distribution(state)
        action = dist.sample()
        return action.detach() # , action_logprob.detach()

    
    def evaluate(self, state, action):
        dist = self.action_distribution(state)
        if self.has_continuous_action_space:
            action = action.reshape(-1, self.action_dim)
            action = action.clamp(-1 + 1e-6, 1 - 1e-6)

        action_logprobs = dist.log_prob(action)
        dist_entropy = (
            -action_logprobs
            if self.has_continuous_action_space
            else dist.entropy()
        )
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPO(RLAlgorithm):
    def __init__(
            self, 
            device,
            has_continuous_action_space: bool,
            action_space: list[float],
            state_dim: float, 
            action_dim: float,
            action_to_thrust_fn,
            lr_actor: float = 3e-4, 
            lr_critic: float = 3e-4,
            lr_std: float = 3e-4,
            gae_lambda: float = 0.95,
            gamma: float = 0.99,
            K_epochs: float = 5,
            weight_decay: float = 0.0,
            clip: float = 0.2,
            learn_epsilon: bool = True,
            epsilon: float = 0.5,
            epsilon_decay_rate: float = 0.99,
            epsilon_decay_every_updates: int = 100,
            epsilon_min: float = 0.2,
            ):
        """
        Args:
            device: Torch device used to perform computations (GPU or CPU).
            has_continuous_action_space (bool): If this agent contains a continuous action space (opposite to a discrete action space).
            action_space (list[float]): Maximum action space. It is used in function 'action_to_thrust', to convert output from networks (actions) to a thrust in polar parameterization.
            state_dim (float): Input size for the Q-network (number of observed features).
            action_dim (float): Output size for the Q-network (number of possible actions/thrusts).
            lr_actor (float): Learning rate for the actor network.
            lr_critic (float): Learning rate for the critic network.
            lr_std (float): Learning rate for the action sampling standard deviation (exploration).
            gae_lambda (float): Lambda value used when computing advantages and returns via Generalized Advantage Estimation (GAE).
            gamma (float): Discount factor for calculating returns (discounted cumulative rewards).
            K_epochs (int): Number of updates per *self.update()* call.
            weight_decay (float): L2 regularization on weights. Prevents overfitting by penalizing large weights when updating the network.
            clip (float): Clip value used on the actor loss (clip-objective).
            learn_epsilon (bool): Let PPO automatically learn how to change action sampling standard deviation (exploration).
            epsilon (float): Initial action sampling standard deviation.
            epsilon_decay_rate (float): Rate at which the standard deviation is decayed. Formula for decay is: ```epsilon *= epsilon_decay_rate```.
            epsilon_decay_every_updates (int): Number of updates between each exploration decay.
            epsilon_min (float): Minimum standard deviation.
        """

        super().__init__(device, has_continuous_action_space, action_space, action_to_thrust_fn)

        self.learn_epsilon = learn_epsilon
        if self.has_continuous_action_space and not learn_epsilon:
            self.epsilon = epsilon
            self.epsilon_decay_rate = epsilon_decay_rate
            self.epsilon_decay_every_updates = epsilon_decay_every_updates
            self.epsilon_min = epsilon_min
            self.updates_for_decay = self.epsilon_decay_every_updates

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.eps_clip = clip
        self.K_epochs = K_epochs
        
        self.memory = RolloutBuffer()

        self.policy = ActorCritic(self.device, state_dim, state_dim, action_dim, has_continuous_action_space, epsilon, learn_epsilon).to(self.device)

        modules = [
            {'params': self.policy.actor.parameters(), 'lr': lr_actor, 'weight_decay': weight_decay},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic, 'weight_decay': weight_decay},
        ]

        if self.has_continuous_action_space and self.learn_epsilon:
            modules += [{'params': [self.policy.log_std], 'lr': lr_std}]

        self.optimizer = torch.optim.Adam(modules)
        
        self.MseLoss = nn.MSELoss()

        self.policy.eval()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                action = (
                    self.policy.actor(state)
                    if self.has_continuous_action_space
                    else self.policy.actor(state).argmax(dim=-1)
                )
            else:
                action = self.policy.act(state)
        return action.detach().cpu().numpy().flatten() if self.has_continuous_action_space else action.item()

    def update(self):

        total_actor_loss = 0
        total_critic_loss = 0

        self.policy.train()

        # Extract stored data from memory
        rewards = self.memory.rewards
        is_terminals = self.memory.is_terminals

        with torch.no_grad():

            # old_states = torch.squeeze(torch.stack(self.memory.states, dim=0)).to(self.device)
            # old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).to(self.device)
            # old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0)).to(self.device)

            old_states = torch.as_tensor(np.array(self.memory.states), dtype=torch.float32).to(self.device)
            action_dtype = torch.float32 if self.has_continuous_action_space else torch.long
            old_actions = torch.as_tensor(
                np.array(self.memory.actions), dtype=action_dtype
            ).to(self.device)

            dist = self.policy.action_distribution(old_states)
            if self.has_continuous_action_space:
                old_actions = old_actions.clamp(-1 + 1e-6, 1 - 1e-6)
            old_logprobs = dist.log_prob(old_actions).to(self.device)

        for _ in range(self.K_epochs):

            # Compute advantages and returns using GAE
            old_state_values = self.policy.critic(old_states).squeeze(-1)
            advantages, returns = self.compute_gae(rewards, old_state_values, is_terminals)

            # Create a DataLoader for batching
            dataset = TensorDataset(old_states, old_actions, old_logprobs, advantages, returns)
            # batch_size = 64  # Define a batch size
            batch_size = len(rewards)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for batch in loader:
                # Unpack the batch
                batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns = batch

                # Evaluate old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)

                # Match state_values tensor dimensions with batch_returns tensor
                state_values = state_values.squeeze(-1)

                # Calculate the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - batch_logprobs.detach())

                # Calculate Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                actor_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
                critic_loss = 0.5 * self.MseLoss(state_values, batch_returns)

                # Final loss (PPO clipped objective)
                loss = actor_loss + critic_loss

                # Take a gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                # torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
                self.optimizer.step()

                # print(actor_loss)
                # print(critic_loss)

                total_actor_loss += actor_loss.mean().item()
                total_critic_loss += critic_loss.item()

        # Clear the buffer
        self.memory.clear()

        self.policy.eval()

        if not self.learn_epsilon:
            self.updates_for_decay -= 1
            # update epsilon (exploration rate)
            if self.updates_for_decay == 0:
                self.updates_for_decay = self.epsilon_decay_every_updates
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay_rate
                    # print(f'epsilon updated to {self.epsilon}')
                    self.policy.set_action_std(self.epsilon)

        metrics = {
            'actor_loss': total_actor_loss / self.K_epochs,
            'critic_loss': total_critic_loss / self.K_epochs
        }

        if self.has_continuous_action_space and not self.learn_epsilon:
            metrics['epsilon'] = self.epsilon
        
        return metrics

    def compute_gae(self, rewards, values, is_terminals):
        """
        Compute Generalized Advantage Estimation (GAE).
        Args:
            rewards: List of rewards for each timestep.
            values: List of value function estimates for each timestep.
            is_terminals: List of booleans indicating terminal states.
        Returns:
            advantages: Computed GAE advantages.
            returns: Computed returns for each timestep.
        """
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(-1)
        terminals = torch.as_tensor(is_terminals, dtype=torch.float32, device=self.device)
        if not (len(rewards) == len(values) == len(terminals)):
            raise ValueError("Rewards, values, and terminal flags must have equal lengths.")

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        next_value = torch.zeros((), dtype=torch.float32, device=self.device)

        for step in reversed(range(len(rewards))):
            continuation = 1.0 - terminals[step]
            delta = rewards[step] + self.gamma * next_value * continuation - values[step]
            gae = delta + self.gamma * self.gae_lambda * continuation * gae
            advantages[step] = gae
            next_value = values[step]

        returns = advantages + values

        # Normalize advantages for better optimization stability
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-7
        )

        return advantages, returns

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint)
        self.policy.to(self.device)