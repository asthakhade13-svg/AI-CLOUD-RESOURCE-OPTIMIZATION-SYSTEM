# rl/agent.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple

class RolloutBuffer:
    """Stores experience trajectories for PPO updates."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

class ActorCritic(nn.Module):
    """Actor-Critic network architecture for PPO."""
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorCritic, self).__init__()
        
        # Actor Network (policy distribution)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic Network (value estimator)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Outputs action, its log probability, and state value estimation."""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        
        return action, action_logprob, state_val
        
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates batch of states and actions for updates."""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    """PPO Agent implementing select_action and policy parameter update steps."""
    def __init__(
        self, 
        state_dim: int = 15, 
        action_dim: int = 5, 
        lr_actor: float = 0.0003, 
        lr_critic: float = 0.001, 
        gamma: float = 0.99, 
        K_epochs: int = 5, 
        eps_clip: float = 0.2,
        entropy_coef: float = 0.01
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        self.buffer = RolloutBuffer()
        
        # Active policy networks
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        # Save old policy weights for surrogate calculations
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state: np.ndarray) -> int:
        """Selects action index from state observation vector."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            action, action_logprob, _ = self.policy_old.act(state_t)
            
        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action.item()
        
    def update(self):
        """Updates PPO policy parameters based on rollout experiences."""
        # Monte Carlo estimate of state returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Convert list to tensor and normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert buffer lists to tensors
        old_states = torch.stack(self.buffer.states).detach()
        old_actions = torch.stack(self.buffer.actions).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old states and actions
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Find ratio: pi_theta / pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Generalized Advantage Estimation (GAE)
            advantages = rewards - state_values.detach()
            
            # Clipped surrogate objective loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Combined Loss: policy gradient + value network baseline + entropy encouragement
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy current policy weights to old policy weights
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
    def save(self, checkpoint_path: str):
        """Saves current policy weights checkpoint."""
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        
    def load(self, checkpoint_path: str):
        """Loads policy weights from checkpoint."""
        self.policy.load_state_dict(torch.load(checkpoint_path))
        self.policy_old.load_state_dict(self.policy.state_dict())
