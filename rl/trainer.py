# rl/trainer.py

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from rl.environment import CloudAutoscalingEnv
from rl.agent import PPOAgent

# Artifact plots output directory (dynamically resolves standard Antigravity workspace paths)
PLOTS_DIR = "artifacts"
MODELS_DIR = "rl/models"

def set_seed(seed: int = 42):
    """Sets random seeds across random, numpy, and torch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_ppo_agent(
    episodes: int = 80, 
    seed: int = 42,
    checkpoint_name: str = "ppo_autoscaler.pth"
) -> Dict[str, List[float]]:
    """
    Trains the PPO agent in the CloudAutoscalingEnv simulation.
    Saves the final weights and plots training metrics.
    """
    set_seed(seed)
    
    # Initialize components
    env = CloudAutoscalingEnv()
    agent = PPOAgent(
        state_dim=15,
        action_dim=5,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=5,
        eps_clip=0.2
    )
    
    # Metric accumulators across episodes
    episode_rewards = []
    episode_costs = []
    episode_sla_violations = []
    episode_latencies = []
    episode_replicas = []
    episode_utilization = []
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, checkpoint_name)
    
    print(f"Starting PPO Agent Training for {episodes} episodes...")
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        
        ep_reward = 0.0
        ep_costs = []
        ep_latencies = []
        ep_replicas = []
        ep_utilization = []
        ep_sla_count = 0
        
        while not done:
            # 1. Action selection
            action_idx = agent.select_action(state)
            
            # 2. Environment step execution
            next_state, reward, done, info = env.step(action_idx)
            
            # Save experience in agent rollout buffer
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            # Record tracking metrics
            ep_reward += reward
            ep_costs.append(info["hourly_cost"])
            ep_latencies.append(info["response_time"])
            ep_replicas.append(info["current_servers"])
            ep_utilization.append(info["cpu_usage"])
            if info["sla_status"] == "VIOLATED":
                ep_sla_count += 1
                
            state = next_state
            
        # 3. Update Policy parameters at the end of the episode
        agent.update()
        
        # Log episodic averages
        avg_cost = np.mean(ep_costs)
        avg_latency = np.mean(ep_latencies)
        avg_replicas = np.mean(ep_replicas)
        avg_util = np.mean(ep_utilization)
        
        episode_rewards.append(ep_reward)
        episode_costs.append(avg_cost)
        episode_sla_violations.append(ep_sla_count)
        episode_latencies.append(avg_latency)
        episode_replicas.append(avg_replicas)
        episode_utilization.append(avg_util)
        
        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep:02d}/{episodes:02d} | Reward: {ep_reward:7.2f} | SLA Violations: {ep_sla_count:3d} | Replicas: {avg_replicas:4.1f} | CPU Util: {avg_util:4.1f}%")
            
    # Save the trained model parameters
    agent.save(checkpoint_path)
    print(f"Saved trained PPO model to {checkpoint_path}")
    
    # Generate and save metric plots
    generate_training_plots(
        episode_rewards, episode_costs, episode_sla_violations,
        episode_replicas, episode_latencies, episode_utilization
    )
    
    return {
        "rewards": episode_rewards,
        "costs": episode_costs,
        "sla_violations": episode_sla_violations,
        "replicas": episode_replicas,
        "latencies": episode_latencies,
        "utilizations": episode_utilization
    }

def generate_training_plots(
    rewards: List[float],
    costs: List[float],
    sla_violations: List[int],
    replicas: List[float],
    latencies: List[float],
    utilizations: List[float]
):
    """Creates evaluation graphs for metrics across episodes and saves them as artifacts."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs("rl/plots", exist_ok=True)
    
    epochs = range(1, len(rewards) + 1)
    
    # Figure 1: Episodic Cumulative Reward
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, rewards, color="#0ea5e9", linewidth=2)
    plt.title("PPO Autoscale Training: Cumulative Reward Progression")
    plt.xlabel("Episode")
    plt.ylabel("Reward Score")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rl_episode_reward.png"), dpi=150)
    plt.savefig("rl/plots/rl_episode_reward.png", dpi=150)
    plt.close()
    
    # Figure 2: Operational Cost
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, costs, color="#ec4899", linewidth=2)
    plt.title("PPO Autoscale Training: Average Hourly Infrastructure Cost")
    plt.xlabel("Episode")
    plt.ylabel("Cost ($/hr)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rl_cost.png"), dpi=150)
    plt.savefig("rl/plots/rl_cost.png", dpi=150)
    plt.close()
    
    # Figure 3: SLA Violations
    plt.figure(figsize=(10, 4))
    plt.bar(epochs, sla_violations, color="#ef4444", alpha=0.8)
    plt.title("PPO Autoscale Training: SLA Violations Frequency Count")
    plt.xlabel("Episode")
    plt.ylabel("Violations Count")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rl_sla_violations.png"), dpi=150)
    plt.savefig("rl/plots/rl_sla_violations.png", dpi=150)
    plt.close()

    # Figure 4: Operational Metrics Grid (Replicas, Latency, CPU)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(epochs, replicas, color="#64748b", linewidth=2)
    axes[0].set_title("Average Replica Count")
    axes[0].set_ylabel("Servers")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    axes[1].plot(epochs, latencies, color="#f59e0b", linewidth=2)
    axes[1].set_title("Average Latency Response Time")
    axes[1].set_ylabel("ms")
    axes[1].axhline(y=200.0, color="#ef4444", linestyle="--", alpha=0.5, label="SLA Threshold")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    axes[2].plot(epochs, utilizations, color="#a855f7", linewidth=2)
    axes[2].set_title("Average CPU Resource Utilization")
    axes[2].set_ylabel("CPU (%)")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    
    plt.xlabel("Episode")
    plt.suptitle("PPO Autoscaling Performance Metrics Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rl_metrics_summary.png"), dpi=150)
    plt.savefig("rl/plots/rl_metrics_summary.png", dpi=150)
    plt.close()
    
    print(f"Generated and saved RL training plots under {PLOTS_DIR}/")

if __name__ == "__main__":
    train_ppo_agent()
