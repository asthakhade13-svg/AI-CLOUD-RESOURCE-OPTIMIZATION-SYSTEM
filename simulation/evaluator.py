# simulation/evaluator.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from simulation.scenarios import WhatIfAnalyzer
from simulation.experiments import ExperimentSuite
from rl.agent import PPOAgent

def run_simulation_benchmarks(checkpoint_path: str = "rl/models/ppo_autoscaler.pth"):
    """
    Executes a standard evaluation experiment under stress traffic spikes 
    and pod failure injections, outputting Markdown reports and performance plots.
    """
    # 1. Initialize PPO agent
    agent = PPOAgent(state_dim=15, action_dim=5)
    model_loaded = False
    if os.path.exists(checkpoint_path):
        try:
            agent.load(checkpoint_path)
            model_loaded = True
        except Exception:
            pass
            
    # 2. Define standard test scenario
    scenario_config = {
        "initial_replicas": 5,
        "max_steps": 288,
        "traffic_multiplier": 1.2, # 20% traffic surge
        "users_multiplier": 1.1,
        "workload_patterns": [
            {
                "type": "sinusoidal",
                "params": {
                    "amplitude": 500.0,
                    "bias": 1000.0,
                    "period": 288.0,
                    "requests_per_user": 4.5
                }
            },
            {
                "type": "spike",
                "params": {
                    "start": 100,
                    "duration": 30,
                    "amplitude": 1200.0,
                    "requests_per_user": 5.0
                }
            }
        ],
        "failures": [
            {"step": 140, "type": "pod_crash", "value": 2}, # 2 pods crash at step 140
            {"step": 180, "type": "slowdown", "value": 2.0} # dependency slowdown at step 180
        ]
    }
    
    # 3. Run all policy evaluations and print summary markdown table
    suite = ExperimentSuite()
    comparison = suite.run_policy_comparison(scenario_config, agent, model_loaded)
    
    df_compare = pd.DataFrame(comparison)
    # Re-order columns for readability
    cols = ["policy", "total_cost", "avg_latency", "p95_latency", "sla_violations", "scaling_events", "over_provisioning_steps", "under_provisioning_steps", "avg_cpu_utilization", "recovery_steps"]
    df_compare = df_compare[[c for c in cols if c in df_compare.columns]]
    
    print("\n--- DIGITAL TWIN POLICY BENCHMARKS ---")
    print(df_compare.to_markdown(index=False))
    
    # 4. Generate visual metrics graphs for each policy history logs
    analyzer = WhatIfAnalyzer()
    histories = {}
    
    for pol in ["STATIC", "THRESHOLD", "HPA", "ML_PREDICTIVE", "RL_PPO"]:
        cfg = scenario_config.copy()
        cfg["policy_name"] = pol
        res = analyzer.run_custom_scenario(cfg, agent, model_loaded)
        histories[pol] = pd.DataFrame(res["history"])
        
    # Plotting comparison grids
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("simulation/plots", exist_ok=True)
    
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle("Digital Twin Policy Scaling Performance Comparison Under Stress", fontsize=16, y=0.98)
    
    # Standard colors
    colors = {
        "STATIC": "#7f8c8d",
        "THRESHOLD": "#e67e22",
        "HPA": "#2980b9",
        "ML_PREDICTIVE": "#9b59b6",
        "RL_PPO": "#2ecc71"
    }
    
    # Subplot 1: Workload (Request rate)
    ax_work = axs[0, 0]
    # Workload is identical across runs, pull from STATIC
    df_stat = histories["STATIC"]
    ax_work.plot(df_stat["step"], df_stat["requests"], label="Requests / sec", color="#2c3e50", linewidth=2)
    ax_work.axvline(x=100, color="red", linestyle="--", alpha=0.7, label="Traffic Spike Start")
    ax_work.axvline(x=140, color="orange", linestyle="--", alpha=0.7, label="Pod Crash Trigger")
    ax_work.set_title("Request Rate Load Profile")
    ax_work.set_xlabel("Simulation Step")
    ax_work.set_ylabel("Requests/sec")
    ax_work.legend()
    ax_work.grid(True)
    
    # Subplot 2: Replica Allocations
    ax_repl = axs[0, 1]
    for name, df in histories.items():
        ax_repl.step(df["step"], df["replicas"], label=name, color=colors[name], where="post", linewidth=1.5)
    ax_repl.set_title("Active Replica Instances Sizing")
    ax_repl.set_xlabel("Simulation Step")
    ax_repl.set_ylabel("Server Replicas Count")
    ax_repl.legend()
    ax_repl.grid(True)
    
    # Subplot 3: CPU Usage
    ax_cpu = axs[1, 0]
    for name, df in histories.items():
        ax_cpu.plot(df["step"], df["cpu"], label=name, color=colors[name], alpha=0.8, linewidth=1.2)
    ax_cpu.set_title("CPU Resource Utilization")
    ax_cpu.set_xlabel("Simulation Step")
    ax_cpu.set_ylabel("CPU Usage (%)")
    ax_cpu.legend()
    ax_cpu.grid(True)
    
    # Subplot 4: Memory Usage
    ax_mem = axs[1, 1]
    for name, df in histories.items():
        ax_mem.plot(df["step"], df["memory"], label=name, color=colors[name], alpha=0.8, linewidth=1.2)
    ax_mem.set_title("Memory Resource Utilization")
    ax_mem.set_xlabel("Simulation Step")
    ax_mem.set_ylabel("Memory Usage (%)")
    ax_mem.legend()
    ax_mem.grid(True)
    
    # Subplot 5: Latency response curve
    ax_lat = axs[2, 0]
    for name, df in histories.items():
        ax_lat.plot(df["step"], df["latency"], label=name, color=colors[name], alpha=0.8, linewidth=1.2)
    ax_lat.axhline(y=200, color="r", linestyle=":", label="SLA SLA Breach Target (200ms)")
    ax_lat.set_title("Response Latency Profile")
    ax_lat.set_xlabel("Simulation Step")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_yscale("log") # log scale due to exponential queuing tails
    ax_repl.legend()
    ax_lat.grid(True)
    
    # Subplot 6: Telemetry Error Rate
    ax_err = axs[2, 1]
    for name, df in histories.items():
        ax_err.plot(df["step"], df["errors"], label=name, color=colors[name], alpha=0.8, linewidth=1.2)
    ax_err.set_title("Telemetry Request Error Rate")
    ax_err.set_xlabel("Simulation Step")
    ax_err.set_ylabel("Error Rate (%)")
    ax_err.legend()
    ax_err.grid(True)
    
    # Subplot 7: Cumulative Infrastructure Cost
    ax_cost = axs[3, 0]
    for name, df in histories.items():
        ax_cost.plot(df["step"], df["cost"].cumsum(), label=name, color=colors[name], linewidth=1.8)
    ax_cost.set_title("Cumulative Infrastructure Hosting Cost")
    ax_cost.set_xlabel("Simulation Step")
    ax_cost.set_ylabel("Total Cost ($)")
    ax_cost.legend()
    ax_cost.grid(True)
    
    # Subplot 8: Summary Grid Box
    ax_summ = axs[3, 1]
    ax_summ.axis("off")
    summary_text = "Experiment Highlights:\n\n"
    for idx, row in df_compare.iterrows():
        summary_text += f"• {row['policy']}: Cost: ${row['total_cost']:.2f} | SLA Vio: {row['sla_violations']} | Events: {row['scaling_events']} | Recovery: {row['recovery_steps']} ticks\n"
    ax_summ.text(0.05, 0.2, summary_text, fontsize=11, family="monospace", bbox=dict(facecolor="#fcfcfc", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig("artifacts/digital_twin_comparison.png", dpi=150)
    plt.savefig("simulation/plots/digital_twin_comparison.png", dpi=150)
    plt.close()
    
    print("\nSimulation performance plots saved to artifacts/digital_twin_comparison.png")
    
if __name__ == "__main__":
    run_simulation_benchmarks()
