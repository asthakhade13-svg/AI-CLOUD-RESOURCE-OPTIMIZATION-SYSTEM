import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_visualizations(csv_path="data/cleaned_workload.csv", output_dir="data/plots"):
    """
    Generates 6 production-quality visualizations representing the historical cloud workloads.
    """
    print(f"\nGenerating workload visualizations from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {csv_path}. Please run pipeline first.")
        
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Set modern plotting aesthetics
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We will plot the first 168 hours (1 week) for better chart readability
    plot_df = df.head(168).copy()
    
    # Define color scheme
    primary_color = "#1f77b4"  # Cool blue
    secondary_color = "#ff7f0e"  # Warm orange
    accent_color = "#2ca02c"  # Green
    grid_color = "#e6e6e6"
    
    # 1. CPU Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["timestamp"], plot_df["cpu_usage"], color=primary_color, linewidth=2, label="CPU Utilization (%)")
    plt.fill_between(plot_df["timestamp"], plot_df["cpu_usage"], color=primary_color, alpha=0.1)
    plt.title("Cloud Infrastructure CPU Utilization Profile (1 Week)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time", fontsize=11, labelpad=10)
    plt.ylabel("CPU Usage (%)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6, color=grid_color)
    plt.ylim(0, 110)
    plt.legend(loc="upper left")
    plt.tight_layout()
    cpu_path = os.path.join(output_dir, "cpu_over_time.png")
    plt.savefig(cpu_path, dpi=150)
    plt.close()
    print(f"-> Saved CPU plot to {cpu_path}")
    
    # 2. Memory Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["timestamp"], plot_df["memory_usage"], color="#9467bd", linewidth=2, label="Memory Utilization (%)")
    plt.fill_between(plot_df["timestamp"], plot_df["memory_usage"], color="#9467bd", alpha=0.1)
    plt.title("Cloud Infrastructure Memory Utilization Profile (1 Week)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time", fontsize=11, labelpad=10)
    plt.ylabel("Memory Usage (%)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6, color=grid_color)
    plt.ylim(0, 110)
    plt.legend(loc="upper left")
    plt.tight_layout()
    mem_path = os.path.join(output_dir, "memory_over_time.png")
    plt.savefig(mem_path, dpi=150)
    plt.close()
    print(f"-> Saved Memory plot to {mem_path}")
    
    # 3. Active Users Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["timestamp"], plot_df["active_users"], color=secondary_color, linewidth=2, label="Active Sessions")
    plt.fill_between(plot_df["timestamp"], plot_df["active_users"], color=secondary_color, alpha=0.1)
    plt.title("Workload Demand: Active User Sessions Profile (1 Week)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time", fontsize=11, labelpad=10)
    plt.ylabel("Active User Sessions", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6, color=grid_color)
    plt.legend(loc="upper left")
    plt.tight_layout()
    users_path = os.path.join(output_dir, "active_users_over_time.png")
    plt.savefig(users_path, dpi=150)
    plt.close()
    print(f"-> Saved Active Users plot to {users_path}")
    
    # 4. Network Traffic Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["timestamp"], plot_df["network_traffic"], color="#2ca02c", linewidth=2.5, label="Total Traffic")
    plt.plot(plot_df["timestamp"], plot_df["network_in"], color="#17becf", linewidth=1.5, alpha=0.7, label="Network In")
    plt.plot(plot_df["timestamp"], plot_df["network_out"], color="#bcbd22", linewidth=1.5, alpha=0.7, label="Network Out")
    plt.title("System Networking Load Profile (1 Week)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time", fontsize=11, labelpad=10)
    plt.ylabel("Throughput (Mbps)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6, color=grid_color)
    plt.legend(loc="upper left")
    plt.tight_layout()
    net_path = os.path.join(output_dir, "network_traffic_over_time.png")
    plt.savefig(net_path, dpi=150)
    plt.close()
    print(f"-> Saved Network Traffic plot to {net_path}")
    
    # 5. Server Count Over Time (Current vs Required)
    plt.figure(figsize=(12, 5))
    plt.step(plot_df["timestamp"], plot_df["current_servers"], where="post", color="#d62728", linewidth=2, label="Current Active Servers")
    plt.step(plot_df["timestamp"], plot_df["required_servers"], where="post", color="#2ca02c", linewidth=2, linestyle="--", alpha=0.8, label="Optimal Required Servers")
    plt.title("Resource Scaling Evaluation: Current vs. Required Server Count", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time", fontsize=11, labelpad=10)
    plt.ylabel("Server Quantity", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6, color=grid_color)
    plt.legend(loc="upper left")
    plt.tight_layout()
    srv_path = os.path.join(output_dir, "server_count_over_time.png")
    plt.savefig(srv_path, dpi=150)
    plt.close()
    print(f"-> Saved Server Count plot to {srv_path}")
    
    # 6. Response Time Over Time
    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["timestamp"], plot_df["response_time"], color="#e377c2", linewidth=2, label="Response Latency")
    # Draw horizontal line showing acceptable SLA (e.g. 300ms)
    plt.axhline(y=300.0, color="red", linestyle=":", linewidth=1.5, label="SLA Target (300ms)")
    plt.title("Service Quality Level: Request Response Time & SLA (1 Week)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time", fontsize=11, labelpad=10)
    plt.ylabel("Response Time (ms)", fontsize=11, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6, color=grid_color)
    plt.legend(loc="upper left")
    plt.tight_layout()
    resp_path = os.path.join(output_dir, "response_time_over_time.png")
    plt.savefig(resp_path, dpi=150)
    plt.close()
    print(f"-> Saved Response Time plot to {resp_path}")
    
    print("\nVisualizations successfully generated and saved to output folder.")

if __name__ == "__main__":
    generate_visualizations()
