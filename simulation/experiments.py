# simulation/experiments.py

from typing import Dict, Any, List, Optional
from simulation.scenarios import WhatIfAnalyzer

class ExperimentSuite:
    """
    Framework to configure and run multi-algorithm policy comparison experiments
    on the Cloud Digital Twin environment.
    """
    def __init__(self, simulator_config: Dict[str, Any] = None):
        self.analyzer = WhatIfAnalyzer(simulator_config)
        
    def run_policy_comparison(
        self, 
        scenario_config: Dict[str, Any], 
        ppo_agent: Optional[Any] = None,
        model_loaded: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Runs the specified scenario against all 5 autoscaling policies
        and aggregates benchmarking outcomes.
        """
        policies = ["STATIC", "THRESHOLD", "HPA", "ML_PREDICTIVE", "RL_PPO"]
        benchmark_results = []
        
        for pol in policies:
            # Clone config and assign target policy
            run_config = scenario_config.copy()
            run_config["policy_name"] = pol
            
            # Execute scenario run
            res = self.analyzer.run_custom_scenario(
                config=run_config,
                ppo_agent=ppo_agent,
                model_loaded=model_loaded
            )
            
            # Extract summary statistics
            summary = res["summary"]
            summary["policy"] = pol
            benchmark_results.append(summary)
            
        return benchmark_results
