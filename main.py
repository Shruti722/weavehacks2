"""
SynErgi Main Orchestration Script

Runs one full simulation tick: Analyst → Planner → Actuator → Environment → Reward
"""

import json
from agents import AnalystAgent, PlannerAgent, ActuatorAgent
from env import DigitalTwinEnv
from utils import compute_reward, init_weave, log_state, log_metrics, create_trace_session


class SynErgiOrchestrator:
    """
    Main orchestrator for the multi-agent RL energy management system.

    Coordinates:
    - Agent execution flow (Analyst → Planner → Actuator)
    - Environment simulation (Digital Twin)
    - Reward computation
    - Tracing and logging
    """

    def __init__(self, config=None):
        """
        Initialize the orchestrator.

        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}

        # Initialize agents
        self.analyst = AnalystAgent()
        self.planner = PlannerAgent()
        self.actuator = ActuatorAgent()

        # Initialize environment
        self.env = DigitalTwinEnv()

        # Initialize Weave tracing
        project_name = self.config.get("weave_project", "synergi-grid-optimization")
        init_weave(project_name)

        # Tracking
        self.current_tick = 0
        self.session_id = None
        self.reward_weights = self.config.get("reward_weights", {
            "alpha": 1.0,
            "beta": 1.5,
            "gamma": 0.5
        })

    def run_tick(self, verbose=True):
        """
        Execute one complete simulation tick.

        Returns:
            dict: Results from the tick including state, reward, and agent outputs
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TICK {self.current_tick}")
            print(f"{'='*60}")

        # 1. Get current grid state
        current_state = self.env.get_state()
        if verbose:
            print(f"\n[1] Current Grid State")
            print(f"    Timestamp: {current_state['timestamp']}")
            print(f"    Total Cost: ${current_state['total_cost']:.2f}")
            print(f"    Fairness Index: {current_state['fairness_index']:.3f}")

        log_state(current_state, tag="pre_action_state")

        # 2. Run Analyst Agent
        if verbose:
            print(f"\n[2] Running AnalystAgent...")

        analysis = self.analyst.run(current_state)

        if verbose:
            print(f"    Trends: {len(analysis.get('trends', []))} nodes analyzed")
            print(f"    Risks: {len(analysis.get('risks', []))} high-risk nodes identified")

        # 3. Run Planner Agent
        if verbose:
            print(f"\n[3] Running PlannerAgent...")

        # Combine state and analysis for planner
        planner_input = {
            **current_state,
            "analysis": analysis
        }

        plan = self.planner.run(planner_input)

        if verbose:
            print(f"    Actions: {len(plan.get('actions', []))} node adjustments planned")
            improvement = plan.get("expected_improvement", {})
            print(f"    Expected: cost ↓{improvement.get('cost_reduction_percent', 0):.1f}%, "
                  f"risk ↓{improvement.get('risk_reduction_percent', 0):.1f}%")

        # 4. Run Actuator Agent
        if verbose:
            print(f"\n[4] Running ActuatorAgent...")

        # Combine everything for actuator
        actuator_input = {
            **planner_input,
            "plan": plan
        }

        execution = self.actuator.run(actuator_input)
        commands = execution.get("commands", [])

        if verbose:
            print(f"    Commands: {len(commands)} control commands prepared")

        # 5. Execute actions on Digital Twin
        if verbose:
            print(f"\n[5] Updating Digital Twin...")

        new_state = self.env.step(commands)

        if verbose:
            print(f"    New Total Cost: ${new_state['total_cost']:.2f}")
            print(f"    New Fairness Index: {new_state['fairness_index']:.3f}")

        log_state(new_state, tag="post_action_state")

        # 6. Compute Reward
        if verbose:
            print(f"\n[6] Computing Reward...")

        reward_result = compute_reward(new_state, self.reward_weights)
        total_reward = reward_result["total_reward"]

        if verbose:
            print(f"    Total Reward: {total_reward:.3f}")
            components = reward_result["components"]
            print(f"    Components: cost={components['cost_penalty']:.3f}, "
                  f"risk={components['risk_penalty']:.3f}, "
                  f"fairness={components['fairness_bonus']:.3f}")

        # Log metrics
        log_metrics({
            "tick": self.current_tick,
            "reward": total_reward,
            "cost": new_state["total_cost"],
            "fairness": new_state["fairness_index"]
        }, step=self.current_tick)

        # Increment tick counter
        self.current_tick += 1

        # Return comprehensive results
        return {
            "tick": self.current_tick - 1,
            "previous_state": current_state,
            "new_state": new_state,
            "analysis": analysis,
            "plan": plan,
            "execution": execution,
            "reward": reward_result
        }

    def run_episode(self, num_ticks=10, verbose=True):
        """
        Run a complete episode with multiple ticks.

        Args:
            num_ticks (int): Number of ticks to simulate
            verbose (bool): Whether to print detailed output

        Returns:
            dict: Episode summary with all tick results
        """
        # Create trace session
        self.session_id = create_trace_session(f"episode_{self.current_tick}")

        if verbose:
            print(f"\n{'#'*60}")
            print(f"STARTING EPISODE: {num_ticks} ticks")
            print(f"Session ID: {self.session_id}")
            print(f"{'#'*60}")

        # Reset environment
        self.env.reset()
        self.current_tick = 0

        # Run ticks
        tick_results = []
        cumulative_reward = 0

        for i in range(num_ticks):
            result = self.run_tick(verbose=verbose)
            tick_results.append(result)
            cumulative_reward += result["reward"]["total_reward"]

        # Episode summary
        avg_reward = cumulative_reward / num_ticks

        if verbose:
            print(f"\n{'#'*60}")
            print(f"EPISODE COMPLETE")
            print(f"{'#'*60}")
            print(f"Total Ticks: {num_ticks}")
            print(f"Cumulative Reward: {cumulative_reward:.3f}")
            print(f"Average Reward: {avg_reward:.3f}")
            print(f"{'#'*60}\n")

        return {
            "session_id": self.session_id,
            "num_ticks": num_ticks,
            "tick_results": tick_results,
            "cumulative_reward": cumulative_reward,
            "average_reward": avg_reward
        }

    def get_current_state(self):
        """Get the current grid state"""
        return self.env.get_state()

    def reset(self):
        """Reset the environment and counters"""
        self.env.reset()
        self.current_tick = 0


def main():
    """Main entry point for running the SynErgi system"""
    print("=" * 60)
    print("SynErgi - Multi-Agent Grid Optimization System")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = SynErgiOrchestrator()

    # Run a single tick as demonstration
    print("\nRunning single tick demonstration...\n")
    result = orchestrator.run_tick(verbose=True)

    # Print final state
    print(f"\n{'='*60}")
    print("Tick completed successfully!")
    print(f"{'='*60}\n")

    # Optionally run a full episode
    # orchestrator.run_episode(num_ticks=10, verbose=True)


if __name__ == "__main__":
    main()
