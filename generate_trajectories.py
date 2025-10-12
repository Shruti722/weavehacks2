"""
Generate trajectories for RL training
Creates 10 scenarios × 5 trajectories each = 50 total trajectories
Each scenario uses the same initial grid state but explores different agent decisions
"""

import os
import json
from pathlib import Path
from datetime import datetime
import weave
from dotenv import load_dotenv

from multi_agent_system import SynErgiMultiAgentSystem
from data.simulator import GridSimulator

load_dotenv()

# Initialize Weave
weave.init(os.getenv("WEAVE_PROJECT", "synergi-rl-training"))


@weave.op()
def generate_single_trajectory(mas, scenario_id, trajectory_id):
    """Generate a single trajectory for GRPO training"""
    print(f"  Trajectory {trajectory_id + 1}/5...")

    # Get simulator from MAS
    simulator = mas.grid_simulator

    # Track states and actions
    states = []
    actions = []

    # Generate initial state
    initial_state = simulator.generate_tick()
    states.append({
        "demand_mw": initial_state['kpis']['city_demand_mw'],
        "supply_mw": initial_state['kpis']['city_supply_mw'],
        "deficit_mw": initial_state['kpis']['city_demand_mw'] - initial_state['kpis']['city_supply_mw'],
        "time": initial_state['drivers']['time_of_day']
    })

    # Run the multi-agent system
    result = mas.run(initial_state)

    # Extract full conversation as actions (messages + tool calls)
    messages = result.get('conversation_history', [])
    for msg in messages:
        action = {
            "role": msg.__class__.__name__ if hasattr(msg, '__class__') else "unknown"
        }

        # Add message content
        if hasattr(msg, 'content'):
            action["content"] = msg.content

        # Add tool calls if present
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            action["tool_calls"] = [
                {
                    "name": tc.get("name", "unknown"),
                    "args": tc.get("args", {})
                }
                for tc in msg.tool_calls
            ]

        actions.append(action)

    # Get final state
    final_state = simulator.generate_tick()
    states.append({
        "demand_mw": final_state['kpis']['city_demand_mw'],
        "supply_mw": final_state['kpis']['city_supply_mw'],
        "deficit_mw": final_state['kpis']['city_demand_mw'] - final_state['kpis']['city_supply_mw'],
        "time": final_state['drivers']['time_of_day']
    })

    # Calculate reward as negative cost (lower cost = higher reward)
    cost = simulator.cumulative_cost
    reward = -cost

    # GRPO format: states, actions, reward (negative cost)
    trajectory_data = {
        "scenario_id": scenario_id,
        "trajectory_id": trajectory_id,
        "states": states,
        "actions": actions,
        "reward": reward,
        "cost": cost
    }

    return trajectory_data


@weave.op()
def generate_scenario(scenario_id, num_trajectories=5):
    """Generate multiple trajectories for the same scenario (same initial grid state)"""
    print(f"\nScenario {scenario_id + 1}/10:")

    trajectories = []

    for traj_id in range(num_trajectories):
        # Create fresh MAS instance for each trajectory
        mas = SynErgiMultiAgentSystem()
        simulator = mas.grid_simulator

        # Set random seed for this scenario
        import random
        random.seed(scenario_id * 1000 + traj_id)

        # Advance simulator to a random deficit scenario
        for _ in range(scenario_id * 10 + random.randint(5, 15)):
            simulator.generate_tick()

        trajectory = generate_single_trajectory(mas, scenario_id, traj_id)
        trajectories.append(trajectory)

    return trajectories


def main():
    """Generate all trajectories and save grouped by scenario"""
    print("=" * 60)
    print("Generating 50 Trajectories (10 scenarios × 5 trajectories)")
    print("=" * 60)

    output_dir = Path("artifacts/trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scenarios = []

    for scenario_id in range(10):
        trajectories = generate_scenario(scenario_id)

        scenario_data = {
            "scenario_id": scenario_id,
            "num_trajectories": len(trajectories),
            "trajectories": trajectories
        }

        all_scenarios.append(scenario_data)

        # Save individual scenario file
        scenario_file = output_dir / f"scenario_{scenario_id:02d}.json"
        with open(scenario_file, 'w') as f:
            json.dump(scenario_data, f, indent=2)

        print(f"  Saved to {scenario_file}")

        # Print scenario summary
        rewards = [t['reward'] for t in trajectories]
        avg_reward = sum(rewards) / len(rewards)
        print(f"  Avg reward: {avg_reward:.2f}, Success rate: {int(sum(rewards))}/{len(rewards)}")

    # Save combined file
    combined_file = output_dir / "all_trajectories.json"
    with open(combined_file, 'w') as f:
        json.dump(all_scenarios, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Generated 50 trajectories")
    print(f"✓ Saved to {output_dir}/")
    print(f"  - 10 scenario files (scenario_00.json to scenario_09.json)")
    print(f"  - 1 combined file (all_trajectories.json)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
