"""
ART GRPO Training for SynErgi Multi-Agent System
Adapted from OpenPipe ART examples for grid optimization
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import ART
import art
from art import TrajectoryGroup, TrainConfig
from art.serverless.backend import ServerlessBackend

# Import your multi-agent system
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from multi_agent_system import SynErgiMultiAgentSystem
from data.simulator import GridSimulator

load_dotenv()


class GridScenario:
    """Wrapper for a grid scenario with initial state"""
    def __init__(self, scenario_id: int, trajectory_id: int, initial_state: Dict[str, Any]):
        self.scenario_id = scenario_id
        self.trajectory_id = trajectory_id
        self.initial_state = initial_state


def load_scenarios(trajectory_dir: str = "artifacts/trajectories", limit: int = None):
    """Load training scenarios from trajectory files"""
    trajectory_path = Path(trajectory_dir)
    scenarios = []

    # Load all scenario files
    for scenario_file in sorted(trajectory_path.glob("scenario_*.json")):
        with open(scenario_file, 'r') as f:
            data = json.load(f)

        scenario_id = data['scenario_id']

        # Each trajectory is a training example
        for traj in data['trajectories']:
            scenario = GridScenario(
                scenario_id=scenario_id,
                trajectory_id=traj['trajectory_id'],
                initial_state=traj['states'][0]  # Use initial state as starting point
            )
            scenarios.append(scenario)

        if limit and len(scenarios) >= limit:
            scenarios = scenarios[:limit]
            break

    print(f"Loaded {len(scenarios)} training scenarios")
    return scenarios

def load_training_scenarios(trajectory_dir: str = "artifacts/trajectories", limit: int = None):
    scenarios = load_scenarios(trajectory_dir=trajectory_dir, limit=limit)

    return scenarios


def load_val_scenarios(trajectory_dir: str = "artifacts/trajectories", num_val_scenarios: int = 5):
    """
    Load a fixed set of validation scenarios.
    These initial states remain constant across all validation runs.
    """
    scenarios = load_scenarios(trajectory_dir=trajectory_dir, limit=num_val_scenarios)

    # Return only the first num_val_scenarios to keep validation set fixed
    return scenarios[:num_val_scenarios]

async def get_model_endpoint(model: art.TrainableModel) -> str:
    """
    Get the inference endpoint for the trained model.
    After training, the model should be deployed and accessible.
    """
    # Check if model has an inference endpoint
    if hasattr(model, 'inference_base_url') and model.inference_base_url:
        return model.name

    # Fallback: model might not be deployed yet
    return None


async def rollout(model: art.TrainableModel, scenario: GridScenario) -> art.Trajectory:
    """
    Generate a single trajectory by running the multi-agent system
    Returns a trajectory with states, actions, and reward
    """
    # Create a fresh simulator and multi-agent system
    # Try to use the trained model

    print(f"    Using trained model: {model.get_inference_name()}")
    mas = SynErgiMultiAgentSystem(model_name=model.get_inference_name())

    simulator = mas.grid_simulator

    # Set up initial state from scenario
    # Advance simulator to get a state similar to the scenario
    for _ in range(scenario.scenario_id * 10 + scenario.trajectory_id):
        simulator.generate_tick()

    initial_state = simulator.generate_tick()

    # Track states and actions
    states = []
    actions = []

    # Add initial state
    states.append({
        "demand_mw": initial_state['kpis']['city_demand_mw'],
        "supply_mw": initial_state['kpis']['city_supply_mw'],
        "deficit_mw": initial_state['kpis']['city_demand_mw'] - initial_state['kpis']['city_supply_mw']
    })

    # Run the multi-agent system
    result = mas.run(initial_state)

    # Extract actions from conversation
    messages = result.get('conversation_history', [])
    for msg in messages:
        action = {
            "role": msg.__class__.__name__ if hasattr(msg, '__class__') else "unknown"
        }
        if hasattr(msg, 'content'):
            action["content"] = msg.content
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            action["tool_calls"] = [
                {"name": tc.get("name", "unknown"), "args": tc.get("args", {})}
                for tc in msg.tool_calls
            ]
        actions.append(action)

    # Get final state
    final_state = simulator.generate_tick()
    states.append({
        "demand_mw": final_state['kpis']['city_demand_mw'],
        "supply_mw": final_state['kpis']['city_supply_mw'],
        "deficit_mw": final_state['kpis']['city_demand_mw'] - final_state['kpis']['city_supply_mw']
    })

    # Calculate reward (negative cost)
    cost = simulator.cumulative_cost
    reward = -cost

    # Convert to ART format: messages_and_choices
    # ART expects OpenAI chat completion format
    messages_and_choices = []
    for action in actions:
        # Map LangChain message types to OpenAI roles
        role_map = {
            "AIMessage": "assistant",
            "SystemMessage": "system",
            "HumanMessage": "user"
        }
        role = role_map.get(action.get("role", ""), "assistant")

        msg = {
            "role": role,
            "content": action.get("content", "")
        }

        # Only add tool_calls if they exist (not None or empty)
        if action.get("tool_calls"):
            msg["tool_calls"] = action["tool_calls"]

        messages_and_choices.append(msg)

    # Create ART trajectory
    trajectory = art.Trajectory(
        messages_and_choices=messages_and_choices,
        reward=reward,
        metadata={
            "scenario_id": scenario.scenario_id,
            "trajectory_id": scenario.trajectory_id,
            "cost": cost,
            "initial_deficit": states[0]["deficit_mw"],
            "final_deficit": states[-1]["deficit_mw"]
        }
    )

    return trajectory


async def run_validation_games(my_model: art.TrainableModel, benchmark_model: str, val_scenarios: List[GridScenario]):
    """
    Run validation games on fixed scenarios.
    The initial states are fixed, but agent interactions vary as the model learns.
    Returns trajectory groups with rewards (negative cost).
    """
    print(f"  Running validation on {len(val_scenarios)} fixed scenarios...")
    val_groups = []

    for scenario in val_scenarios:
        # Generate trajectories from both models
        val_traj = await rollout(my_model, scenario)


        # Create validation trajectory with win/loss reward
        val_traj = art.Trajectory(
            messages_and_choices=val_traj.messages_and_choices,
            reward=val_traj.reward,
            metadata={
                "validation": True,
                "trained_cost": -val_traj.reward,
                "scenario_id": scenario.scenario_id,
            }
        )

        val_groups.append(TrajectoryGroup([val_traj]))

    # Calculate average validation metrics
    avg_cost = sum(-g.trajectories[0].reward for g in val_groups) / len(val_groups)
    print(f"  Validation avg cost: ${avg_cost:.0f}")

    return val_groups


async def score_trajectory_group(group: TrajectoryGroup) -> TrajectoryGroup:
    """
    Score a group of trajectories based on their rewards
    In GRPO, we use relative rewards within each group

    Note: ART/GRPO handles the relative scoring internally,
    so we just return the group with rewards as-is
    """
    # Extract rewards for logging
    rewards = [traj.reward for traj in group.trajectories]

    # Log group statistics
    mean_reward = sum(rewards) / len(rewards)
    print(f"    Group rewards: {[f'{r:.0f}' for r in rewards]}, mean={mean_reward:.0f}")

    # ART handles GRPO scoring internally, so just return the group
    return group


def iterate_dataset(scenarios: List[GridScenario], groups_per_step: int, num_epochs: int, initial_step: int = 0):
    """Iterate over dataset in groups"""
    step = initial_step

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        for i in range(0, len(scenarios), groups_per_step):
            batch_scenarios = scenarios[i:i + groups_per_step]

            yield type('Batch', (), {
                'step': step,
                'items': batch_scenarios
            })()

            step += 1


async def main():
    """Main training loop for SynErgi multi-agent system"""
    print("=" * 60)
    print("ART GRPO Training for SynErgi Multi-Agent System")
    print("=" * 60)

    # Load training scenarios
    training_scenarios = load_training_scenarios(
        trajectory_dir="artifacts/trajectories",
        limit=10  # Start with 10 scenarios for testing
    )

    # Initialize model
    print("\nInitializing model...")
    model = art.TrainableModel(
        name="synergi-agent-v1",
        project="synergi-grid-optimization",
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )

    backend = ServerlessBackend()
    await model.register(backend)

    # Training configuration
    training_config = {
        "groups_per_step": 2,      # 2 scenarios per training step
        "num_epochs": 5,            # 5 epochs
        "rollouts_per_group": 4,    # 4 rollouts per scenario (for GRPO comparison)
        "learning_rate": 1e-5,
        "max_steps": 50,
    }

    print(f"\nTraining config:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")

    # Load fixed validation scenarios (same initial states used throughout training)
    print("\nLoading fixed validation scenarios...")
    val_scenarios = load_val_scenarios(num_val_scenarios=5)
    print(f"Loaded {len(val_scenarios)} validation scenarios with fixed initial states")

    # Training iterator
    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    # Training loop
    print("\nStarting training loop...")
    for batch in training_iterator:
        print(f"\n--- Step {batch.step} ---")

        # Generate trajectory groups
        print(f"Generating {len(batch.items)} trajectory groups...")
        groups = [
            TrajectoryGroup(
                [
                    await rollout(model, GridScenario(
                        scenario_id=scenario.scenario_id,
                        trajectory_id=i,
                        initial_state=scenario.initial_state
                    ))
                    for i in range(training_config["rollouts_per_group"])
                ]
            )
            for scenario in batch.items
        ]

        # Score trajectories using GRPO
        print("Scoring trajectory groups...")
        scored_groups = [
            await score_trajectory_group(group)
            for group in groups
        ]

        # Print statistics
        for i, group in enumerate(scored_groups):
            rewards = [traj.reward for traj in group.trajectories]
            costs = [traj.metadata['cost'] for traj in group.trajectories]
            print(f"  Group {i}: rewards={[f'{r:.0f}' for r in rewards]}, costs={[f'${c:.0f}' for c in costs]}")

        # Train model
        print("Training model...")
        await model.train(
            trajectory_groups=scored_groups,
            config=TrainConfig(learning_rate=training_config["learning_rate"]),
        )
        await model.delete_checkpoints()

        print(f"✓ Step {batch.step} complete")

        # Run validation every 2 steps (after training step completes)
        if (batch.step + 1) % 2 == 0:
            print("\nRunning validation on fixed scenarios...")
            val_groups = await run_validation_games(
                my_model=model,
                benchmark_model="Qwen/Qwen2.5-14B-Instruct",
                val_scenarios=val_scenarios,  # Same scenarios every time
            )
            await model.log(val_groups)

        # Stop after max steps
        if batch.step >= training_config["max_steps"]:
            print(f"\nReached max steps ({training_config['max_steps']})")
            break

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model: {model.name}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
