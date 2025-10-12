"""
Episode Collection Script for RL Training
Collects episodes and logs them as Weave traces

Usage:
    python scripts/collect_episodes.py --num 2000 --name iteration_0
"""

import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weave
from multi_agent_system import SynErgiMultiAgentSystem
from data.simulator import GridSimulator


@weave.op()
def collect_single_episode(mas: SynErgiMultiAgentSystem, simulator: GridSimulator, episode_num: int):
    """
    Collect a single episode with full trajectory.
    This automatically logs as a Weave trace.

    Returns:
        dict: Episode summary with reward and metadata
    """
    # Generate grid state
    grid_state = simulator.generate_tick()

    # Run multi-agent system (already traced with @weave.op())
    result = mas.run(grid_state)

    # Extract reward and key metrics
    reward_data = result.get("actuator_output", {}).get("reward_data", {})

    episode_summary = {
        "episode_num": episode_num,
        "timestamp": datetime.now().isoformat(),

        # Reward information
        "reward": reward_data.get("reward", 0),
        "raw_score": reward_data.get("raw_score", 0),
        "threshold": reward_data.get("threshold", 0.7),

        # Component scores
        "deficit_score": reward_data.get("deficit_score", 0),
        "cost_score": reward_data.get("cost_score", 0),
        "risk_score": reward_data.get("risk_score", 0),
        "fairness_score": reward_data.get("fairness_score", 0),
        "violation_score": reward_data.get("violation_score", 0),

        # Metrics
        "cost_usd": reward_data.get("cost_usd", 0),
        "deficit_before_mw": reward_data.get("deficit_before_mw", 0),
        "deficit_after_mw": reward_data.get("deficit_after_mw", 0),
        "deficit_improvement_mw": reward_data.get("deficit_improvement_mw", 0),

        # Grid state metadata
        "initial_demand_mw": grid_state.get("kpis", {}).get("city_demand_mw", 0),
        "initial_supply_mw": grid_state.get("kpis", {}).get("city_supply_mw", 0),
        "time_of_day": grid_state.get("drivers", {}).get("time_of_day", ""),

        # Full result for reference (Weave trace has full conversation)
        "has_full_trace": True
    }

    return episode_summary


@weave.op()
def collect_episode_batch(num_episodes: int, model_name: str = "Qwen/Qwen2.5-14B-Instruct",
                         checkpoint_interval: int = 100):
    """
    Collect a batch of episodes with progress tracking.

    Args:
        num_episodes: Number of episodes to collect
        model_name: Base model to use (or fine-tuned version)
        checkpoint_interval: Print progress every N episodes

    Returns:
        dict: Summary statistics and list of episodes
    """
    print("=" * 70)
    print(f"🚀 Starting Episode Collection")
    print("=" * 70)
    print(f"Target episodes: {num_episodes}")
    print(f"Model: {model_name}")
    print(f"Checkpoint interval: {checkpoint_interval}")
    print()

    # Initialize system
    print("Initializing multi-agent system...")
    mas = SynErgiMultiAgentSystem(model_name=model_name)
    simulator = GridSimulator()
    print("✓ System initialized\n")

    episodes = []
    start_time = datetime.now()

    for i in range(num_episodes):
        try:
            # Collect episode (automatically creates Weave trace)
            episode = collect_single_episode(mas, simulator, i)
            episodes.append(episode)

            # Progress checkpoint
            if (i + 1) % checkpoint_interval == 0 or i == 0:
                success_count = sum(1 for e in episodes if e["reward"] == 1)
                success_rate = success_count / len(episodes)
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / len(episodes)
                remaining_time = avg_time * (num_episodes - len(episodes)) / 60

                print(f"📊 Episode {i + 1}/{num_episodes}")
                print(f"   Success rate: {success_rate:.1%} ({success_count}/{len(episodes)})")
                print(f"   Avg time/episode: {avg_time:.1f}s")
                print(f"   Est. remaining: {remaining_time:.1f} min")
                print()

        except KeyboardInterrupt:
            print("\n⚠️  Collection interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error on episode {i}: {e}")
            # Continue with next episode
            continue

    # Final statistics
    total_time = (datetime.now() - start_time).total_seconds()
    success_count = sum(1 for e in episodes if e["reward"] == 1)
    success_rate = success_count / len(episodes) if episodes else 0

    avg_raw_score = sum(e["raw_score"] for e in episodes) / len(episodes) if episodes else 0
    avg_deficit_improvement = sum(e["deficit_improvement_mw"] for e in episodes) / len(episodes) if episodes else 0
    total_cost = sum(e["cost_usd"] for e in episodes)

    summary = {
        "total_episodes": len(episodes),
        "successful_episodes": success_count,
        "failed_episodes": len(episodes) - success_count,
        "success_rate": success_rate,
        "avg_raw_score": avg_raw_score,
        "avg_deficit_improvement_mw": avg_deficit_improvement,
        "total_cost_usd": total_cost,
        "total_time_seconds": total_time,
        "model_used": model_name,
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes
    }

    print("=" * 70)
    print("✅ Collection Complete!")
    print("=" * 70)
    print(f"Total episodes: {len(episodes)}")
    print(f"Successful: {success_count} ({success_rate:.1%})")
    print(f"Failed: {len(episodes) - success_count} ({1-success_rate:.1%})")
    print(f"Avg raw score: {avg_raw_score:.3f}")
    print(f"Avg deficit improvement: {avg_deficit_improvement:.1f} MW")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect RL training episodes")
    parser.add_argument("--num", type=int, default=2000, help="Number of episodes to collect")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                       help="Model name to use")
    parser.add_argument("--name", type=str, default="training_batch",
                       help="Name for this collection batch")
    parser.add_argument("--checkpoint", type=int, default=100,
                       help="Print progress every N episodes")

    args = parser.parse_args()

    # Initialize Weave
    weave.init("synergi-rl-training")

    print(f"🔍 Weave project: synergi-rl-training")
    print(f"📝 Batch name: {args.name}")
    print()

    # Collect episodes
    summary = collect_episode_batch(
        num_episodes=args.num,
        model_name=args.model,
        checkpoint_interval=args.checkpoint
    )

    # Publish summary as Weave Dataset
    dataset_name = f"episodes_{args.name}"
    weave.publish(summary, name=dataset_name)

    print(f"💾 Published to Weave as dataset: '{dataset_name}'")
    print(f"🔗 View at: https://wandb.ai/ruchib-northwestern-university/synergi-rl-training/weave")
    print()
    print("=" * 70)
    print("Next steps:")
    print("1. View traces in Weave dashboard")
    print("2. Run: python scripts/export_training_data.py")
    print("3. Fine-tune model on successful episodes")
    print("=" * 70)


if __name__ == "__main__":
    main()
