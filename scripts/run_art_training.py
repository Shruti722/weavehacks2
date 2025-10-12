#!/usr/bin/env python3
"""
Run OpenPipe ART Training on Existing Trajectories
Loads trajectories from artifacts/weave-traces/traces.jsonl
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Import ART components
from art.serverless.backend import ServerlessBackend
from art import TrainableModel, TrajectoryGroup, Trajectory, Messages, TrainConfig


def load_trajectories_from_jsonl(jsonl_path):
    """Load trajectories from the exported JSONL file"""
    from art import Messages

    trajectories = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            trace = json.loads(line)

            # Extract reward from output
            output = trace.get("output", {})
            reward = float(output.get("reward", 0.0))

            # Create messages from prompt/completion
            if "prompt" in trace and "completion" in trace:
                messages = [
                    {"role": "user", "content": str(trace["prompt"])},
                    {"role": "assistant", "content": str(trace["completion"])}
                ]

                # Create trajectory with required fields
                trajectory = Trajectory(
                    messages_and_choices=messages,
                    reward=reward
                )

                trajectories.append(trajectory)

    return trajectories


async def main():
    """Run ART training on existing trajectories"""

    print("=" * 70)
    print("OpenPipe ART Training - Existing Trajectories")
    print("=" * 70)
    print()

    # Read config from environment
    wandb_api_key = os.getenv("WANDB_API_KEY")
    openpipe_api_key = os.getenv("OPENPIPE_API_KEY")
    base_model = os.getenv("BASE_MODEL", "qwen2.5-14b")
    wandb_project = os.getenv("WANDB_PROJECT", "synergi-grid-optimization")

    if not wandb_api_key or not openpipe_api_key:
        print("❌ Missing API keys!")
        print("   Set WANDB_API_KEY and OPENPIPE_API_KEY in .env")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  Base Model: {base_model}")
    print(f"  W&B Project: {wandb_project}")
    print()

    # Load trajectories from JSONL
    jsonl_path = Path("artifacts/weave-traces/traces.jsonl")
    print(f"📂 Loading trajectories from {jsonl_path}...")
    trajectories = load_trajectories_from_jsonl(jsonl_path)
    print(f"✓ Loaded {len(trajectories)} trajectories")
    print()

    # Initialize serverless backend
    print("🚀 Initializing ServerlessBackend...")
    backend = ServerlessBackend(api_key=wandb_api_key)

    # Create trainable model
    print(f"📦 Creating TrainableModel...")
    model = TrainableModel(
        project=wandb_project,
        name="synergi-grid-agent",
        base_model=f"Qwen/{base_model.replace('qwen', 'Qwen')}-Instruct"
    )

    # Register model with backend (ASYNC - must await!)
    print("🔗 Registering model with backend...")
    await model.register(backend)

    print()
    print("=" * 70)
    print("✅ Starting Training on Existing Trajectories!")
    print("=" * 70)
    print()

    # Group trajectories for training
    # ART expects TrajectoryGroup objects
    trajectory_groups = []
    batch_size = 32

    for i in range(0, len(trajectories), batch_size):
        batch = trajectories[i:i+batch_size]
        group = TrajectoryGroup(trajectories=batch)
        trajectory_groups.append(group)

    print(f"📊 Created {len(trajectory_groups)} training batches (batch_size={batch_size})")
    print()

    # Train the model on the trajectory groups
    train_config = TrainConfig(
        learning_rate=5e-6,  # Conservative LR for RL fine-tuning
        beta=0.01  # KL penalty coefficient
    )

    print("🚀 Training model on trajectories...")
    await model.train(
        trajectory_groups=trajectory_groups,
        config=train_config,
        verbose=True
    )

    print()
    print("=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Batches: {len(trajectory_groups)}")
    print(f"  Model: {model.name}")
    print(f"  Project: {wandb_project}")
    print()
    print("🎯 Fine-tuned model saved to W&B Artifacts")
    print()


if __name__ == "__main__":
    asyncio.run(main())
