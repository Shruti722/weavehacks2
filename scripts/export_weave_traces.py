#!/usr/bin/env python3
"""
Export Weave traces to JSONL format for supervised learning
Fetches traces from: ruchib-northwestern-university/synergi-rl-training
Creates W&B Artifact: weave-traces-ds:latest
"""

import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import weave
import wandb

load_dotenv()


def export_weave_traces_to_jsonl():
    """Export Weave traces from the project to JSONL format"""

    # Initialize Weave
    weave_project = os.getenv("WEAVE_PROJECT", "ruchib-northwestern-university/synergi-rl-training")
    wandb_project = os.getenv("WANDB_PROJECT", "synergi-grid-optimization")

    print("=" * 70)
    print("Exporting Weave Traces to JSONL")
    print("=" * 70)
    print(f"  Weave Project: {weave_project}")
    print(f"  W&B Project: {wandb_project}")
    print()

    # Check existing traces
    output_file = Path("artifacts/weave-traces") / "traces.jsonl"
    existing_ids = set()
    if output_file.exists():
        print("📂 Loading existing traces...")
        with open(output_file) as f:
            for line in f:
                trace = json.loads(line)
                existing_ids.add(trace["id"])
        print(f"  Found {len(existing_ids)} existing traces")
        print()

    # Initialize Weave client
    print("🔗 Connecting to Weave...")
    client = weave.init(weave_project)

    # Fetch traces - targeting collect_single_episode calls
    print("📦 Fetching ALL traces...")

    # Get ALL calls (no limit - fetch all 2000)
    calls_list = []

    try:
        # Fetch calls
        print("  Fetching all collect_single_episode traces...")
        calls = client.get_calls()

        for call in calls:
            if "collect_single_episode" in (call.op_name or ""):
                # Skip if already processed
                if call.id not in existing_ids:
                    calls_list.append(call)
                if len(calls_list) % 500 == 0:
                    print(f"    Progress: {len(calls_list)} new traces fetched...")

        print(f"  Finished fetching")

    except Exception as e:
        print(f"  Error fetching calls: {e}")
        return None

    print(f"✓ Fetched {len(calls_list)} total traces")

    if len(calls_list) == 0:
        print("\n❌ No traces found matching 'collect_single_episode'")
        return None

    # Convert to JSONL format
    traces = []
    for call in calls_list:
        try:
            # Extract call data
            trace = {
                "id": call.id,
                "op_name": call.op_name,
                "timestamp": call.started_at.isoformat() if hasattr(call, 'started_at') and call.started_at else None,
                "inputs": call.inputs or {},
                "output": call.output or {},
                "attributes": {
                    "trace_id": call.trace_id if hasattr(call, 'trace_id') else None,
                    "parent_id": call.parent_id if hasattr(call, 'parent_id') else None,
                },
            }

            # For supervised learning: extract grid_state if available
            if isinstance(call.inputs, dict) and "grid_state" in call.inputs:
                trace["grid_state"] = call.inputs["grid_state"]

            # Extract prompt/completion for SFT
            if isinstance(call.inputs, dict):
                trace["prompt"] = _format_prompt(call.inputs)

            if call.output:
                trace["completion"] = _format_completion(call.output)

            traces.append(trace)

        except Exception as e:
            print(f"  ⚠️  Error processing call {call.id}: {e}")
            continue

    print(f"✓ Processed {len(traces)} traces")

    if len(traces) == 0:
        print("\n❌ No traces could be exported!")
        return None

    # Filter top 50% by reward
    print(f"\n🔝 Filtering top 50% by reward...")

    # Extract rewards
    traces_with_rewards = []
    for trace in traces:
        reward = _extract_reward(trace["output"])
        traces_with_rewards.append((trace, reward))

    # Sort by reward (descending)
    traces_with_rewards.sort(key=lambda x: x[1], reverse=True)

    # Take top 50%
    top_50_count = max(1, len(traces_with_rewards) // 2)
    top_traces = [t[0] for t in traces_with_rewards[:top_50_count]]

    print(f"  Total traces: {len(traces)}")
    print(f"  Top 50%: {len(top_traces)} traces")
    print(f"  Reward range: {traces_with_rewards[0][1]:.3f} to {traces_with_rewards[top_50_count-1][1]:.3f}")

    traces = top_traces  # Use filtered traces

    # Save to JSONL (append mode if file exists)
    output_dir = Path("artifacts/weave-traces")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "a" if output_file.exists() and len(existing_ids) > 0 else "w"
    with open(output_file, mode) as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    print(f"✓ Saved to {output_file}")
    print()

    # Show sample
    if traces:
        print("Sample trace (first 500 chars):")
        sample = json.dumps(traces[0], indent=2)
        print(sample[:500] + "..." if len(sample) > 500 else sample)
        print()

    return output_file, traces


def _format_prompt(inputs):
    """Format inputs as a prompt string"""
    if "grid_state" in inputs:
        grid_state = inputs["grid_state"]
        kpis = grid_state.get("kpis", {})
        return f"""Analyze the power grid state and recommend actions:

Timestamp: {grid_state.get('timestamp', 'N/A')}
Demand: {kpis.get('city_demand_mw', 0):.1f} MW
Supply: {kpis.get('city_supply_mw', 0):.1f} MW
Risk: {kpis.get('avg_overload_risk', 0):.3f}

Provide analysis, plan, and actions."""
    else:
        return json.dumps(inputs, indent=2)


def _format_completion(output):
    """Format output as a completion string"""
    if isinstance(output, dict):
        # Try to extract structured output
        if "analyst_output" in output or "planner_output" in output:
            return json.dumps({
                "analysis": output.get("analyst_output", {}),
                "plan": output.get("planner_output", {}),
                "actions": output.get("actuator_output", {})
            }, indent=2)
        else:
            return json.dumps(output, indent=2)
    else:
        return str(output)


def _extract_reward(output):
    """Extract reward value from output for ranking"""
    if isinstance(output, dict):
        # Try various reward locations
        if "reward" in output:
            return float(output["reward"])
        if "actuator_output" in output:
            actuator = output["actuator_output"]
            if isinstance(actuator, dict):
                if "reward_data" in actuator:
                    reward_data = actuator["reward_data"]
                    if isinstance(reward_data, dict) and "reward" in reward_data:
                        return float(reward_data["reward"])
                if "reward" in actuator:
                    return float(actuator["reward"])
    return 0.0  # Default reward if not found


def create_wandb_artifact(jsonl_file, traces):
    """Create W&B Artifact from JSONL file"""

    wandb_project = os.getenv("WANDB_PROJECT", "synergi-grid-optimization")

    print("📤 Creating W&B Artifact...")

    # Initialize W&B
    run = wandb.init(
        project=wandb_project,
        job_type="create-dataset",
        name=f"create-weave-traces-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    # Create artifact
    artifact = wandb.Artifact(
        name="weave-traces-ds",
        type="dataset",
        description="Weave trace exports for grid optimization (supervised learning format)",
        metadata={
            "num_traces": len(traces),
            "format": "jsonl",
            "source": "weave_export",
            "exported_at": datetime.now().isoformat(),
        }
    )

    # Add JSONL file
    artifact.add_file(str(jsonl_file))

    # Log artifact
    run.log_artifact(artifact)
    run.finish()

    print(f"✓ Artifact created: weave-traces-ds:latest")
    print(f"  Project: {wandb_project}")
    print(f"  Traces: {len(traces)}")
    print()


def main():
    """Main export function"""

    print("\n🔍 Target: collect_single_episode traces")
    print("   URL: https://wandb.ai/ruchib-northwestern-university/synergi-rl-training/weave/traces")
    print()

    # Export traces
    result = export_weave_traces_to_jsonl()

    if result is None:
        print("\n❌ Export failed - no traces available")
        return

    jsonl_file, traces = result

    # Create W&B artifact
    create_wandb_artifact(jsonl_file, traces)

    print("=" * 70)
    print("✅ Export Complete!")
    print("=" * 70)
    print()
    print("Artifact ready for training:")
    print(f"  📁 Local: {jsonl_file}")
    print(f"  ☁️  W&B: weave-traces-ds:latest")
    print()
    print("Next steps:")
    print("  1. Verify artifact exists:")
    print("     .venv/bin/python -c \"import wandb; print(wandb.Api().artifact('ruchib-northwestern-university/synergi-grid-optimization/weave-traces-ds:latest'))\"")
    print()
    print("  2. Enable in .env:")
    print("     VERIFY_ARTIFACT=true")
    print()
    print("  3. Run training:")
    print("     .venv/bin/python scripts/run_art_training.py")
    print()


if __name__ == "__main__":
    main()
