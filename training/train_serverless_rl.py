"""
OpenPipe ServerlessRL (ART) Training Script for SynErgi Grid Multi-Agent System
Exposes gym-style environment for PPO fine-tuning of qwen2.5-14b

Environment factory: make_env()
ART controls episode loops via reset()/step()
Model calls routed through OpenAI-compatible endpoint (OpenPipe)
Dataset: W&B Artifact (weave-traces-ds:latest) for seeding contexts
"""

import os
import sys
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import weave
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.simulator import GridSimulator
from multi_agent_system import SynErgiMultiAgentSystem


def _validate_env_vars():
    """Validate required environment variables - fail fast"""
    required = {
        "OPENPIPE_API_KEY": "Get from https://openpipe.ai",
        "WANDB_API_KEY": "Get from https://wandb.ai/authorize",
        "WEAVE_PROJECT": "Format: username/project-name",
        "WANDB_PROJECT": "Your W&B project name",
    }

    missing = []
    for var, hint in required.items():
        if not os.getenv(var):
            missing.append(f"  ❌ {var} - {hint}")

    if missing:
        raise ValueError(
            "Missing required environment variables:\n" + "\n".join(missing)
        )

    # Validate model name
    base_model = os.getenv("BASE_MODEL", "qwen2.5-14b")
    valid_models = ["qwen2.5-14b", "qwen2.5-7b", "qwen2-72b"]
    if base_model not in valid_models:
        raise ValueError(
            f"Invalid BASE_MODEL='{base_model}'. Must be one of: {valid_models}"
        )

    print("✓ Environment variables validated")
    print(f"  BASE_MODEL: {base_model}")
    print(f"  WANDB_PROJECT: {os.getenv('WANDB_PROJECT')}")
    print(f"  WEAVE_PROJECT: {os.getenv('WEAVE_PROJECT')}")


def _load_dataset_artifact(artifact_ref: str, verify_only: bool = False) -> Optional[List[Dict]]:
    """
    Load W&B Artifact dataset (Weave trace export - JSONL format)

    Args:
        artifact_ref: W&B artifact reference (e.g., weave-traces-ds:latest)
        verify_only: If True, only check existence without loading data

    Returns:
        List of trace rows (prompt, response, metadata), or None if not found
    """
    try:
        # Initialize W&B (needed for artifact access)
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            job_type="artifact-verify" if verify_only else "artifact-download",
            name=f"dataset-{'verify' if verify_only else 'load'}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            reinit=True,
        )

        # Check artifact existence
        artifact = run.use_artifact(artifact_ref, type="dataset")

        if verify_only:
            run.finish()
            print(f"✓ Artifact verified: {artifact_ref}")
            return []

        # Download artifact
        artifact_dir = artifact.download()

        # Load JSONL files (Weave trace exports)
        dataset = []
        for jsonl_file in Path(artifact_dir).glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        # Extract trace fields (prompt, response, metadata)
                        dataset.append(row)

        run.finish()

        print(f"✓ Loaded dataset artifact: {artifact_ref}")
        print(f"  Trace rows: {len(dataset)}")
        if dataset:
            # Show sample structure
            sample_keys = list(dataset[0].keys())[:5]
            print(f"  Sample keys: {sample_keys}")
        return dataset

    except Exception as e:
        if verify_only:
            raise  # Fail fast on verification
        warnings.warn(f"⚠️  Failed to load dataset artifact '{artifact_ref}': {e}")
        warnings.warn("   Will use synthetic data fallback")
        return None


def verify_artifact_exists(artifact_ref: str) -> bool:
    """
    Verify W&B artifact exists before training

    Args:
        artifact_ref: W&B artifact reference

    Returns:
        True if exists, raises error otherwise
    """
    try:
        _load_dataset_artifact(artifact_ref, verify_only=True)
        return True
    except Exception as e:
        raise ValueError(
            f"❌ Required artifact '{artifact_ref}' not found!\n"
            f"   Error: {e}\n"
            f"   Please create the artifact by exporting Weave traces first."
        )


class GridRLEnvironment:
    """
    RL Environment for SynErgi Grid Multi-Agent System
    Compatible with OpenPipe ServerlessRL (ART)

    ART controls training loop - this just exposes reset()/step()
    Supports dataset seeding from W&B Artifact
    """

    def __init__(
        self,
        base_model: str = "qwen2.5-14b",
        dataset: Optional[List[Dict]] = None,
        horizon: int = 12,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        """
        Initialize environment

        Args:
            base_model: Base model name (qwen2.5-14b, etc.)
            dataset: Pre-loaded dataset rows for seeding contexts
            horizon: Episode length in steps (default: 12 = 1 hour at 5-min intervals)
            temperature: Sampling temperature for policy (ART can override)
            top_p: Nucleus sampling parameter (ART can override)
        """
        # Model routing through OpenPipe (OpenAI-compatible)
        openpipe_base_url = "https://api.openpipe.ai/v1"
        model_map = {
            "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
            "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2-72b": "Qwen/Qwen2-72B-Instruct",
        }

        self.base_model = base_model
        self.model_name = model_map.get(base_model, f"Qwen/{base_model}-Instruct")
        self.horizon = horizon
        self.dataset = dataset
        self.temperature = temperature
        self.top_p = top_p

        # Initialize components
        self.simulator = GridSimulator("data/data.json")
        self.mas = SynErgiMultiAgentSystem(
            model_name=self.model_name,
            max_turns=10
        )

        # Set OpenAI API routing for OpenPipe
        os.environ["OPENAI_API_BASE"] = openpipe_base_url
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENPIPE_API_KEY")

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.current_state = None

        print(f"✓ GridRLEnvironment initialized")
        print(f"  Model: {self.model_name}")
        print(f"  Horizon: {self.horizon} steps")
        print(f"  Dataset: {len(dataset) if dataset else 0} rows")
        print(f"  Temperature: {temperature}, Top-p: {top_p}")
        print(f"  OpenPipe endpoint: {openpipe_base_url}")

    @weave.op()
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment for new episode
        Called by ART at episode start

        Seeds initial task/context from Weave trace dataset if available

        Returns:
            observation: Initial grid state observation
        """
        self.step_count = 0
        self.episode_count += 1

        # Seed from Weave trace dataset if available
        if self.dataset and len(self.dataset) > 0:
            # Sample random trace row
            trace_row = random.choice(self.dataset)

            # Extract trace fields (prompt, response, metadata, grid_state)
            # Weave exports can have: inputs, output, attributes, summary, etc.
            seed_context = self._extract_trace_context(trace_row)

            if seed_context.get("grid_state"):
                # Use grid state from trace
                self.current_state = seed_context["grid_state"]
                print(f"[Episode {self.episode_count}] Seeded from trace (grid_state)")
            elif seed_context.get("prompt"):
                # Use prompt to seed task, generate grid state
                self.simulator.reset(start_time=datetime.now())
                self.current_state = self.simulator.generate_tick()
                # Store prompt for multi-agent system context
                self.current_state["_trace_prompt"] = seed_context["prompt"]
                print(f"[Episode {self.episode_count}] Seeded from trace (prompt)")
            else:
                # Fallback: use synthetic
                self.simulator.reset(start_time=datetime.now())
                self.current_state = self.simulator.generate_tick()
                print(f"[Episode {self.episode_count}] Trace incomplete, using synthetic")
        else:
            # No dataset: use synthetic
            self.simulator.reset(start_time=datetime.now())
            self.current_state = self.simulator.generate_tick()
            print(f"[Episode {self.episode_count}] Using synthetic data")

        return self._get_observation()

    def _extract_trace_context(self, trace_row: Dict) -> Dict[str, Any]:
        """
        Extract context from Weave trace row

        Handles various Weave export formats:
        - {inputs: {prompt: ...}, output: {...}, attributes: {...}}
        - {grid_state: {...}, prompt: ...}
        - Raw trace data

        Args:
            trace_row: Raw trace row from artifact

        Returns:
            Dict with extracted fields (prompt, grid_state, metadata)
        """
        context = {}

        # Try common Weave export structures
        if "inputs" in trace_row:
            # Standard Weave format
            inputs = trace_row.get("inputs", {})
            context["prompt"] = inputs.get("prompt") or inputs.get("grid_state")
            context["grid_state"] = inputs.get("grid_state")
        elif "grid_state" in trace_row:
            # Direct grid_state field
            context["grid_state"] = trace_row["grid_state"]
            context["prompt"] = trace_row.get("prompt")
        else:
            # Try to extract any prompt-like or state-like fields
            for key in ["query", "input", "user_prompt", "task"]:
                if key in trace_row:
                    context["prompt"] = trace_row[key]
                    break

        # Store metadata
        context["metadata"] = trace_row.get("attributes") or trace_row.get("metadata") or {}

        return context

    @weave.op()
    def step(self, action: Optional[Dict] = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one environment step
        Called by ART for each timestep

        Args:
            action: Ignored - multi-agent system decides autonomously

        Returns:
            observation: Next state observation
            reward: Scalar reward (float) from simulator
            done: Episode termination flag
            info: Additional metrics dict
        """
        self.step_count += 1

        # Run multi-agent system
        result = self.mas.run(self.current_state)

        # Extract reward from actuator
        actuator_output = result.get("actuator_output", {})
        reward_data = actuator_output.get("reward_data", {})

        # Scalar reward - ART expects float
        reward = float(reward_data.get("reward", 0.0) if reward_data else 0.0)

        # Update state
        self.current_state = result.get("grid_state", self.current_state)
        self.current_state = self.simulator.generate_tick()

        # Episode termination
        done = self.step_count >= self.horizon

        # Info dict for Weave logging
        info = {
            "episode": self.episode_count,
            "step": self.step_count,
            "reward": reward,
            "cost_usd": reward_data.get("cost_usd", 0) if reward_data else 0,
            "deficit_improvement_mw": reward_data.get("deficit_improvement_mw", 0) if reward_data else 0,
            "actions_count": len(actuator_output.get("actions_parsed", [])),
            "demand_mw": self.current_state["kpis"]["city_demand_mw"],
            "supply_mw": self.current_state["kpis"]["city_supply_mw"],
            "risk": self.current_state["kpis"]["avg_overload_risk"],
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> Dict[str, Any]:
        """Convert grid state to observation dict"""
        kpis = self.current_state.get("kpis", {})
        drivers = self.current_state.get("drivers", {})

        return {
            "timestamp": self.current_state.get("timestamp"),
            "time_of_day": drivers.get("time_of_day"),
            "demand_mw": kpis.get("city_demand_mw", 0),
            "supply_mw": kpis.get("city_supply_mw", 0),
            "deficit_mw": kpis.get("city_demand_mw", 0) - kpis.get("city_supply_mw", 0),
            "risk": kpis.get("avg_overload_risk", 0),
            "fairness": kpis.get("fairness_index", 0),
            "step": self.step_count,
        }


def make_env():
    """
    Environment factory for OpenPipe ServerlessRL (ART)

    ART calls this to create environment instances.
    All config read from environment variables.

    Returns:
        GridRLEnvironment: Configured environment instance
    """
    # Validate environment first
    _validate_env_vars()

    # Initialize Weave for tracing (NOT a new W&B run per episode)
    weave_project = os.getenv("WEAVE_PROJECT")
    weave.init(weave_project)
    print(f"✓ Weave initialized: {weave_project}")

    # Read config from env
    base_model = os.getenv("BASE_MODEL", "qwen2.5-14b")
    dataset_ref = os.getenv("DATASET_REF", "weave-traces-ds:latest")
    horizon = int(os.getenv("HORIZON", "12"))
    temperature = float(os.getenv("TEMPERATURE", "0.3"))
    top_p = float(os.getenv("TOP_P", "0.9"))
    verify_artifact = os.getenv("VERIFY_ARTIFACT", "true").lower() == "true"

    # Verify artifact exists before training (fail early)
    if verify_artifact and dataset_ref:
        print(f"\n🔍 Verifying artifact: {dataset_ref}")
        try:
            verify_artifact_exists(dataset_ref)
        except ValueError as e:
            print(str(e))
            print("\n⚠️  To skip verification, set VERIFY_ARTIFACT=false")
            raise

    # Load dataset artifact if specified
    dataset = None
    if dataset_ref:
        print(f"\n📦 Loading artifact: {dataset_ref}")
        dataset = _load_dataset_artifact(dataset_ref)

        if dataset is None and verify_artifact:
            raise ValueError(
                f"❌ Failed to load required artifact '{dataset_ref}'\n"
                f"   Set VERIFY_ARTIFACT=false to continue with synthetic data only"
            )

    # Create environment
    env = GridRLEnvironment(
        base_model=base_model,
        dataset=dataset,
        horizon=horizon,
        temperature=temperature,
        top_p=top_p,
    )

    print(f"\n✓ Environment factory complete")
    print(f"  LoRA checkpoints → W&B Artifacts (project: {os.getenv('WANDB_PROJECT')})")
    print(f"  Weave traces → {weave_project}")
    print(f"  Ready for ART training!\n")
    return env


if __name__ == "__main__":
    """
    Test environment locally (not for ART training)

    For actual training, use OpenPipe CLI:
    $ openpipe art train \\
        --env training.train_serverless_rl:make_env \\
        --model qwen2.5-14b \\
        --episodes 300
    """
    print("=" * 60)
    print("Testing GridRLEnvironment locally")
    print("=" * 60)

    env = make_env()

    # Run single episode
    obs = env.reset()
    print(f"\n[Test] Initial obs: demand={obs['demand_mw']:.1f} MW")

    done = False
    total_reward = 0

    while not done:
        obs, reward, done, info = env.step()
        total_reward += reward
        print(f"  Step {info['step']}: reward={reward:.3f}, demand={obs['demand_mw']:.1f} MW")

    print(f"\n✓ Test episode complete. Total reward: {total_reward:.3f}")
    print("=" * 60)
