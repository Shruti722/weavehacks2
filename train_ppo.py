import os, json, random
from pathlib import Path
from typing import Dict, Any, List

import wandb
import weave
import yaml

# ---- Weights & Biases Weave client for Qwen ----
# Using weave.Model for inference instead of OpenAI client

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Expand environment variables
    for key in ["policy_model", "ref_model"]:
        if key in cfg and isinstance(cfg[key], str) and cfg[key].startswith("${") and cfg[key].endswith("}"):
            env_var = cfg[key][2:-1]
            cfg[key] = os.environ.get(env_var, cfg[key])
    if "logging" in cfg and "wandb_project" in cfg["logging"]:
        if cfg["logging"]["wandb_project"].startswith("${") and cfg["logging"]["wandb_project"].endswith("}"):
            env_var = cfg["logging"]["wandb_project"][2:-1]
            cfg["logging"]["wandb_project"] = os.environ.get(env_var, cfg["logging"]["wandb_project"])
    return cfg

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines

def load_weave_traces(traces_path: str) -> List[Dict[str, Any]]:
    """Load and convert Weave traces into training examples"""
    traces = load_jsonl(traces_path)
    training_examples = []

    for trace in traces:
        # Only use traces with both prompt and completion (output)
        if "prompt" in trace and "completion" in trace and trace.get("output"):
            output_data = trace["output"]

            # Skip traces without reward data
            if "reward" not in output_data:
                continue

            # Create training example
            example = {
                "prompt": trace["prompt"],
                "completion": trace["completion"],
                "reward": output_data["reward"],
                "metadata": {
                    "episode_num": output_data.get("episode_num"),
                    "raw_score": output_data.get("raw_score"),
                    "deficit_improvement_mw": output_data.get("deficit_improvement_mw"),
                    "cost_usd": output_data.get("cost_usd"),
                    "timestamp": trace.get("timestamp")
                }
            }
            training_examples.append(example)

    return training_examples

# Global model cache to avoid reloading on every call
_model_cache = {}

@weave.op()
def gen_policy(prompt: str, model_name: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Generate response using Weave-tracked model inference"""
    global _model_cache

    try:
        # Check if model is already loaded
        if model_name not in _model_cache:
            print(f"Loading model {model_name}...")

            # Option 1: Load from local artifacts directory
            local_model_path = f"./artifacts/models/{model_name}"
            if os.path.exists(local_model_path):
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                model = AutoModelForCausalLM.from_pretrained(local_model_path)
                _model_cache[model_name] = {"model": model, "tokenizer": tokenizer}
                print(f"✓ Loaded model from {local_model_path}")
            else:
                # Option 2: Load from HuggingFace (e.g., "Qwen/Qwen2.5-14B-Instruct")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                hf_model_name = f"Qwen/Qwen2.5-14B-Instruct" if "qwen2.5-14b" in model_name.lower() else model_name
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                model = AutoModelForCausalLM.from_pretrained(hf_model_name)
                _model_cache[model_name] = {"model": model, "tokenizer": tokenizer}
                print(f"✓ Loaded model from HuggingFace: {hf_model_name}")

        # Get cached model
        model = _model_cache[model_name]["model"]
        tokenizer = _model_cache[model_name]["tokenizer"]

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    except Exception as e:
        print(f"Error in gen_policy: {e}")
        # Fallback - simple placeholder
        return f"[Error generating response: {str(e)}]"

@weave.op()
def reward_reference_similarity(output: str, reference: str) -> float:
    """Simple baseline reward (Jaccard/overlap)"""
    out_words = set(output.lower().split())
    ref_words = set(reference.lower().split())
    if not out_words or not ref_words:
        return 0.0
    jaccard = len(out_words & ref_words) / len(out_words | ref_words)
    return float(jaccard * 10.0)  # scale to ~[0,10]

def main():
    cfg = load_config("ppo.yaml")
    data_cfg = cfg["data"]
    roll_cfg = cfg["rollout"]
    train_cfg = cfg["train"]
    rew_cfg = cfg["reward"]
    log_cfg = cfg["logging"]

    # Init tracking
    wandb.init(project=log_cfg["wandb_project"], config=cfg)
    weave.init(os.environ["WEAVE_PROJECT"])

    # Load training data from Weave traces
    traces_path = "artifacts/weave-traces/traces.jsonl"
    if os.path.exists(traces_path):
        print(f"Loading training data from {traces_path}")
        dataset = load_weave_traces(traces_path)
    else:
        # Fallback to original JSONL data
        print(f"Weave traces not found, loading from {data_cfg['path']}")
        dataset = load_jsonl(data_cfg["path"])

    random.shuffle(dataset)

    print(f"Loaded {len(dataset)} training examples")
    print(f"Policy model: {cfg['policy_model']}")
    print(f"Reference model: {cfg['ref_model']}")

    # ---- PPO Training Loop ----
    # Note: This is a simplified loop. For full PPO, you'd need:
    # 1. Advantage calculation (GAE)
    # 2. Value function training
    # 3. Policy gradient with clipping
    # 4. KL penalty vs reference model

    total_updates = train_cfg["total_updates"]
    bs = roll_cfg["batch_size"]

    for step in range(total_updates):
        # Sample batch
        start_idx = (step * bs) % len(dataset)
        end_idx = start_idx + bs
        if end_idx <= len(dataset):
            batch = dataset[start_idx:end_idx]
        else:
            # Wrap around
            batch = dataset[start_idx:] + dataset[:end_idx - len(dataset)]

        # Handle both Weave trace format and original JSONL format
        if "prompt" in batch[0]:
            # Weave trace format
            prompts = [ex["prompt"] for ex in batch]
            refs = [ex.get("completion", "") for ex in batch]
        else:
            # Original JSONL format
            prompts = [ex[data_cfg["prompt_key"]] for ex in batch]
            refs = [ex.get(data_cfg["reference_key"], "") for ex in batch]

        # Rollout - generate responses
        outputs = [
            gen_policy(
                p,
                cfg["policy_model"],
                roll_cfg["max_gen_tokens"],
                roll_cfg["temperature"],
                roll_cfg["top_p"]
            )
            for p in prompts
        ]

        # Calculate rewards
        if rew_cfg["type"] == "heuristic":
            rewards = [reward_reference_similarity(o, r) for o, r in zip(outputs, refs)]
        elif rew_cfg["type"] == "http":
            import requests
            rewards = []
            for p, o, r in zip(prompts, outputs, refs):
                try:
                    res = requests.post(
                        rew_cfg["url"],
                        json={"prompt": p, "output": o, "reference": r},
                        timeout=rew_cfg.get("timeout_s", 30)
                    )
                    rewards.append(float(res.json().get("reward", 0.0)))
                except Exception as e:
                    print(f"Reward API error: {e}")
                    rewards.append(0.0)
        else:
            rewards = [0.0] * len(outputs)

        # PPO update would go here
        # For now, just logging the rewards
        # In full implementation, you'd:
        # 1. Compute advantages using GAE
        # 2. Update policy with clipped objective
        # 3. Update value function
        # 4. Apply KL penalty

        # Log metrics
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        wandb.log({
            "step": step,
            "reward/mean": mean_reward,
            "reward/max": max(rewards) if rewards else 0.0,
            "reward/min": min(rewards) if rewards else 0.0,
        })

        if (step + 1) % log_cfg.get("log_every", 10) == 0:
            print(f"[step {step+1}/{total_updates}] mean reward = {mean_reward:.3f}")

        if (step + 1) % log_cfg.get("eval_every", 100) == 0:
            print(f"[eval] step {step+1}: mean reward = {mean_reward:.3f}")

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
