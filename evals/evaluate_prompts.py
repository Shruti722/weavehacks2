# weavehacks2/eval/evaluate_prompts.py
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from critique import critique
from model_api import call_model
from objective import objective

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = ROOT / "data" / "bench" / "scenarios.json"
PROMPTS_DIR = ROOT / "eval" / "prompts"
ACTIVE_OUT = ROOT / "eval" / "active_prompt.txt"

def load_scenarios() -> List[Dict[str, Any]]:
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(f"Missing scenarios file at {SCENARIOS_PATH}")
    return json.loads(SCENARIOS_PATH.read_text())

def evaluate_prompt(prompt_path: Path, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_text = prompt_path.read_text()
    Js, accepted = [], 0
    for scn in scenarios:
        raw = call_model(prompt_text, scn)
        gate = critique(raw)
        if gate["verdict"] != "ACCEPT":
            Js.append(float("inf"))   # penalize invalid output
            continue
        plan = json.loads(raw)
        J = objective(plan, scn)
        Js.append(J)
        accepted += 1
    mean_J = float(np.mean(Js)) if Js else float("inf")
    return {
        "prompt": prompt_path.name,
        "accept_rate": accepted / len(scenarios) if scenarios else 0.0,
        "mean_J": mean_J,
    }

if __name__ == "__main__":
    scenarios = load_scenarios()
    prompt_files = sorted(PROMPTS_DIR.glob("*.txt"))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files found in {PROMPTS_DIR}")

    results = [evaluate_prompt(p, scenarios) for p in prompt_files]

    # Optional accept-rate gate (kept simple for baseline)
    survivors = [r for r in results if r["accept_rate"] >= 0.8] or results
    survivors.sort(key=lambda r: r["mean_J"])

    print("\n=== Leaderboard (lower mean_J is better) ===")
    for r in survivors:
        print(f"{r['prompt']:28}  mean_J={r['mean_J']:.4f}  accept_rate={r['accept_rate']:.2f}")

    champion = survivors[0]["prompt"]
    ACTIVE_OUT.write_text((PROMPTS_DIR / champion).read_text())
    print(f"\nChampion -> {champion}")
    print(f"Saved to {ACTIVE_OUT}")
