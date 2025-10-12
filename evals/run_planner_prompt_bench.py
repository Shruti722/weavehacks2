"""
Run all Planner prompt variants against all scenarios.
Analyst is fixed; Planner varies by prompt. Results -> data/bench/planner_runs.jsonl
"""

import os, sys, json
from pathlib import Path

# ensure package imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import yaml
from agents.analyst_agent import AnalystAgent
from agents.planner_agent import PlannerAgent

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "data" / "bench" / "scenarios.json"
PROMPTS   = ROOT / "agents" / "prompts" / "planner_prompts.yaml"
OUT       = ROOT / "data" / "bench" / "planner_runs.jsonl"


def to_agent_state(s):
    """Normalize scenario dict for agents."""
    st = dict(s)
    ts = s.get("sim", {}).get("sim_time")
    if ts:
        st["timestamp"] = ts
    # guarantee containers the tools expect
    st.setdefault("kpis", {})
    st.setdefault("drivers", {}).setdefault("price", {}).setdefault("da_usd_per_kwh", 0.20)
    return st


def main():
    print("[run_planner_prompt_bench] Starting benchmark...")
    assert SCENARIOS.exists(), f"Missing {SCENARIOS}"
    assert PROMPTS.exists(), f"Missing {PROMPTS}"

    scenarios = json.loads(SCENARIOS.read_text())
    prompts = yaml.safe_load(PROMPTS.read_text())

    analyst = AnalystAgent()           # fixed prompt
    planner = PlannerAgent()           # we’ll pass system_prompt_override per call

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for s_idx, raw in enumerate(scenarios):
            print(f"\n[Scenario {s_idx+1}/{len(scenarios)}]")
            base_state = to_agent_state(raw)

            # 1) run Analyst once (fixed)
            try:
                analysis = analyst.run(base_state)
            except Exception as e:
                analysis = {"analysis_text": f"[analyst_error] {e}"}

            # compose planner input (state + analyst output)
            planner_input = dict(base_state)
            planner_input["analysis"] = (
                analysis if isinstance(analysis, dict) else {"analysis_text": str(analysis)}
            )

            # 2) loop all planner prompts
            for p in prompts:
                pid = p["id"]
                sys_prompt = p["system"]
                print(f"  → Planner variant: {pid}")

                try:
                    plan = planner.run(
                        planner_input,
                        system_prompt_override=sys_prompt
                    )
                    plan_text = plan.get("plan_text", "")
                    actions = plan.get("actions", [])
                except Exception as e:
                    plan_text, actions = f"[planner_error] {e}", []

                rec = {
                    "scenario_idx": s_idx,
                    "prompt_id": pid,
                    "prompt_label": p.get("label", pid),
                    "timestamp": planner_input.get("timestamp"),
                    "analysis_text": planner_input.get("analysis", {}).get("analysis_text"),
                    "plan_text": plan_text,
                    "actions": actions
                }
                f.write(json.dumps(rec) + "\n")

    print(f"\n✅ Done. Wrote {OUT}")


if __name__ == "__main__":
    main()
