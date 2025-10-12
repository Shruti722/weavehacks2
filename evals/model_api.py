# weavehacks2/eval/model_api.py
import json
from pathlib import Path
from typing import Dict, Any, Callable

from weavehacks2.agents.analyst_agent import AnalystAgent
from weavehacks2.agents.planner_agent import PlannerAgent

# Keep singletons to avoid repeated model init
_ANALYST = AnalystAgent()
_PLANNER = PlannerAgent()

def _scenario_to_state(scn: Dict[str, Any]) -> Dict[str, Any]:
    """Map your scenario JSON into the state shape agents expect."""
    ts = scn.get("sim", {}).get("sim_time") or scn.get("timestamp")
    return {
        "timestamp": ts,
        "drivers": scn.get("drivers", {}),
        "nodes": scn.get("nodes", {}),
        "kpis": scn.get("kpis", {}),
    }

def _with_planner_user_suffix(agent: PlannerAgent, user_suffix: str) -> Callable[[Dict[str, Any]], str]:
    """
    Monkey-patch wrapper for PlannerAgent._build_planning_prompt to append the prompt text.
    Does NOT modify the class globally; only the instance for this call.
    """
    orig = agent._build_planning_prompt

    def wrapped(state: Dict[str, Any]) -> str:
        base = orig(state)
        suffix = ("\n\n" + user_suffix.strip()) if user_suffix and user_suffix.strip() else ""
        return base + suffix

    return wrapped

def call_model(prompt_text: str, scenario: Dict[str, Any]) -> str:
    """
    Adapter entrypoint used by the evaluator.
    - Builds state from scenario
    - Runs AnalystAgent then PlannerAgent
    - Injects `prompt_text` as a suffix into the Planner's user prompt (Option 2)
    - Returns a normalized JSON string: {"actions": [...], "plan_text": "...", "agent": "...", "timestamp": "..."}
    """
    # Build base state
    base_state = _scenario_to_state(scenario)

    # 1) Analyst
    analysis = _ANALYST.run(base_state)

    # 2) Planner with prompt injection (suffix)
    #    We wrap only for this invocation to avoid side effects.
    original_fn = _PLANNER._build_planning_prompt
    try:
        _PLANNER._build_planning_prompt = _with_planner_user_suffix(_PLANNER, prompt_text)
        plan_input = {**base_state, "analysis": analysis}
        plan = _PLANNER.run(plan_input)
    finally:
        _PLANNER._build_planning_prompt = original_fn  # restore

    # 3) Normalize planner output to a simple envelope for scoring
    out = {
        "actions": plan.get("actions", []),           # [{node_id, adjustments{...}}, ...]
        "plan_text": plan.get("plan_text", ""),
        "agent": plan.get("agent"),
        "timestamp": plan.get("timestamp") or base_state.get("timestamp"),
    }
    return json.dumps(out)
