# weavehacks2/eval/objective.py
from typing import Dict, Any, List
import json

from weavehacks2.agents.tools.planner_tools import simulate_plan, cost_risk_analysis
from weavehacks2.utils.reward import compute_reward

def _nodes_dict_to_reward_nodes(nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    reward.compute_reward expects state['nodes'] as a LIST with 'risk_index' per node.
    Your scenarios use a dict and store risk at nodes[*]['risk']['overload'].
    """
    out = []
    for node_id, node in nodes.items():
        risk_overload = 0.0
        if isinstance(node.get("risk"), dict):
            risk_overload = float(node["risk"].get("overload", 0.0))
        out.append({
            "id": node_id,
            "risk_index": risk_overload
        })
    return out

def _normalize_actions_flat(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert {'adjustments': {'discharge_storage': 5.0, ...}} into a flat list of
    {'node_id': ..., 'action_type': 'discharge_storage', 'target_mw': 5.0} items.
    Useful if you later want stricter critique or tool checks.
    """
    flat = []
    for a in actions:
        node_id = a.get("node_id")
        adj = a.get("adjustments", {}) or {}
        for act, val in adj.items():
            flat.append({"node_id": node_id, "action_type": act, "target_mw": float(val)})
    return flat

def objective(plan: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Returns scalar J (lower is better).
    Steps:
      1) Simulate plan on the scenario
      2) Estimate projected cost/risk with cost_risk_analysis
      3) Build a reward-compatible state and compute reward
      4) J = - total_reward
    """
    # Build current_state (agents and tools expect dicts with kpis/drivers/nodes)
    current_state = {
        "timestamp": scenario.get("sim", {}).get("sim_time") or scenario.get("timestamp"),
        "drivers": scenario.get("drivers", {}),
        "nodes": scenario.get("nodes", {}),
        "kpis": scenario.get("kpis", {}),
        # these fields can be absent; defaults are fine for analysis functions
        "total_cost": scenario.get("total_cost", 0),
        "fairness_index": scenario.get("kpis", {}).get("fairness_index", 0),
    }

    planned_actions = plan.get("actions", [])

    # 1) Simulate plan impact
    sim_result = simulate_plan(current_state, planned_actions)
    simulated_state = sim_result["simulated_state"]

    # 2) Cost/risk analysis to get a projected cost number
    cra = cost_risk_analysis(current_state, simulated_state, planned_actions)
    projected_cost_usd = cra.get("cost_analysis", {}).get("projected_cost_usd", 0.0)

    # 3) Build reward-state for compute_reward
    reward_state = {
        # compute_reward expects:
        # - total_cost (normalized inside)
        # - fairness_index
        # - nodes: LIST with 'risk_index' per node
        "total_cost": projected_cost_usd,
        "fairness_index": simulated_state.get("kpis", {}).get("fairness_index", current_state["fairness_index"]),
        "nodes": _nodes_dict_to_reward_nodes(simulated_state.get("nodes", {})),
    }

    reward = compute_reward(reward_state)  # dict with 'total_reward'
    total_reward = float(reward.get("total_reward", 0.0))

    # 4) Our evaluation metric to MINIMIZE:
    J = -total_reward
    return J
