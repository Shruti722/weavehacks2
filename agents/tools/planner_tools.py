"""
PlannerAgent Tools
Tools for simulating plans, analyzing cost/risk, and retrieving past policies
"""

import json
import copy
from typing import Dict, List
from datetime import datetime


# Mock policy database (in production, this would be a real database)
POLICY_VAULT_DB = []


def simulate_plan(current_state: Dict, planned_actions: List[Dict]) -> Dict:
    """
    Simulate what will happen when ActuatorAgent executes the plan.
    Projects the future grid state after actions are applied.

    Args:
        current_state: Current grid state
        planned_actions: List of planned actions (from planner)

    Returns:
        Dict with simulated future state
    """
    # Deep copy current state to simulate
    simulated_state = copy.deepcopy(current_state)

    # Track what changes
    changes = []

    for action in planned_actions:
        node_id = action.get("node_id")
        adjustments = action.get("adjustments", {})

        # Find the node in simulated state
        nodes = simulated_state.get("nodes", {})
        if node_id not in nodes:
            continue

        node = nodes[node_id]

        # Apply each adjustment
        for adj_type, value in adjustments.items():
            if "increase_supply" in adj_type:
                old_supply = node.get("supply_mw", 0)
                new_supply = old_supply + value
                node["supply_mw"] = new_supply
                changes.append({
                    "node_id": node_id,
                    "action": "increase_supply",
                    "change_mw": value,
                    "old_value": old_supply,
                    "new_value": new_supply
                })

            elif "reduce_supply" in adj_type:
                old_supply = node.get("supply_mw", 0)
                new_supply = max(0, old_supply - value)
                node["supply_mw"] = new_supply
                changes.append({
                    "node_id": node_id,
                    "action": "reduce_supply",
                    "change_mw": -value,
                    "old_value": old_supply,
                    "new_value": new_supply
                })

            elif "discharge_storage" in adj_type:
                # Discharging adds to supply
                storage = node.get("storage", {})
                old_soc = storage.get("soc", 0)
                capacity_mwh = storage.get("energy_mwh_cap", 50)

                # Convert MW to energy (5 min = 1/12 hour)
                energy_mwh = value / 12
                new_soc = max(0, old_soc - (energy_mwh / capacity_mwh))
                storage["soc"] = new_soc

                # Add to supply
                node["supply_mw"] = node.get("supply_mw", 0) + value

                changes.append({
                    "node_id": node_id,
                    "action": "discharge_storage",
                    "change_mw": value,
                    "old_soc": old_soc,
                    "new_soc": new_soc
                })

            elif "charge_storage" in adj_type:
                # Charging uses surplus supply
                storage = node.get("storage", {})
                old_soc = storage.get("soc", 0)
                capacity_mwh = storage.get("energy_mwh_cap", 50)

                energy_mwh = value / 12
                new_soc = min(1.0, old_soc + (energy_mwh / capacity_mwh))
                storage["soc"] = new_soc

                changes.append({
                    "node_id": node_id,
                    "action": "charge_storage",
                    "change_mw": value,
                    "old_soc": old_soc,
                    "new_soc": new_soc
                })

        # Recalculate risk after changes
        demand = node.get("demand_mw", 0)
        supply = node.get("supply_mw", 0)
        imbalance_ratio = abs(supply - demand) / max(demand, 1.0)
        node["risk"]["overload"] = min(1.0, imbalance_ratio * 0.6)

    # Recalculate city-wide KPIs
    nodes = simulated_state.get("nodes", {})
    total_demand = sum(n.get("demand_mw", 0) for n in nodes.values())
    total_supply = sum(n.get("supply_mw", 0) for n in nodes.values())

    simulated_state["kpis"]["city_demand_mw"] = total_demand
    simulated_state["kpis"]["city_supply_mw"] = total_supply
    simulated_state["kpis"]["unserved_energy_proxy_mw"] = max(0, total_demand - total_supply)

    return {
        "simulation_timestamp": simulated_state.get("timestamp"),
        "simulated_state": simulated_state,
        "changes_applied": changes,
        "summary": {
            "total_changes": len(changes),
            "nodes_affected": len(set(c["node_id"] for c in changes)),
            "projected_unserved_energy": simulated_state["kpis"]["unserved_energy_proxy_mw"]
        }
    }


def cost_risk_analysis(current_state: Dict, simulated_state: Dict, planned_actions: List[Dict]) -> Dict:
    """
    Analyze the cost and risk of executing the plan.

    Args:
        current_state: Current grid state
        simulated_state: Simulated future state after plan execution
        planned_actions: List of planned actions

    Returns:
        Dict with cost and risk breakdown
    """
    # Extract KPIs
    current_kpis = current_state.get("kpis", {})
    simulated_kpis = simulated_state.get("kpis", {})

    current_cost = current_state.get("total_cost", 0)
    current_risk = current_kpis.get("avg_overload_risk", 0)
    current_fairness = current_kpis.get("fairness_index", 0)

    # Calculate simulated cost (approximate)
    price_per_kwh = current_state.get("drivers", {}).get("price", {}).get("da_usd_per_kwh", 0.25)
    simulated_supply = simulated_kpis.get("city_supply_mw", 0)
    simulated_cost = simulated_supply * price_per_kwh * (5/60)  # 5-minute interval

    # Calculate simulated risk
    simulated_nodes = simulated_state.get("nodes", {})
    if simulated_nodes:
        simulated_risk = sum(n.get("risk", {}).get("overload", 0) for n in simulated_nodes.values()) / len(simulated_nodes)
    else:
        simulated_risk = current_risk

    simulated_fairness = simulated_kpis.get("fairness_index", current_fairness)

    # Calculate deltas
    cost_delta = simulated_cost - current_cost
    risk_delta = simulated_risk - current_risk
    fairness_delta = simulated_fairness - current_fairness

    # Action-specific costs
    action_costs = []
    for action in planned_actions:
        adjustments = action.get("adjustments", {})
        node_cost = 0

        for adj_type, value in adjustments.items():
            if "increase_supply" in adj_type:
                # Gas turbine cost ~$80/MWh
                node_cost += value * 80 * (5/60)
            elif "discharge_storage" in adj_type:
                # Battery discharge - minimal cost
                node_cost += value * 5 * (5/60)

        action_costs.append({
            "node_id": action.get("node_id"),
            "estimated_cost_usd": round(node_cost, 2)
        })

    total_action_cost = sum(ac["estimated_cost_usd"] for ac in action_costs)

    # Risk assessment
    risk_assessment = {
        "current_risk": round(current_risk, 3),
        "projected_risk": round(simulated_risk, 3),
        "risk_change": round(risk_delta, 3),
        "risk_improvement": "Yes" if risk_delta < 0 else "No",
        "high_risk_nodes_count": sum(1 for n in simulated_nodes.values() if n.get("risk", {}).get("overload", 0) > 0.4)
    }

    return {
        "analysis_timestamp": current_state.get("timestamp"),
        "cost_analysis": {
            "current_cost_usd": round(current_cost, 2),
            "projected_cost_usd": round(simulated_cost, 2),
            "cost_delta_usd": round(cost_delta, 2),
            "action_costs": action_costs,
            "total_action_cost_usd": round(total_action_cost, 2)
        },
        "risk_analysis": risk_assessment,
        "fairness_analysis": {
            "current_fairness": round(current_fairness, 3),
            "projected_fairness": round(simulated_fairness, 3),
            "fairness_delta": round(fairness_delta, 3),
            "fairness_improvement": "Yes" if fairness_delta > 0 else "No"
        },
        "recommendation": "Proceed" if risk_delta < 0 and cost_delta < 1000 else "Revise plan",
        "recommendation_reason": f"Risk {'decreases' if risk_delta < 0 else 'increases'} by {abs(risk_delta):.3f}, Cost {'increases' if cost_delta > 0 else 'decreases'} by ${abs(cost_delta):.2f}"
    }


def policy_vault(query_type: str = "similar", context: Dict = None, limit: int = 5) -> Dict:
    """
    Retrieve past successful plans from the policy vault.
    Finds old plans that worked in similar situations.

    Args:
        query_type: Type of query ("similar", "best", "recent")
        context: Current grid context for similarity matching
        limit: Maximum number of policies to return

    Returns:
        Dict with retrieved policies
    """
    # In production, this would query a vector database or policy storage
    # For now, use in-memory mock database

    if query_type == "similar" and context:
        # Find policies with similar grid conditions
        current_demand = context.get("kpis", {}).get("city_demand_mw", 0)
        current_risk = context.get("kpis", {}).get("avg_overload_risk", 0)

        # Filter similar policies
        similar_policies = []
        for policy in POLICY_VAULT_DB:
            policy_demand = policy.get("context", {}).get("city_demand_mw", 0)
            policy_risk = policy.get("context", {}).get("avg_overload_risk", 0)

            # Simple similarity score
            demand_diff = abs(policy_demand - current_demand) / max(current_demand, 1)
            risk_diff = abs(policy_risk - current_risk)

            similarity_score = 1.0 - (demand_diff * 0.5 + risk_diff * 0.5)

            if similarity_score > 0.7:  # Threshold
                similar_policies.append({
                    **policy,
                    "similarity_score": round(similarity_score, 3)
                })

        # Sort by similarity
        similar_policies.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = similar_policies[:limit]

    elif query_type == "best":
        # Find policies with best outcomes
        sorted_policies = sorted(
            POLICY_VAULT_DB,
            key=lambda x: x.get("outcome", {}).get("reward", 0),
            reverse=True
        )
        results = sorted_policies[:limit]

    elif query_type == "recent":
        # Return most recent policies
        sorted_policies = sorted(
            POLICY_VAULT_DB,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        results = sorted_policies[:limit]

    else:
        results = POLICY_VAULT_DB[:limit]

    return {
        "query_type": query_type,
        "policies_found": len(results),
        "policies": results,
        "total_vault_size": len(POLICY_VAULT_DB)
    }


def save_policy_to_vault(plan: Dict, context: Dict, outcome: Dict) -> Dict:
    """
    Save a successful plan to the policy vault for future reference.

    Args:
        plan: The plan that was executed
        context: Grid context when plan was created
        outcome: Results after execution (reward, metrics, etc.)

    Returns:
        Dict with save confirmation
    """
    policy = {
        "policy_id": f"policy_{len(POLICY_VAULT_DB) + 1:04d}",
        "timestamp": datetime.now().isoformat(),
        "plan": plan,
        "context": context,
        "outcome": outcome
    }

    POLICY_VAULT_DB.append(policy)

    return {
        "status": "saved",
        "policy_id": policy["policy_id"],
        "vault_size": len(POLICY_VAULT_DB)
    }
