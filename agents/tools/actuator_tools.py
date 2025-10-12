"""
ActuatorAgent Tools - Enhanced for Execution Wisdom
Tools that help Actuator execute SAFELY and EFFICIENTLY
"""

from typing import Dict, List


def validate_execution_sequence(planned_actions: List[Dict], grid_state: Dict) -> Dict:
    """
    Determine the SAFEST execution order for actions.
    Some actions must happen before others to avoid instability.

    Returns optimal sequence with reasoning.
    """
    # Categorize actions by type
    supply_increases = []
    demand_reductions = []
    storage_operations = []

    for i, action in enumerate(planned_actions):
        action_with_idx = {**action, "original_index": i}
        action_type = action.get("action_type", "").lower()

        if "increase_supply" in action_type or "discharge" in action_type:
            supply_increases.append(action_with_idx)
        elif "reduce_demand" in action_type:
            demand_reductions.append(action_with_idx)
        elif "charge" in action_type:
            storage_operations.append(action_with_idx)
        else:
            supply_increases.append(action_with_idx)  # Default

    # Optimal sequence:
    # 1. Supply increases FIRST (fixes root cause)
    # 2. Demand reductions SECOND (customer impact)
    # 3. Storage operations LAST (balance)

    optimal_sequence = supply_increases + demand_reductions + storage_operations

    return {
        "original_order_count": len(planned_actions),
        "optimal_sequence": [
            {
                "execution_order": i+1,
                "node_id": a.get("node_id"),
                "action_type": a.get("action_type"),
                "original_index": a.get("original_index")
            }
            for i, a in enumerate(optimal_sequence)
        ],
        "sequence_changed": optimal_sequence != planned_actions,
        "insight": (
            f"OPTIMAL SEQUENCE: Execute {len(supply_increases)} supply increases first (fixes deficit), "
            f"then {len(demand_reductions)} demand reductions (minimizes customer impact), "
            f"then {len(storage_operations)} storage operations (balancing). "
            f"This prevents voltage dips and ensures stability."
        ),
        "safety_notes": [
            "Supply increases should ramp gradually (0.5 MW/min)" if supply_increases else None,
            "Coordinate demand response to avoid sudden drops" if demand_reductions else None,
            "Monitor SOC after each discharge to prevent over-depletion" if storage_operations else None
        ]
    }


def estimate_execution_time(actions: List[Dict]) -> Dict:
    """
    Calculate how long execution will take, including ramp times.
    Helps Actuator set realistic expectations.
    """
    total_time_min = 0
    action_timings = []

    for action in actions:
        action_type = action.get("action_type", "").lower()
        target_mw = action.get("target_mw", 0)

        # Ramp rates (MW/min)
        if "discharge" in action_type or "increase_supply" in action_type:
            ramp_rate = 0.5  # Conservative 0.5 MW/min
            ramp_time = target_mw / ramp_rate
            execution_time = ramp_time + 2  # +2 min for startup/stabilization

        elif "reduce_demand" in action_type:
            # Demand response: notification + implementation
            execution_time = 5 + target_mw * 0.5  # 5 min base + scaling

        elif "charge" in action_type:
            ramp_rate = 0.3
            execution_time = target_mw / ramp_rate + 1

        else:
            execution_time = 5  # Default

        action_timings.append({
            "node_id": action.get("node_id"),
            "action_type": action_type,
            "execution_time_min": round(execution_time, 1)
        })

        total_time_min += execution_time

    # Can we parallelize?
    max_parallel_actions = 3  # Grid can handle 3 simultaneous actions
    if len(actions) > max_parallel_actions:
        # Sequential batches
        batches = (len(actions) + max_parallel_actions - 1) // max_parallel_actions
        parallel_time = total_time_min / batches
    else:
        parallel_time = max([a["execution_time_min"] for a in action_timings]) if action_timings else 0

    return {
        "total_actions": len(actions),
        "sequential_time_min": round(total_time_min, 1),
        "parallel_time_min": round(parallel_time, 1),
        "time_saved_min": round(total_time_min - parallel_time, 1),
        "action_timings": action_timings,
        "insight": (
            f"SEQUENTIAL execution: {total_time_min:.1f} minutes total. "
            f"PARALLEL execution (3 at a time): {parallel_time:.1f} minutes. "
            f"Recommendation: Execute in parallel to save {total_time_min - parallel_time:.1f} minutes."
        )
    }


def identify_execution_risks(actions: List[Dict], grid_state: Dict) -> Dict:
    """
    Identify potential risks BEFORE executing.
    Prevents Actuator from causing cascading failures.
    """
    risks = []
    warnings = []
    nodes = grid_state.get("nodes", {})

    # Risk 1: Over-depleting storage
    nodes_with_low_soc = []
    for action in actions:
        if "discharge" in action.get("action_type", "").lower():
            node_id = action.get("node_id")
            if node_id in nodes:
                soc = nodes[node_id].get("storage", {}).get("soc", 0)
                if soc < 0.15:
                    risks.append({
                        "risk_type": "storage_depletion",
                        "node_id": node_id,
                        "current_soc": round(soc, 2),
                        "severity": "HIGH",
                        "description": f"Discharging at SOC={soc:.1%} will critically deplete storage. No backup for next hour."
                    })
                    nodes_with_low_soc.append(node_id)

    # Risk 2: Simultaneous actions on connected nodes
    nodes_affected = [a.get("node_id") for a in actions]
    if len(set(nodes_affected)) < len(nodes_affected):
        risks.append({
            "risk_type": "duplicate_node_action",
            "severity": "MEDIUM",
            "description": "Multiple actions planned for same node. May cause oscillations."
        })

    # Risk 3: Large sudden changes
    large_actions = [a for a in actions if a.get("target_mw", 0) > 10]
    if large_actions:
        warnings.append({
            "warning_type": "large_change",
            "count": len(large_actions),
            "description": f"{len(large_actions)} actions exceed 10 MW. Ensure gradual ramp to avoid voltage transients."
        })

    # Risk 4: No backup plan
    total_discharge = sum(a.get("target_mw", 0) for a in actions if "discharge" in a.get("action_type", "").lower())
    if total_discharge > 50:
        warnings.append({
            "warning_type": "high_discharge",
            "total_mw": round(total_discharge, 1),
            "description": f"Total discharge of {total_discharge:.0f} MW is very high. Ensure generation backup is available."
        })

    return {
        "total_risks": len(risks),
        "total_warnings": len(warnings),
        "risks": risks,
        "warnings": warnings,
        "execution_recommended": len([r for r in risks if r.get("severity") == "HIGH"]) == 0,
        "insight": (
            f"✓ SAFE TO EXECUTE: {len(warnings)} warnings but no blocking risks."
            if len([r for r in risks if r.get("severity") == "HIGH"]) == 0
            else f"⚠️ {len([r for r in risks if r.get('severity') == 'HIGH'])} HIGH-SEVERITY RISKS DETECTED. "
                 f"Recommend revising plan before execution: {risks[0].get('description', '')}"
        )
    }


def charge_battery(node_id: str, power_mw: float, duration_min: float = 5) -> Dict:
    """
    Charge battery storage at a specific node.

    Args:
        node_id: Node where battery is located
        power_mw: Charging power (MW)
        duration_min: How long to charge (minutes)

    Returns:
        Dict with charge command details
    """
    # Battery charging specs
    max_charge_rate_mw = 10  # Max 10 MW charging rate
    efficiency = 0.95

    # Validate charging rate
    actual_power = min(power_mw, max_charge_rate_mw)

    # Calculate energy stored
    energy_mwh = actual_power * (duration_min / 60) * efficiency

    return {
        "command": "charge_battery",
        "node_id": node_id,
        "power_mw": actual_power,
        "duration_min": duration_min,
        "energy_stored_mwh": round(energy_mwh, 3),
        "efficiency": efficiency,
        "status": "ready_to_execute",
        "execution_details": {
            "ramp_rate_mw_per_min": max_charge_rate_mw / 2,
            "sequence": 1
        }
    }


def discharge_battery(node_id: str, power_mw: float, duration_min: float = 5) -> Dict:
    """
    Discharge battery storage to supply power to the grid.

    Args:
        node_id: Node where battery is located
        power_mw: Discharge power (MW)
        duration_min: How long to discharge (minutes)

    Returns:
        Dict with discharge command details
    """
    # Battery discharge specs
    max_discharge_rate_mw = 10
    efficiency = 0.95

    # Validate discharge rate
    actual_power = min(power_mw, max_discharge_rate_mw)

    # Calculate energy released
    energy_mwh = actual_power * (duration_min / 60) / efficiency

    return {
        "command": "discharge_battery",
        "node_id": node_id,
        "power_mw": actual_power,
        "duration_min": duration_min,
        "energy_released_mwh": round(energy_mwh, 3),
        "efficiency": efficiency,
        "status": "ready_to_execute",
        "execution_details": {
            "ramp_rate_mw_per_min": max_discharge_rate_mw / 2,
            "sequence": 1
        }
    }


def reconfigure_lines(source_node: str, target_node: str, action: str, power_mw: float = None) -> Dict:
    """
    Reconfigure transmission lines to reduce load or redirect power flow.

    Args:
        source_node: Node where power comes from
        target_node: Node where power goes to
        action: Type of reconfiguration ("open_line", "close_line", "redirect_flow", "reduce_capacity")
        power_mw: Power to redirect/reduce (if applicable)

    Returns:
        Dict with line reconfiguration command
    """
    edge_id = f"{source_node}|{target_node}"

    commands = {
        "open_line": {
            "action": "open_line",
            "description": f"Open line between {source_node} and {target_node} to isolate nodes",
            "impact": "Stops power flow on this line",
            "reversible": True
        },
        "close_line": {
            "action": "close_line",
            "description": f"Close/reconnect line between {source_node} and {target_node}",
            "impact": "Enables power flow on this line",
            "reversible": True
        },
        "redirect_flow": {
            "action": "redirect_flow",
            "description": f"Redirect {power_mw}MW from {source_node} to {target_node}",
            "impact": f"Changes power flow by {power_mw}MW",
            "power_mw": power_mw,
            "reversible": True
        },
        "reduce_capacity": {
            "action": "reduce_capacity",
            "description": f"Reduce line capacity between {source_node} and {target_node} by {power_mw}MW",
            "impact": f"Limits flow to reduce load on overloaded line",
            "power_mw": power_mw,
            "reversible": True
        }
    }

    if action not in commands:
        return {
            "command": "reconfigure_lines",
            "status": "error",
            "error": f"Unknown action: {action}. Valid actions: {list(commands.keys())}"
        }

    config = commands[action]

    return {
        "command": "reconfigure_lines",
        "edge_id": edge_id,
        "source_node": source_node,
        "target_node": target_node,
        "action": config["action"],
        "description": config["description"],
        "impact": config["impact"],
        "power_mw": power_mw if power_mw else None,
        "reversible": config["reversible"],
        "status": "ready_to_execute",
        "execution_details": {
            "switching_time_sec": 10,
            "sequence": 2
        }
    }


def update_grid_twin(commands: List[Dict], current_state: Dict) -> Dict:
    """
    Update the Digital Twin with executed commands to reflect new grid state.

    Args:
        commands: List of commands that were executed
        current_state: Current grid state before commands

    Returns:
        Dict with updated grid state
    """
    import copy

    # Deep copy state to avoid modifying original
    new_state = copy.deepcopy(current_state)

    execution_log = []

    for cmd in commands:
        command_type = cmd.get("command")
        node_id = cmd.get("node_id")

        if command_type == "charge_battery":
            # Update battery SOC
            if node_id in new_state.get("nodes", {}):
                node = new_state["nodes"][node_id]
                storage = node.get("storage", {})

                current_soc = storage.get("soc", 0)
                capacity_mwh = storage.get("energy_mwh_cap", 50)
                energy_stored = cmd.get("energy_stored_mwh", 0)

                new_soc = min(1.0, current_soc + (energy_stored / capacity_mwh))
                storage["soc"] = new_soc

                execution_log.append({
                    "command": "charge_battery",
                    "node_id": node_id,
                    "old_soc": current_soc,
                    "new_soc": new_soc,
                    "status": "executed"
                })

        elif command_type == "discharge_battery":
            # Update battery SOC and add to supply
            if node_id in new_state.get("nodes", {}):
                node = new_state["nodes"][node_id]
                storage = node.get("storage", {})

                current_soc = storage.get("soc", 0)
                capacity_mwh = storage.get("energy_mwh_cap", 50)
                energy_released = cmd.get("energy_released_mwh", 0)

                new_soc = max(0.0, current_soc - (energy_released / capacity_mwh))
                storage["soc"] = new_soc

                # Add discharged power to supply
                power_mw = cmd.get("power_mw", 0)
                node["supply_mw"] = node.get("supply_mw", 0) + power_mw

                execution_log.append({
                    "command": "discharge_battery",
                    "node_id": node_id,
                    "old_soc": current_soc,
                    "new_soc": new_soc,
                    "power_added_mw": power_mw,
                    "status": "executed"
                })

        elif command_type == "reconfigure_lines":
            # Update edge state
            edge_id = cmd.get("edge_id")
            action = cmd.get("action")

            if edge_id in new_state.get("edges", {}):
                edge = new_state["edges"][edge_id]

                if action == "open_line":
                    edge["active"] = False
                    edge["flow_mw"] = 0
                elif action == "close_line":
                    edge["active"] = True
                elif action == "redirect_flow":
                    power_mw = cmd.get("power_mw", 0)
                    edge["flow_mw"] = edge.get("flow_mw", 0) + power_mw
                elif action == "reduce_capacity":
                    power_mw = cmd.get("power_mw", 0)
                    edge["capacity_mw"] = max(0, edge.get("capacity_mw", 50) - power_mw)

                execution_log.append({
                    "command": "reconfigure_lines",
                    "edge_id": edge_id,
                    "action": action,
                    "status": "executed"
                })

    # Recalculate KPIs after updates
    nodes = new_state.get("nodes", {})
    total_demand = sum(n.get("demand_mw", 0) for n in nodes.values())
    total_supply = sum(n.get("supply_mw", 0) for n in nodes.values())

    new_state["kpis"]["city_demand_mw"] = total_demand
    new_state["kpis"]["city_supply_mw"] = total_supply
    new_state["kpis"]["unserved_energy_proxy_mw"] = max(0, total_demand - total_supply)

    return {
        "command": "update_grid_twin",
        "status": "updated",
        "updated_state": new_state,
        "execution_log": execution_log,
        "changes_count": len(execution_log),
        "timestamp": new_state.get("timestamp")
    }
