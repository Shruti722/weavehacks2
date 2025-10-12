"""
ActuatorAgent Tools
Tools for executing grid control actions
"""

from typing import Dict, List


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
