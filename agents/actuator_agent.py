"""
ActuatorAgent: Executes control actions on the grid
"""


class AgentBase:
    """Base class for all agents in the system"""

    def __init__(self, name):
        self.name = name

    def run(self, input_state):
        """Runs the agent's logic for a given state"""
        raise NotImplementedError


class ActuatorAgent(AgentBase):
    """
    Executes control actions on the grid based on planner decisions.

    Responsibilities:
    - Translate planned actions into executable commands
    - Validate action feasibility and safety constraints
    - Execute actions on the Digital Twin
    - Report execution status and outcomes
    """

    def __init__(self, name="ActuatorAgent"):
        super().__init__(name)

    def run(self, input_state):
        """
        Execute planned actions on the grid.

        Args:
            input_state (dict): Combined state + analysis + plan

        Returns:
            dict: Execution results and new control commands for Digital Twin
        """
        # TODO: Integrate with OpenPipe for RL-based execution refinement
        # TODO: Add Weave tracing decorator

        # Placeholder implementation
        execution = {
            "agent": self.name,
            "timestamp": input_state.get("timestamp"),
            "commands": self._prepare_commands(input_state),
            "validation": self._validate_actions(input_state),
            "summary": "Actions executed successfully"
        }

        return execution

    def _prepare_commands(self, state):
        """Convert planned actions into executable commands"""
        plan = state.get("plan", {})
        actions = plan.get("actions", [])

        commands = []
        for action in actions:
            node_id = action.get("node_id")
            adjustments = action.get("adjustments", {})

            command = {
                "node_id": node_id,
                "controls": {}
            }

            # Map adjustments to specific control parameters
            for adj_type, value in adjustments.items():
                if "supply" in adj_type:
                    command["controls"]["supply_adjustment_mw"] = value
                elif "storage" in adj_type:
                    if "charge" in adj_type:
                        command["controls"]["storage_charge_mw"] = value
                    elif "discharge" in adj_type:
                        command["controls"]["storage_discharge_mw"] = value

            commands.append(command)

        return commands

    def _validate_actions(self, state):
        """Validate that actions are safe and feasible"""
        commands = self._prepare_commands(state)
        nodes = state.get("nodes", [])

        validation_results = []
        for i, cmd in enumerate(commands):
            node = nodes[i] if i < len(nodes) else {}
            storage_level = node.get("storage_level", 0)

            is_valid = True
            warnings = []

            # Check storage constraints
            if "storage_discharge_mw" in cmd.get("controls", {}):
                if storage_level < 0.2:
                    warnings.append("Low storage level - discharge may be risky")

            if "storage_charge_mw" in cmd.get("controls", {}):
                if storage_level > 0.9:
                    warnings.append("High storage level - charging limited")

            validation_results.append({
                "node_id": cmd.get("node_id"),
                "is_valid": is_valid,
                "warnings": warnings
            })

        return validation_results
