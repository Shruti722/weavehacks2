"""
ActuatorAgent: Executes control actions on the grid using Qwen 2.5 LLM
"""

import json
import os
from dotenv import load_dotenv
import weave

load_dotenv()


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
    - Translate high-level actions into SPECIFIC equipment operations
    - Choose HOW to execute (batteries, generators, DR, load shifting)
    - Sequence operations (startup order, ramp rates, timing)
    - Validate feasibility and safety constraints
    """

    def __init__(self, name="ActuatorAgent", model="qwen-2.5-72b"):
        super().__init__(name)
        self.model = model
        self.conversation_history = []

        # Define available resources per node
        self.resources = {
            "battery": {"ramp_rate_mw_per_min": 10, "efficiency": 0.95},
            "gas_turbine": {"ramp_rate_mw_per_min": 2, "startup_min": 10, "cost_per_mwh": 80},
            "solar": {"ramp_rate_mw_per_min": 5, "variability": 0.1},
            "demand_response": {"max_reduction_percent": 15, "fatigue_penalty": 0.02},
            "load_shift": {"max_shift_mw": 3, "duration_min": 30}
        }

        # Initialize Weave client
        weave.init(os.getenv("WEAVE_PROJECT", "synergi-grid-optimization"))

    @weave.op()
    def run(self, input_state, conversation_history=None):
        """
        Execute planned actions on the grid using Qwen 2.5 for HOW decisions.

        Args:
            input_state (dict): Combined state + analysis + plan
            conversation_history (list, optional): Previous agent conversations

        Returns:
            dict: Execution results with detailed operation sequences
        """
        # Use provided history or agent's own history
        if conversation_history is not None:
            self.conversation_history = conversation_history

        # Build the prompt for Qwen 2.5
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_execution_prompt(input_state)

        # Call Qwen 2.5 via Weave
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Add conversation history
        if self.conversation_history:
            messages = self.conversation_history + messages

        # Make LLM call for HOW decisions
        response = self._call_qwen(messages)

        # Generate commands with HOW details
        commands = self._prepare_commands(input_state)
        validation = self._validate_actions(input_state)

        execution = {
            "agent": self.name,
            "timestamp": input_state.get("timestamp"),
            "execution_text": response,
            "commands": commands,
            "validation": validation,
            "conversation_history": messages + [{"role": "assistant", "content": response}]
        }

        # Update internal history
        self.conversation_history = execution["conversation_history"]

        return execution

    def _build_system_prompt(self):
        """Build system prompt for ActuatorAgent"""
        return f"""You are the ActuatorAgent, an expert in power system operations and equipment control.

Your role is to:
1. Review the PlannerAgent's WHAT actions
2. Decide HOW to execute each action using specific equipment
3. Sequence operations (order, timing, ramp rates)
4. Validate feasibility and safety

Available resources and their characteristics:
{json.dumps(self.resources, indent=2)}

For each planned action, specify:
- Which equipment to use (battery, gas_turbine, solar, DR, load_shift)
- Sequence order (what to activate first)
- Timing and ramp rates
- Safety considerations

Be specific about HOW to execute. Your instructions will control real equipment."""

    def _build_execution_prompt(self, state):
        """Build user prompt with planner's decisions"""
        timestamp = state.get("timestamp")
        plan = state.get("plan", {})
        plan_text = plan.get("plan_text", "No plan available")
        actions = plan.get("actions", [])

        # Summarize planned actions
        actions_summary = []
        for action in actions[:5]:  # Top 5 actions
            node_id = action.get("node_id")
            adjustments = action.get("adjustments", {})
            actions_summary.append(f"  - {node_id}: {adjustments}")

        actions_list = "\n".join(actions_summary) if actions_summary else "  No actions planned"

        prompt = f"""Based on the Planner's decisions, specify HOW to execute each action.

**Time:** {timestamp}

**Planner's Decision:**
{plan_text}

**Planned Actions (structured):**
{actions_list}

**Your task:**
For each action, specify HOW to execute it:
1. Which equipment/method to use (battery, gas turbine, solar, DR, etc.)
2. Sequence order (what happens first, second, third)
3. Ramp rates and timing
4. Safety checks and constraints

For example, if the plan says "increase supply by 8MW in Financial District":
- Step 1: Discharge battery (5MW, fast response, 0.5 min)
- Step 2: Ramp solar (3MW, 0.6 min)
- Safety: Check battery SOC > 30%, solar irradiance sufficient

Be specific and actionable."""

        return prompt

    def _call_qwen(self, messages):
        """Call Qwen 2.5 via Weave API"""
        try:
            # Use Weave's Model API for Qwen 2.5
            model = weave.Model(self.model)
            response = model.predict(messages=messages)

            # Extract text from response
            if isinstance(response, dict):
                return response.get("content", str(response))
            return str(response)

        except Exception as e:
            # Fallback: return rule-based execution if LLM fails
            print(f"[ActuatorAgent] LLM call failed: {e}, using fallback")
            return self._fallback_execution(messages[-1]["content"])

    def _fallback_execution(self, prompt):
        """Rule-based fallback if LLM is unavailable"""
        return """[Fallback Execution]
HOW to execute:
- Use battery discharge first (fast response)
- Then ramp solar if available
- Gas turbine as backup
- Validate all safety constraints

Note: This is a fallback execution plan. Qwen 2.5 LLM connection unavailable."""

    def _prepare_commands(self, state):
        """Convert planned actions into SPECIFIC executable commands with HOW details"""
        plan = state.get("plan", {})
        actions = plan.get("actions", [])
        nodes = state.get("nodes", [])

        commands = []
        for i, action in enumerate(actions):
            node_id = action.get("node_id")
            adjustments = action.get("adjustments", {})

            # Get node state
            node = nodes[i] if i < len(nodes) else {}
            storage_soc = node.get("storage_level", 0)
            current_demand = node.get("demand_mw", 0)

            command = {
                "node_id": node_id,
                "operations": []  # List of specific operations
            }

            # Process each adjustment and decide HOW to execute
            for adj_type, value in adjustments.items():

                if "increase_supply" in adj_type:
                    # Need more supply - choose method
                    command["operations"].extend(self._how_to_increase_supply(value, storage_soc))

                elif "reduce_supply" in adj_type:
                    # Need less supply
                    command["operations"].extend(self._how_to_reduce_supply(value))

                elif "charge_storage" in adj_type:
                    # Charge battery
                    command["operations"].append({
                        "method": "battery_charge",
                        "target_mw": value,
                        "ramp_rate": self.resources["battery"]["ramp_rate_mw_per_min"],
                        "duration_min": value / self.resources["battery"]["ramp_rate_mw_per_min"],
                        "sequence": 1
                    })

                elif "discharge_storage" in adj_type:
                    # Discharge battery
                    command["operations"].append({
                        "method": "battery_discharge",
                        "target_mw": value,
                        "ramp_rate": self.resources["battery"]["ramp_rate_mw_per_min"],
                        "duration_min": value / self.resources["battery"]["ramp_rate_mw_per_min"],
                        "sequence": 1
                    })

                elif "reduce_demand" in adj_type:
                    # Use demand response
                    command["operations"].extend(self._how_to_reduce_demand(value, current_demand))

            commands.append(command)

        return commands

    def _how_to_increase_supply(self, needed_mw, storage_soc):
        """Decide HOW to increase supply using available resources"""
        operations = []

        # Strategy: Use fastest/cheapest methods first
        remaining = needed_mw

        # 1. Discharge battery (if available and fast)
        if storage_soc > 0.3 and remaining > 0:
            battery_contribution = min(remaining, 5.0)  # Max 5MW from battery
            operations.append({
                "method": "battery_discharge",
                "target_mw": battery_contribution,
                "ramp_rate": self.resources["battery"]["ramp_rate_mw_per_min"],
                "duration_min": battery_contribution / self.resources["battery"]["ramp_rate_mw_per_min"],
                "sequence": 1,  # Do this first (fast)
                "reason": "Fast response from battery"
            })
            remaining -= battery_contribution

        # 2. Ramp up solar (if daytime)
        if remaining > 0:
            solar_contribution = min(remaining, 3.0)
            operations.append({
                "method": "solar_ramp_up",
                "target_mw": solar_contribution,
                "ramp_rate": self.resources["solar"]["ramp_rate_mw_per_min"],
                "duration_min": solar_contribution / self.resources["solar"]["ramp_rate_mw_per_min"],
                "sequence": 2,
                "reason": "Clean energy, no fuel cost"
            })
            remaining -= solar_contribution

        # 3. Start gas turbine (if still needed)
        if remaining > 0:
            operations.append({
                "method": "gas_turbine_start",
                "target_mw": remaining,
                "ramp_rate": self.resources["gas_turbine"]["ramp_rate_mw_per_min"],
                "startup_delay_min": self.resources["gas_turbine"]["startup_min"],
                "duration_min": remaining / self.resources["gas_turbine"]["ramp_rate_mw_per_min"],
                "sequence": 3,  # Do this last (slow startup)
                "cost_per_mwh": self.resources["gas_turbine"]["cost_per_mwh"],
                "reason": "Backup dispatchable generation"
            })

        return operations

    def _how_to_reduce_supply(self, reduction_mw):
        """Decide HOW to reduce supply"""
        operations = []

        # Strategy: Reduce expensive sources first
        remaining = reduction_mw

        # 1. Reduce gas turbine output first (most expensive)
        if remaining > 0:
            turbine_reduction = min(remaining, 10.0)
            operations.append({
                "method": "gas_turbine_ramp_down",
                "target_mw": -turbine_reduction,
                "ramp_rate": self.resources["gas_turbine"]["ramp_rate_mw_per_min"],
                "duration_min": turbine_reduction / self.resources["gas_turbine"]["ramp_rate_mw_per_min"],
                "sequence": 1,
                "reason": "Reduce expensive generation first"
            })
            remaining -= turbine_reduction

        # 2. Curtail solar if needed (last resort)
        if remaining > 0:
            operations.append({
                "method": "solar_curtail",
                "target_mw": -remaining,
                "sequence": 2,
                "reason": "Curtail renewable as last resort"
            })

        return operations

    def _how_to_reduce_demand(self, reduction_mw, current_demand):
        """Decide HOW to reduce demand"""
        operations = []

        max_dr_mw = current_demand * (self.resources["demand_response"]["max_reduction_percent"] / 100)
        actual_reduction = min(reduction_mw, max_dr_mw)

        # Demand Response activation
        operations.append({
            "method": "demand_response_activate",
            "target_mw": actual_reduction,
            "duration_min": 60,  # Typical DR event duration
            "sequence": 1,
            "max_reduction_percent": self.resources["demand_response"]["max_reduction_percent"],
            "reason": f"DR event: reduce {actual_reduction:.1f}MW for 1 hour"
        })

        # Load shifting (defer non-critical loads)
        if reduction_mw > actual_reduction:
            shift_amount = min(reduction_mw - actual_reduction, self.resources["load_shift"]["max_shift_mw"])
            operations.append({
                "method": "load_shift",
                "target_mw": shift_amount,
                "shift_to_time": "off_peak",
                "duration_min": self.resources["load_shift"]["duration_min"],
                "sequence": 2,
                "reason": f"Shift {shift_amount:.1f}MW to off-peak hours"
            })

        return operations

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
