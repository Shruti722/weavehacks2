"""
PlannerAgent: Decides optimal grid adjustments using Qwen 2.5 LLM
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


class PlannerAgent(AgentBase):
    """
    Decides optimal grid adjustments to maximize efficiency and minimize cost/risk using Qwen 2.5.

    Responsibilities:
    - Generate action plans based on analyst recommendations
    - Optimize for cost, risk, and fairness
    - Decide load shifts and storage utilization
    - Balance supply-demand across nodes
    - Maintain conversation history for RL
    """

    def __init__(self, name="PlannerAgent", model="qwen-2.5-72b"):
        super().__init__(name)
        self.model = model
        self.conversation_history = []

        # Initialize Weave client
        weave.init(os.getenv("WEAVE_PROJECT", "synergi-grid-optimization"))

    @weave.op()
    def run(self, input_state, conversation_history=None):
        """
        Generate optimal action plan for the grid using Qwen 2.5.

        Args:
            input_state (dict): Combined state from Digital Twin + Analyst analysis
            conversation_history (list, optional): Previous agent conversations

        Returns:
            dict: Action plan with specific adjustments for each node
        """
        # Use provided history or agent's own history
        if conversation_history is not None:
            self.conversation_history = conversation_history

        # Build the prompt for Qwen 2.5
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_planning_prompt(input_state)

        # Call Qwen 2.5 via Weave
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Add conversation history
        if self.conversation_history:
            messages = self.conversation_history + messages

        # Make LLM call
        response = self._call_qwen(messages)

        # Parse response into action plan
        plan = {
            "agent": self.name,
            "timestamp": input_state.get("timestamp"),
            "plan_text": response,
            "actions": self._extract_actions(response, input_state),
            "conversation_history": messages + [{"role": "assistant", "content": response}]
        }

        # Update internal history
        self.conversation_history = plan["conversation_history"]

        return plan

    def _build_system_prompt(self):
        """Build system prompt for PlannerAgent"""
        return """You are the PlannerAgent, an expert in power grid optimization and control.

Your role is to:
1. Review the AnalystAgent's findings
2. Decide WHAT actions to take for each problematic node
3. Prioritize actions based on: cost, risk, fairness, and feasibility
4. Balance trade-offs between competing objectives

Available actions for each node:
- increase_supply: Add more generation (specify MW)
- reduce_supply: Decrease generation (specify MW)
- charge_storage: Store excess energy (specify MW)
- discharge_storage: Release stored energy (specify MW)
- reduce_demand: Activate demand response (specify MW)

Output format:
For each node that needs action, specify:
- Node ID
- Action type
- Target MW
- Rationale

Be specific and actionable. Your plan will be executed by the ActuatorAgent."""

    def _build_planning_prompt(self, state):
        """Build user prompt with grid state + analyst findings"""
        timestamp = state.get("timestamp")
        analysis = state.get("analysis", {})
        analyst_text = analysis.get("analysis_text", "No analysis available")

        # Extract high-risk nodes
        nodes = state.get("nodes", {})
        high_risk_nodes = []
        for node_id, node_data in nodes.items():
            risk = node_data.get("risk", {}).get("overload", 0)
            if risk > 0.4:  # High risk threshold
                demand = node_data.get("demand_mw", 0)
                supply = node_data.get("supply_mw", 0)
                balance = supply - demand
                soc = node_data.get("storage", {}).get("soc", 0)
                high_risk_nodes.append(
                    f"  - {node_id}: Balance={balance:+.1f}MW, Risk={risk:.2f}, SOC={soc:.2f}"
                )

        risk_summary = "\n".join(high_risk_nodes) if high_risk_nodes else "  No high-risk nodes"

        # Get KPIs
        kpis = state.get("kpis", {})

        prompt = f"""Based on the current grid state and analyst findings, create an action plan.

**Time:** {timestamp}

**Analyst Report:**
{analyst_text}

**High-Risk Nodes (Risk > 0.4):**
{risk_summary}

**City-wide KPIs:**
- Total Demand: {kpis.get('city_demand_mw', 0):.1f} MW
- Total Supply: {kpis.get('city_supply_mw', 0):.1f} MW
- Unserved Energy: {kpis.get('unserved_energy_proxy_mw', 0):.1f} MW
- Avg Overload Risk: {kpis.get('avg_overload_risk', 0):.3f}
- Fairness Index: {kpis.get('fairness_index', 0):.3f}

**Your task:**
Create a plan to address the top 3-5 most critical issues. For each action:
1. Specify the node ID
2. Choose action type (increase_supply, discharge_storage, reduce_demand, etc.)
3. Specify target MW
4. Explain your rationale

Optimize for:
- Minimize cost and risk
- Maximize fairness
- Prioritize equity-weighted zones (Tenderloin, Bayview, etc.)"""

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
            # Fallback: return rule-based plan if LLM fails
            print(f"[PlannerAgent] LLM call failed: {e}, using fallback")
            return self._fallback_plan(messages[-1]["content"])

    def _extract_actions(self, plan_text, state):
        """Parse natural language plan into structured actions"""
        # For now, use simple rule-based extraction
        # In production, use structured output or more sophisticated parsing

        actions = []
        nodes = state.get("nodes", {})

        # Find nodes with imbalances and create actions
        for node_id, node_data in list(nodes.items())[:10]:  # Top 10 nodes
            demand = node_data.get("demand_mw", 0)
            supply = node_data.get("supply_mw", 0)
            balance = supply - demand
            risk = node_data.get("risk", {}).get("overload", 0)
            soc = node_data.get("storage", {}).get("soc", 0)

            action = {"node_id": node_id, "adjustments": {}}

            # Simple rule-based planning (will be replaced by LLM-extracted actions)
            if balance < -3 and risk > 0.3:  # Deficit + high risk
                if soc > 0.3:
                    action["adjustments"]["discharge_storage"] = min(abs(balance) * 0.5, 5.0)
                else:
                    action["adjustments"]["increase_supply"] = abs(balance) * 0.8

            elif balance > 3:  # Surplus
                if soc < 0.8:
                    action["adjustments"]["charge_storage"] = min(balance * 0.6, 5.0)
                else:
                    action["adjustments"]["reduce_supply"] = balance * 0.5

            if action["adjustments"]:
                actions.append(action)

        return actions

    def _fallback_plan(self, prompt):
        """Rule-based fallback if LLM is unavailable"""
        return """[Fallback Plan]
Action 1: Increase supply in high-deficit nodes by 5-10 MW
Action 2: Discharge batteries in nodes with SOC > 0.5
Action 3: Activate demand response in equity zones if needed

Note: This is a fallback plan. Qwen 2.5 LLM connection unavailable."""
