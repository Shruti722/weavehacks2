"""
AnalystAgent: Analyzes grid state using Qwen 2.5 LLM
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


class AnalystAgent(AgentBase):
    """
    Analyzes grid state and identifies demand/supply trends and risks using Qwen 2.5.

    Responsibilities:
    - Analyze current vs historical demand patterns
    - Identify supply-demand imbalances
    - Calculate risk indicators for each node
    - Detect anomalies or emerging issues
    - Maintain conversation history for RL
    """

    def __init__(self, name="AnalystAgent", model="qwen-2.5-72b"):
        super().__init__(name)
        self.model = model
        self.conversation_history = []

        # Initialize Weave client
        weave.init(os.getenv("WEAVE_PROJECT", "synergi-grid-optimization"))

    @weave.op()
    def run(self, input_state, conversation_history=None):
        """
        Analyze the grid state using Qwen 2.5.

        Args:
            input_state (dict): Current grid state from Digital Twin
            conversation_history (list, optional): Previous agent conversations

        Returns:
            dict: Analysis results including trends, risks, and natural language summary
        """
        # Use provided history or agent's own history
        if conversation_history is not None:
            self.conversation_history = conversation_history

        # Build the prompt for Qwen 2.5
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_analysis_prompt(input_state)

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

        # Parse response
        analysis = {
            "agent": self.name,
            "timestamp": input_state.get("timestamp"),
            "analysis_text": response,
            "conversation_history": messages + [{"role": "assistant", "content": response}]
        }

        # Update internal history
        self.conversation_history = analysis["conversation_history"]

        return analysis

    def _build_system_prompt(self):
        """Build system prompt for AnalystAgent"""
        return """You are the AnalystAgent, an expert in power grid analysis.

Your role is to:
1. Analyze the current grid state data
2. Identify supply-demand imbalances across nodes
3. Detect high-risk areas (overload risk, low storage, etc.)
4. Spot trends and patterns (demand spikes, weather impacts, etc.)
5. Provide clear, concise analysis in natural language

Focus on:
- Which neighborhoods have deficits vs surplus
- Risk levels (overload risk, n-1 margin)
- Storage state of charge issues
- Weather impacts on demand/supply
- Fairness concerns across equity-weighted zones

Be concise and actionable. Your analysis will be used by the PlannerAgent to make decisions."""

    def _build_analysis_prompt(self, state):
        """Build user prompt with grid state data"""
        # Extract key metrics
        timestamp = state.get("timestamp")
        drivers = state.get("drivers", {})
        kpis = state.get("kpis", {})
        nodes = state.get("nodes", {})

        # Summarize weather
        weather = drivers.get("weather", {})
        weather_summary = f"Temp: {weather.get('temp_c')}°C, Solar: {weather.get('solar_irradiance_wm2')}W/m², Wind: {weather.get('wind_mps')}m/s"

        # Summarize pricing
        price = drivers.get("price", {})
        price_summary = f"DA: ${price.get('da_usd_per_kwh')}/kWh, RT: ${price.get('rt_usd_per_kwh')}/kWh"

        # Top deficit and surplus nodes
        node_list = []
        for node_id, node_data in list(nodes.items())[:10]:  # Sample first 10 nodes
            demand = node_data.get("demand_mw", 0)
            supply = node_data.get("supply_mw", 0)
            balance = supply - demand
            risk = node_data.get("risk", {}).get("overload", 0)
            soc = node_data.get("storage", {}).get("soc", 0)

            node_list.append(f"  - {node_id}: Demand={demand:.1f}MW, Supply={supply:.1f}MW, Balance={balance:+.1f}MW, Risk={risk:.2f}, SOC={soc:.2f}")

        nodes_summary = "\n".join(node_list)

        prompt = f"""Analyze the following SF power grid state:

**Time:** {timestamp}

**Weather:** {weather_summary}
**Prices:** {price_summary}

**City-wide KPIs:**
- Total Demand: {kpis.get('city_demand_mw', 0):.1f} MW
- Total Supply: {kpis.get('city_supply_mw', 0):.1f} MW
- Unserved Energy: {kpis.get('unserved_energy_proxy_mw', 0):.1f} MW
- Avg Overload Risk: {kpis.get('avg_overload_risk', 0):.3f}
- Fairness Index: {kpis.get('fairness_index', 0):.3f}

**Sample Nodes (first 10):**
{nodes_summary}

Provide a concise analysis covering:
1. Overall grid health status
2. Top 3 nodes with issues (deficit, risk, or storage)
3. Key trends or patterns observed
4. Recommended focus areas for the Planner"""

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
            # Fallback: return rule-based analysis if LLM fails
            print(f"[AnalystAgent] LLM call failed: {e}, using fallback")
            return self._fallback_analysis(messages[-1]["content"])

    def _fallback_analysis(self, prompt):
        """Rule-based fallback if LLM is unavailable"""
        return """[Fallback Analysis]
Grid status: Operational with minor imbalances.
Key issues: Supply deficit observed in several nodes.
Recommendation: Increase supply or activate demand response in high-risk areas.

Note: This is a fallback analysis. Qwen 2.5 LLM connection unavailable."""
