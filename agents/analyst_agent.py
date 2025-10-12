# weavehacks2/agents/analyst_agent.py
"""
AnalystAgent: Analyzes grid state using Qwen 2.5 LLM
- Now supports system prompt overrides so other components can keep evolving independently.
"""

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

    New:
    - system_prompt override (constructor or per-call) so Planner can evolve separately.
    """

    def __init__(self, name="AnalystAgent", model="qwen-2.5-72b", system_prompt=None):
        super().__init__(name)
        self.model = model
        self.conversation_history = []
        self._system_prompt_override = system_prompt  # << added

        # Initialize Weave client
        weave.init(os.getenv("WEAVE_PROJECT", "synergi-grid-optimization"))

    @weave.op()
    def run(self, input_state, conversation_history=None, system_prompt_override=None):
        """
        Analyze the grid state using Qwen 2.5.

        Args:
            input_state (dict): Current grid state from Digital Twin
            conversation_history (list, optional): Previous agent conversations
            system_prompt_override (str, optional): Per-call override for system prompt

        Returns:
            dict: Analysis results including trends, risks, and natural language summary
        """
        # Use provided history or agent's own history
        if conversation_history is not None:
            self.conversation_history = conversation_history

        # Choose system prompt (priority: per-call > ctor override > default)
        system_prompt = (
            system_prompt_override
            or self._system_prompt_override
            or self._build_system_prompt()
        )

        # Build the prompt for Qwen 2.5
        user_prompt = self._build_analysis_prompt(input_state)

        # Compose messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Add conversation history if present
        if self.conversation_history:
            messages = self.conversation_history + messages

        # Make LLM call
        response = self._call_qwen(messages)

        # Parse response
        analysis = {
            "agent": self.name,
            "timestamp": input_state.get("timestamp"),
            "analysis_text": response,
            "conversation_history": messages + [{"role": "assistant", "content": response}],
        }

        # Update internal history
        self.conversation_history = analysis["conversation_history"]

        return analysis

    # ---------------------------
    # Prompt builders & helpers
    # ---------------------------

    def _build_system_prompt(self):
        """Default system prompt for AnalystAgent"""
        return (
            "You are the AnalystAgent, an expert in power grid analysis.\n\n"
            "Your role is to:\n"
            "1. Analyze the current grid state data\n"
            "2. Identify supply-demand imbalances across nodes\n"
            "3. Detect high-risk areas (overload risk, low storage, etc.)\n"
            "4. Spot trends and patterns (demand spikes, weather impacts, etc.)\n"
            "5. Provide clear, concise analysis in natural language\n\n"
            "Focus on:\n"
            "- Which neighborhoods have deficits vs surplus\n"
            "- Risk levels (overload risk, n-1 margin)\n"
            "- Storage state of charge issues\n"
            "- Weather impacts on demand/supply\n"
            "- Fairness concerns across equity-weighted zones\n\n"
            "Be concise and actionable. Your analysis will be used by the PlannerAgent to make decisions."
        )

    def _build_analysis_prompt(self, state):
        """Build user prompt with grid state data"""
        # Extract key metrics
        timestamp = state.get("timestamp")
        drivers = state.get("drivers", {})
        kpis = state.get("kpis", {})
        nodes = state.get("nodes", {})

        # Summarize weather
        weather = drivers.get("weather", {})
        weather_summary = (
            f"Temp: {weather.get('temp_c')}°C, "
            f"Solar: {weather.get('solar_irradiance_wm2')}W/m², "
            f"Wind: {weather.get('wind_mps')}m/s"
        )

        # Summarize pricing
        price = drivers.get("price", {})
        price_summary = (
            f"DA: ${price.get('da_usd_per_kwh')}/kWh, "
            f"RT: ${price.get('rt_usd_per_kwh')}/kWh"
        )

        # Sample nodes (first 10)
        node_lines = []
        for node_id, node_data in list(nodes.items())[:10]:
            demand = node_data.get("demand_mw", 0.0)
            supply = node_data.get("supply_mw", 0.0)
            balance = supply - demand
            risk = node_data.get("risk", {}).get("overload", 0.0)
            soc = node_data.get("storage", {}).get("soc", 0.0)

            node_lines.append(
                f"  - {node_id}: Demand={demand:.1f}MW, "
                f"Supply={supply:.1f}MW, Balance={balance:+.1f}MW, "
                f"Risk={risk:.2f}, SOC={soc:.2f}"
            )

        nodes_summary = "\n".join(node_lines) if node_lines else "  (no nodes provided)"

        prompt = (
            f"Analyze the following SF power grid state:\n\n"
            f"**Time:** {timestamp}\n\n"
            f"**Weather:** {weather_summary}\n"
            f"**Prices:** {price_summary}\n\n"
            f"**City-wide KPIs:**\n"
            f"- Total Demand: {kpis.get('city_demand_mw', 0):.1f} MW\n"
            f"- Total Supply: {kpis.get('city_supply_mw', 0):.1f} MW\n"
            f"- Unserved Energy: {kpis.get('unserved_energy_proxy_mw', 0):.1f} MW\n"
            f"- Avg Overload Risk: {kpis.get('avg_overload_risk', 0):.3f}\n"
            f"- Fairness Index: {kpis.get('fairness_index', 0):.3f}\n\n"
            f"**Sample Nodes (first 10):**\n"
            f"{nodes_summary}\n\n"
            f"Provide a concise analysis covering:\n"
            f"1. Overall grid health status\n"
            f"2. Top 3 nodes with issues (deficit, risk, or storage)\n"
            f"3. Key trends or patterns observed\n"
            f"4. Recommended focus areas for the Planner"
        )

        return prompt

    def _call_qwen(self, messages):
        """Call Qwen 2.5 via Weave API"""
        try:
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

    def _fallback_analysis(self, _prompt_text):
        """Rule-based fallback if LLM is unavailable"""
        return (
            "[Fallback Analysis]\n"
            "Grid status: Operational with minor imbalances.\n"
            "Key issues: Supply deficit observed in several nodes.\n"
            "Recommendation: Increase supply or activate demand response in high-risk areas.\n\n"
            "Note: This is a fallback analysis. Qwen 2.5 LLM connection unavailable."
        )
