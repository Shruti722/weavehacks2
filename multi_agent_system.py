"""
SynErgi Multi-Agent System using LangGraph
Agent-to-agent communication with ReAct pattern
"""

import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# Import our custom tools
from agents.tools import (
    load_forecast, risk_scan, anomaly_detection,
    simulate_plan, cost_risk_analysis, policy_vault,
    charge_battery, discharge_battery, reconfigure_lines, update_grid_twin
)

load_dotenv()


# Define the shared state for all agents
class AgentState(TypedDict):
    """Shared state passed between agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    grid_state: dict
    analyst_output: dict
    planner_output: dict
    actuator_output: dict
    next_agent: str


# Convert our tools to LangChain tool format
@tool
def query_grid_state(query: str) -> dict:
    """
    Query the Digital Twin for current grid state or specific information.

    Args:
        query: What to query (e.g., "current_state", "node:SF_Downtown", "edges")
    """
    # This will be populated from the state
    return {"status": "tool_call", "query": query}


@tool
def analyst_load_forecast(nodes: list, forecast_horizon_min: int = 30) -> dict:
    """Forecast energy allocation for nodes"""
    return load_forecast(nodes, forecast_horizon_min)


@tool
def analyst_risk_scan(nodes: list, threshold: float = 0.4) -> dict:
    """Scan grid for risk indicators"""
    return risk_scan(nodes, threshold)


@tool
def analyst_anomaly_detection(nodes: list) -> dict:
    """Detect anomalies in grid behavior"""
    return anomaly_detection(nodes)


@tool
def planner_simulate_plan(current_state: dict, planned_actions: list) -> dict:
    """Simulate what happens when plan is executed"""
    return simulate_plan(current_state, planned_actions)


@tool
def planner_cost_risk_analysis(current_state: dict, simulated_state: dict, planned_actions: list) -> dict:
    """Analyze cost and risk of the plan"""
    return cost_risk_analysis(current_state, simulated_state, planned_actions)


@tool
def planner_policy_vault(query_type: str, context: dict = None, limit: int = 5) -> dict:
    """Retrieve past successful plans"""
    return policy_vault(query_type, context, limit)


@tool
def actuator_charge_battery(node_id: str, power_mw: float, duration_min: float = 5) -> dict:
    """Charge battery at node"""
    return charge_battery(node_id, power_mw, duration_min)


@tool
def actuator_discharge_battery(node_id: str, power_mw: float, duration_min: float = 5) -> dict:
    """Discharge battery at node"""
    return discharge_battery(node_id, power_mw, duration_min)


@tool
def actuator_reconfigure_lines(source_node: str, target_node: str, action: str, power_mw: float = None) -> dict:
    """Reconfigure transmission lines"""
    return reconfigure_lines(source_node, target_node, action, power_mw)


@tool
def actuator_update_grid_twin(commands: list, current_state: dict) -> dict:
    """Update Digital Twin with executed commands"""
    return update_grid_twin(commands, current_state)


# Agent tool assignments
analyst_tools = [
    query_grid_state,
    analyst_load_forecast,
    analyst_risk_scan,
    analyst_anomaly_detection
]

planner_tools = [
    query_grid_state,
    planner_simulate_plan,
    planner_cost_risk_analysis,
    planner_policy_vault
]

actuator_tools = [
    actuator_charge_battery,
    actuator_discharge_battery,
    actuator_reconfigure_lines,
    actuator_update_grid_twin
]


class SynErgiMultiAgentSystem:
    """
    Multi-agent system for grid optimization using LangGraph.

    Agent flow:
    1. DigitalTwin → AnalystAgent (provides data)
    2. AnalystAgent → PlannerAgent (sends analysis)
    3. PlannerAgent ↔ AnalystAgent (can query for more details)
    4. PlannerAgent ↔ DigitalTwin (can simulate changes)
    5. PlannerAgent → ActuatorAgent (sends plan)
    6. ActuatorAgent ↔ PlannerAgent (can ask for feedback)
    7. ActuatorAgent → DigitalTwin (updates state)
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        """
        Initialize multi-agent system with W&B Inference.

        Default: Qwen2.5 14B Instruct
        - 14B parameters (good for RL fine-tuning)
        - Strong reasoning capabilities
        - Supports tool use and multi-turn conversations

        Other options:
        - meta-llama/Llama-3.1-8B-Instruct (8B, lighter)
        - microsoft/Phi-4-mini-instruct (3.8B, very lightweight)
        """
        self.model_name = model_name
        self.graph = self._build_graph()

    def _create_llm_with_tools(self, tools):
        """Create LLM instance with tools bound"""
        from langchain_openai import ChatOpenAI

        # Use W&B Inference API (OpenAI-compatible)
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError(
                "WANDB_API_KEY not found. Get it from https://wandb.ai/authorize"
            )

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=api_key,
            base_url="https://api.inference.wandb.ai/v1",  # Correct W&B Inference endpoint
            temperature=0.3,  # Lower temp for reasoning
            max_tokens=4096,  # Detailed responses
            request_timeout=120  # Timeout for inference
        )

        return llm.bind_tools(tools)

    def digital_twin_node(self, state: AgentState) -> AgentState:
        """Digital Twin node - provides grid state data"""
        grid_state = state.get("grid_state", {})

        message = SystemMessage(
            content=f"""Digital Twin State Update:
Timestamp: {grid_state.get('timestamp')}
Total Demand: {grid_state.get('kpis', {}).get('city_demand_mw', 0):.1f} MW
Total Supply: {grid_state.get('kpis', {}).get('city_supply_mw', 0):.1f} MW
Nodes: {len(grid_state.get('nodes', {}))}

State data ready for Analyst."""
        )

        return {
            **state,
            "messages": [message],
            "next_agent": "analyst"
        }

    def analyst_node(self, state: AgentState) -> AgentState:
        """Analyst Agent node - analyzes grid and responds to queries"""
        llm_with_tools = self._create_llm_with_tools(analyst_tools)

        grid_state = state.get("grid_state", {})
        messages = state.get("messages", [])

        # System prompt for analyst
        system_msg = SystemMessage(content="""You are the AnalystAgent.
Your job:
1. Analyze the grid state and provide insights WITHOUT using tools
2. Identify critical issues, risks, and trends from the data provided
3. Focus on: supply-demand imbalances, high-risk nodes, storage levels
4. Provide concise, actionable analysis in natural language

DO NOT call tools - just analyze the data given to you.""")

        # Add grid state context - provide actual data
        kpis = grid_state.get("kpis", {})
        nodes_sample = list(grid_state.get("nodes", {}).items())[:10]
        context_msg = HumanMessage(content=f"""Analyze this grid state:

City-wide Metrics:
- Total Demand: {kpis.get('city_demand_mw', 0):.1f} MW
- Total Supply: {kpis.get('city_supply_mw', 0):.1f} MW
- Deficit: {kpis.get('city_demand_mw', 0) - kpis.get('city_supply_mw', 0):.1f} MW
- Avg Risk: {kpis.get('avg_overload_risk', 0):.3f}
- Fairness: {kpis.get('fairness_index', 0):.3f}

Sample Nodes (first 10):
{self._format_nodes_for_display(nodes_sample)}

Provide a brief analysis identifying the top 3-5 critical issues.""")

        # Call LLM with tools
        response = llm_with_tools.invoke([system_msg] + messages + [context_msg])

        # Extract analysis results
        analyst_output = {
            "analysis_text": response.content,
            "timestamp": grid_state.get("timestamp")
        }

        return {
            **state,
            "messages": [AIMessage(content=response.content, name="Analyst")],
            "analyst_output": analyst_output,
            "next_agent": "planner"
        }

    def _format_nodes_for_display(self, nodes_list):
        """Format node data for display"""
        output = []
        for node_id, node_data in nodes_list:
            demand = node_data.get("demand_mw", 0)
            supply = node_data.get("supply_mw", 0)
            balance = supply - demand
            risk = node_data.get("risk", {}).get("overload", 0)
            soc = node_data.get("storage", {}).get("soc", 0)
            output.append(f"  {node_id}: Demand={demand:.1f}MW, Supply={supply:.1f}MW, Balance={balance:+.1f}MW, Risk={risk:.2f}, SOC={soc:.2f}")
        return "\n".join(output)

    def planner_node(self, state: AgentState) -> AgentState:
        """Planner Agent node - creates plans and queries analyst/twin"""
        llm_with_tools = self._create_llm_with_tools(planner_tools)

        messages = state.get("messages", [])
        analyst_output = state.get("analyst_output", {})
        grid_state = state.get("grid_state", {})

        system_msg = SystemMessage(content="""You are the PlannerAgent.
Your job:
1. Review AnalystAgent's findings
2. Create a specific action plan for the top 3-5 critical nodes
3. For each node, specify: increase_supply, discharge_storage, or reduce_demand
4. Prioritize equity-weighted zones and minimize costs

DO NOT use tools - just create a concrete action plan based on the analysis.""")

        context_msg = HumanMessage(content=f"""Analyst Report:
{analyst_output.get('analysis_text')}

Create a specific action plan with:
- Node ID
- Action type (increase_supply, discharge_storage, reduce_demand)
- Target MW adjustment
- Rationale

Focus on the 3-5 most critical nodes.""")

        response = llm_with_tools.invoke([system_msg] + messages + [context_msg])

        planner_output = {
            "plan_text": response.content,
            "timestamp": grid_state.get("timestamp"),
            "actions": []  # Could parse from response text
        }

        return {
            **state,
            "messages": [AIMessage(content=response.content, name="Planner")],
            "planner_output": planner_output,
            "next_agent": "actuator"
        }

    def actuator_node(self, state: AgentState) -> AgentState:
        """Actuator Agent node - executes plan and can query planner"""
        llm_with_tools = self._create_llm_with_tools(actuator_tools)

        messages = state.get("messages", [])
        planner_output = state.get("planner_output", {})
        grid_state = state.get("grid_state", {})

        system_msg = SystemMessage(content="""You are the ActuatorAgent.
Your job:
1. Review Planner's action plan
2. Translate each action into specific HOW instructions
3. Specify equipment, timing, and sequence for each action
4. Explain safety checks and constraints

DO NOT use tools - just provide detailed execution instructions for how to implement the plan.""")

        context_msg = HumanMessage(content=f"""Plan to execute:
{planner_output.get('plan_text')}

For each action in the plan, specify:
- Which equipment to use (battery, solar, gas turbine, demand response)
- Sequence/order of operations
- Ramp rates and timing
- Safety checks

Provide concrete HOW-TO execution steps.""")

        response = llm_with_tools.invoke([system_msg] + messages + [context_msg])

        actuator_output = {
            "execution_text": response.content,
            "commands": [],  # Could parse from response text
            "timestamp": grid_state.get("timestamp")
        }

        return {
            **state,
            "messages": [AIMessage(content=response.content, name="Actuator")],
            "actuator_output": actuator_output,
            "next_agent": "end"
        }

    def should_continue(self, state: AgentState) -> str:
        """Router function to determine next agent"""
        next_agent = state.get("next_agent", "end")

        if next_agent == "end":
            return END

        return next_agent

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("digital_twin", self.digital_twin_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("actuator", self.actuator_node)

        # Define edges (agent-to-agent communication paths)
        workflow.set_entry_point("digital_twin")

        # Main flow
        workflow.add_conditional_edges(
            "digital_twin",
            self.should_continue,
            {
                "analyst": "analyst",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "analyst",
            self.should_continue,
            {
                "planner": "planner",
                "analyst": "analyst",  # Can loop back for queries
                END: END
            }
        )

        workflow.add_conditional_edges(
            "planner",
            self.should_continue,
            {
                "actuator": "actuator",
                "analyst": "analyst",  # Can query analyst
                "digital_twin": "digital_twin",  # Can query twin
                "planner": "planner",  # Can loop for simulation
                END: END
            }
        )

        workflow.add_conditional_edges(
            "actuator",
            self.should_continue,
            {
                "planner": "planner",  # Can ask for feedback
                "actuator": "actuator",  # Can loop for multi-step execution
                END: END
            }
        )

        return workflow.compile()

    def run(self, initial_grid_state: dict) -> dict:
        """Run the multi-agent system for one grid optimization cycle"""
        initial_state = AgentState(
            messages=[],
            grid_state=initial_grid_state,
            analyst_output={},
            planner_output={},
            actuator_output={},
            next_agent="digital_twin"
        )

        # Execute the graph
        final_state = self.graph.invoke(initial_state)

        return {
            "conversation_history": final_state.get("messages", []),
            "analyst_output": final_state.get("analyst_output", {}),
            "planner_output": final_state.get("planner_output", {}),
            "actuator_output": final_state.get("actuator_output", {}),
            "grid_state": final_state.get("grid_state", {})
        }


if __name__ == "__main__":
    print("=" * 60)
    print("SynErgi Multi-Agent System (LangGraph)")
    print("=" * 60)

    # Load initial grid state
    from data.simulator import GridSimulator

    sim = GridSimulator()
    grid_state = sim.generate_tick()

    # Initialize multi-agent system
    mas = SynErgiMultiAgentSystem()

    print("\nRunning multi-agent optimization cycle...\n")

    # Run the system
    result = mas.run(grid_state)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAnalyst: {result['analyst_output'].get('analysis_text', 'N/A')[:200]}...")
    print(f"\nPlanner: {result['planner_output'].get('plan_text', 'N/A')[:200]}...")
    print(f"\nActuator: {result['actuator_output'].get('execution_text', 'N/A')[:200]}...")
