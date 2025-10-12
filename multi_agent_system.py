"""
SynErgi Multi-Agent System using LangGraph
Agent-to-agent communication with ReAct pattern
"""

import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
import operator
import weave

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# Import our custom tools
from agents.tools import (
    # Analyst tools - enhanced reasoning
    identify_spatial_clusters, analyze_storage_strategy,
    compare_to_baseline, assess_cascading_failure_risk,
    # Analyst tools - legacy
    load_forecast, risk_scan, anomaly_detection,
    # Planner tools - simplified
    get_top_deficit_nodes,
    # Actuator tools
    charge_battery, discharge_battery, reconfigure_lines, update_grid_twin
)

load_dotenv()

# Initialize Weave for tracing - changed to synergi-rl-training for RL experiments
weave.init(os.getenv("WEAVE_PROJECT", "synergi-rl-training"))


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
def planner_get_top_deficit_nodes(grid_state: dict, limit: int = 5) -> dict:
    """Get nodes with biggest deficits - these need help most"""
    return get_top_deficit_nodes(grid_state, limit)


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
    # Enhanced reasoning tools (NEW - prioritized)
    identify_spatial_clusters,
    analyze_storage_strategy,
    compare_to_baseline,
    assess_cascading_failure_risk,
    # Legacy tools
    query_grid_state,
    analyst_load_forecast,
    analyst_risk_scan,
    analyst_anomaly_detection
]

planner_tools = [
    planner_get_top_deficit_nodes
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

    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct", max_turns: int = 10):
        """
        Initialize multi-agent system with W&B Inference.

        Default: Qwen2.5 14B Instruct
        - 14B parameters (good for RL fine-tuning)
        - Strong reasoning capabilities
        - Supports tool use and multi-turn conversations

        Other options:
        - meta-llama/Llama-3.1-8B-Instruct (8B, lighter)
        - microsoft/Phi-4-mini-instruct (3.8B, very lightweight)

        Args:
            max_turns: Maximum conversation turns before forcing end (prevents infinite loops)
        """
        self.model_name = model_name
        self.max_turns = max_turns

        # Initialize grid simulator for RL reward calculation
        try:
            import sys
            sys.path.append('data')
            from simulator import GridSimulator
            self.grid_simulator = GridSimulator("data/data.json")
            print("✓ Grid simulator initialized for RL reward calculation")
        except Exception as e:
            print(f"⚠️  Grid simulator not available: {e}")
            self.grid_simulator = None
        self.graph = self._build_graph()

    def _create_llm_with_tools(self, tools, temperature: float = None, top_p: float = None):
        """
        Create LLM instance with tools bound
        Routes through OpenAI-compatible endpoints (OpenPipe or W&B)

        Args:
            tools: LangChain tools to bind
            temperature: Sampling temperature (default from env or 0.3)
            top_p: Nucleus sampling parameter (default from env or 0.9)
        """
        from langchain_openai import ChatOpenAI

        # Priority: OpenPipe (for RL training) > W&B (for inference)
        openpipe_key = os.getenv("OPENPIPE_API_KEY")
        wandb_key = os.getenv("WANDB_API_KEY")

        # Allow ART to control sampling params via env
        if temperature is None:
            temperature = float(os.getenv("TEMPERATURE", "0.3"))
        if top_p is None:
            top_p = float(os.getenv("TOP_P", "0.9"))

        if openpipe_key and self.model_name.startswith("openpipe:"):
            # OpenPipe for RL training - routes model calls through their endpoint
            print(f"[LLM] Using OpenPipe: {self.model_name}")
            print(f"      Temperature: {temperature}, Top-p: {top_p}")
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=openpipe_key,
                base_url="https://api.openpipe.ai/v1",
                temperature=temperature,
                top_p=top_p,
                max_tokens=4096,
                request_timeout=120
            )
        elif wandb_key:
            # W&B Inference for base models (non-RL)
            print(f"[LLM] Using W&B Inference: {self.model_name}")
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=wandb_key,
                base_url="https://api.inference.wandb.ai/v1",
                temperature=temperature,
                max_tokens=4096,
                request_timeout=120
            )
        else:
            raise ValueError(
                "Missing API key! Set one of:\n"
                "  OPENPIPE_API_KEY - for RL training (https://app.openpipe.ai/settings)\n"
                "  WANDB_API_KEY - for inference (https://wandb.ai/authorize)"
            )

        return llm.bind_tools(tools)

    @weave.op()
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

    @weave.op()
    def analyst_node(self, state: AgentState) -> AgentState:
        """Analyst Agent node - analyzes grid and responds to queries WITH TOOL CALLING"""
        llm_with_tools = self._create_llm_with_tools(analyst_tools)

        grid_state = state.get("grid_state", {})
        messages = state.get("messages", [])
        nodes = grid_state.get("nodes", {})
        drivers = grid_state.get("drivers", {})

        # Check if Planner is asking a question
        last_message = messages[-1] if messages else None
        is_responding_to_planner = (
            last_message and
            hasattr(last_message, 'name') and
            last_message.name == "Planner" and
            "?" in last_message.content
        )

        # System prompt for analyst WITH tool instructions
        system_msg = SystemMessage(content="""You are the AnalystAgent - an expert grid analyst with powerful analysis tools.

IMPORTANT: You have tools available! USE THEM to get insights:
- identify_spatial_clusters(nodes) - Check if problems are clustered or scattered
- analyze_storage_strategy(nodes) - Check if storage can be used
- compare_to_baseline(current_state, time_of_day) - Detect anomalies
- assess_cascading_failure_risk(nodes) - Check urgency

YOUR WORKFLOW:
1. CALL TOOLS to get strategic insights (don't just describe data!)
2. SYNTHESIZE tool outputs into clear recommendations
3. ASK Planner strategic questions based on tool insights

Example of GOOD workflow:
- Call identify_spatial_clusters(nodes) → Get clustering insight
- Call analyze_storage_strategy(nodes) → Get storage viability
- Synthesize: "Tools show HIGH CLUSTERING in financial area + LOW STORAGE. This is a transmission issue, not capacity. Planner: Should we reroute power or reduce demand in financial district?"

Example of BAD workflow:
- Just listing node deficits without calling tools
- Saying "I'll analyze" but not calling any tools

IMPORTANT: CALL THE TOOLS! They return strategic insights you can't get from raw data.""")

        if is_responding_to_planner:
            # Responding to Planner's question
            context_msg = HumanMessage(content=f"""The Planner just asked you: "{last_message.content[-200:]}"

Respond to their specific question based on the grid data. Be direct and concise.""")
        elif len(messages) > 2:
            # Already had conversation, check if we need more analysis
            context_msg = HumanMessage(content="""Review the conversation so far.
If the Planner has a solid plan, say "The plan looks good, proceed with execution."
Otherwise, provide any missing critical information.""")
        else:
            # First analysis - provide SUMMARY, tools will get full details
            kpis = grid_state.get("kpis", {})
            nodes = grid_state.get("nodes", {})
            drivers = grid_state.get("drivers", {})

            # Lightweight summary
            deficit_count = len([n for n in nodes.values() if n.get("demand_mw", 0) > n.get("supply_mw", 0)])
            low_soc_count = len([n for n in nodes.values() if n.get("storage", {}).get("soc", 0) < 0.2])
            time_of_day = drivers.get("time_of_day", "unknown")

            context_msg = HumanMessage(content=f"""Analyze this grid using your TOOLS:

**City Overview:**
- Total Demand: {kpis.get('city_demand_mw', 0):.1f} MW
- Total Supply: {kpis.get('city_supply_mw', 0):.1f} MW
- Deficit: {kpis.get('city_demand_mw', 0) - kpis.get('city_supply_mw', 0):.1f} MW
- Nodes in deficit: {deficit_count}/45
- Low storage nodes: {low_soc_count}/45
- Time: {time_of_day}

**Use your tools to analyze:**
1. Call identify_spatial_clusters(nodes) - are problems clustered?
2. Call analyze_storage_strategy(nodes) - can we use batteries?
3. Call compare_to_baseline(current_state, time_of_day) - is this normal?
4. Call assess_cascading_failure_risk(nodes) - how urgent?

Then synthesize insights and recommend strategy to Planner.""")

        # TOOL EXECUTION LOOP - Keep calling tools until LLM gives final answer
        conversation = [system_msg] + messages + [context_msg]
        max_tool_iterations = 5
        iteration = 0
        all_tool_results = []

        while iteration < max_tool_iterations:
            response = llm_with_tools.invoke(conversation)

            # Check if LLM wants to call tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"[Analyst] Calling {len(response.tool_calls)} tools...")

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']

                    print(f"  - {tool_name}({list(tool_args.keys())})")

                    # Find and execute the tool
                    tool_func = None
                    for tool in analyst_tools:
                        if hasattr(tool, 'name') and tool.name == tool_name:
                            tool_func = tool.func
                            break
                        elif hasattr(tool, '__name__') and tool.__name__ == tool_name:
                            tool_func = tool
                            break

                    if tool_func:
                        try:
                            # Pass grid state data if needed
                            if 'nodes' in tool_args or tool_name == 'identify_spatial_clusters' or tool_name == 'analyze_storage_strategy':
                                tool_args['nodes'] = nodes
                            if 'current_state' in tool_args or tool_name == 'compare_to_baseline':
                                tool_args['current_state'] = grid_state
                                tool_args['time_of_day'] = drivers.get('time_of_day', '12:00')

                            result = tool_func(**tool_args)
                            all_tool_results.append({"tool": tool_name, "result": result})

                            # Extract only key insights to reduce token usage
                            insight_summary = result.get('insight', str(result)[:200])

                            # Add compact tool result to conversation WITHOUT full tool_call args
                            from langchain_core.messages import ToolMessage
                            # Create lightweight tool call reference (without massive args)
                            lightweight_tool_call = {
                                'name': tool_call['name'],
                                'id': tool_call['id'],
                                'args': {}  # Empty args to avoid bloating context
                            }
                            conversation.append(AIMessage(content="", tool_calls=[lightweight_tool_call]))
                            conversation.append(ToolMessage(
                                content=insight_summary,
                                tool_call_id=tool_call['id']
                            ))
                        except Exception as e:
                            print(f"    Error executing {tool_name}: {e}")
                            conversation.append(ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call['id']
                            ))

                iteration += 1
            else:
                # No more tool calls - LLM gave final answer
                break

        # Extract final analysis
        analyst_output = {
            "analysis_text": response.content,
            "tool_results": all_tool_results,
            "timestamp": grid_state.get("timestamp")
        }

        # Check if Analyst is asking a question to Planner
        is_asking_question = "?" in response.content and ("planner" in response.content.lower() or "should" in response.content.lower())

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response.content, name="Analyst")],
            "analyst_output": analyst_output,
            "next_agent": "planner" if not is_asking_question else "planner"
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

    @weave.op()
    def planner_node(self, state: AgentState) -> AgentState:
        """Planner Agent node - creates plans and queries analyst/twin"""
        llm_with_tools = self._create_llm_with_tools(planner_tools)

        messages = state.get("messages", [])
        analyst_output = state.get("analyst_output", {})
        grid_state = state.get("grid_state", {})

        # Check if Analyst asked a question or if Actuator is asking for clarification
        last_message = messages[-1] if messages else None
        is_responding = last_message and hasattr(last_message, 'name') and "?" in last_message.content

        system_msg = SystemMessage(content="""You are the PlannerAgent. Create action plans based on Analyst's recommendations.

YOUR WORKFLOW:
1. Read the Analyst's analysis and acknowledge their key findings
2. Based on their insights, select 3-5 nodes with biggest deficits
3. Create a concrete action plan

RESPONSE FORMAT:
First, briefly acknowledge Analyst's recommendations (1-2 sentences).
Then output your action plan in this EXACT format:

=== ACTION PLAN ===
Node: financial_district
Action Type: increase_supply
Target MW Adjustment: 5.0

Node: russian_hill
Action Type: increase_supply
Target MW Adjustment: 3.0
===================

IMPORTANT:
- First line should acknowledge Analyst (e.g., "Based on Analyst's recommendation to address scattered deficits...")
- Use ACTUAL node IDs from the grid
- Only use "increase_supply" as action type
- MUST include === ACTION PLAN === markers""")

        # Get nodes with deficits
        nodes = grid_state.get("nodes", {})
        deficit_nodes = []
        for node_id, node_data in nodes.items():
            demand = node_data.get("demand_mw", 0)
            supply = node_data.get("supply_mw", 0)
            deficit = demand - supply
            if deficit > 0:
                deficit_nodes.append({
                    "node_id": node_id,
                    "deficit_mw": round(deficit, 1)
                })

        deficit_nodes.sort(key=lambda x: x["deficit_mw"], reverse=True)
        top_3 = deficit_nodes[:3]

        # Get Analyst's key insight
        analyst_text = analyst_output.get('analysis_text', '')
        analyst_summary = analyst_text[:200] if analyst_text else "capacity shortage with scattered deficits"

        # Simple, direct instruction with actual data
        nodes_text = "\n".join([f"- {n['node_id']}: {n['deficit_mw']} MW deficit" for n in top_3])

        context_msg = HumanMessage(content=f"""The Analyst says: "{analyst_summary}..."

Top 3 nodes with deficits:
{nodes_text}

Acknowledge the Analyst's insight, then create your action plan:

Example:
"Based on the Analyst's finding that deficits are scattered, I'll target the top deficit nodes with supply increases."

=== ACTION PLAN ===
Node: {top_3[0]['node_id']}
Action Type: increase_supply
Target MW Adjustment: {top_3[0]['deficit_mw']}

Node: {top_3[1]['node_id']}
Action Type: increase_supply
Target MW Adjustment: {top_3[1]['deficit_mw']}

Node: {top_3[2]['node_id']}
Action Type: increase_supply
Target MW Adjustment: {top_3[2]['deficit_mw']}
===================""")

        # Direct invocation - no tool loop needed since we give it all the data
        conversation = [system_msg] + messages + [context_msg]
        llm_no_tools = self._create_llm_with_tools([])  # No tools - simpler
        response = llm_no_tools.invoke(conversation)

        # If it doesn't have action plan, force it
        if "=== ACTION PLAN ===" not in response.content:
            print("[Planner] No action plan detected, forcing output...")
            conversation.append(AIMessage(content=response.content, name="Planner"))
            conversation.append(HumanMessage(content="You MUST output === ACTION PLAN === with the 3 nodes NOW!"))
            response = llm_no_tools.invoke(conversation)

        planner_output = {
            "plan_text": response.content,
            "tool_results": [],
            "timestamp": grid_state.get("timestamp"),
            "actions": []
        }

        response_content = response.content

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response_content, name="Planner")],
            "planner_output": planner_output,
            "next_agent": "actuator"
        }

    @weave.op()
    def actuator_node(self, state: AgentState) -> AgentState:
        """Actuator Agent node - executes plan and updates grid"""
        messages = state.get("messages", [])
        planner_output = state.get("planner_output", {})
        grid_state = state.get("grid_state", {})

        # Step 1: Parse the plan into structured actions
        actions = self._parse_plan_to_actions(planner_output.get("plan_text", ""))

        # Step 2: Execute actions using tools
        executed_commands = []
        execution_log = []

        for action in actions:
            node_id = action.get("node_id")
            action_type = action.get("action_type")
            target_mw = action.get("target_mw", 0)

            try:
                if action_type == "increase_supply":
                    # Use discharge_battery or assume supply increase
                    result = discharge_battery(node_id, target_mw, duration_min=10)
                    executed_commands.append(result)
                    execution_log.append(f"✓ Increased supply at {node_id} by {target_mw} MW (battery discharge)")

                elif action_type == "discharge_storage":
                    result = discharge_battery(node_id, target_mw, duration_min=5)
                    executed_commands.append(result)
                    execution_log.append(f"✓ Discharged storage at {node_id}: {target_mw} MW")

                elif action_type == "charge_storage":
                    result = charge_battery(node_id, target_mw, duration_min=5)
                    executed_commands.append(result)
                    execution_log.append(f"✓ Charged storage at {node_id}: {target_mw} MW")

                elif action_type == "reduce_demand":
                    # Simulate demand reduction (no specific tool, just log it)
                    executed_commands.append({
                        "action": "reduce_demand",
                        "node_id": node_id,
                        "reduction_mw": target_mw,
                        "method": "demand_response"
                    })
                    execution_log.append(f"✓ Reduced demand at {node_id} by {target_mw} MW (demand response)")

            except Exception as e:
                execution_log.append(f"✗ Failed to execute {action_type} at {node_id}: {str(e)}")

        # Step 3: Update the grid state with executed commands
        updated_grid_state = self._apply_actions_to_grid(grid_state, executed_commands)

        # Step 3.5: Calculate RL reward (before vs after comparison)
        reward_data = None
        if len(actions) > 0 and hasattr(self, 'grid_simulator'):
            try:
                reward_data = self.grid_simulator.calculate_rl_reward(
                    actions,
                    grid_state,  # BEFORE
                    updated_grid_state  # AFTER
                )
                reward_val = reward_data['reward']
                raw_score = reward_data['raw_score']
                print(f"\n💰 RL REWARD: {int(reward_val)} {'✅' if reward_val == 1 else '❌'} (raw: {raw_score:.3f}, threshold: {reward_data['threshold']})")
                print(f"   Cost: ${reward_data['cost_usd']}")
                print(f"   Deficit: {reward_data['deficit_before_mw']:.1f} → {reward_data['deficit_after_mw']:.1f} MW (Δ {reward_data['deficit_improvement_mw']:+.1f})")
                print(f"   Components: deficit={reward_data['deficit_score']:.2f}, cost={reward_data['cost_score']:.2f}, risk={reward_data['risk_score']:.2f}")
            except Exception as e:
                print(f"⚠️  Could not calculate reward: {e}")

        # Step 4: Check if we should ask Planner for clarification
        should_query_planner = len(actions) == 0 or any(a.get("action_type") == "unknown" for a in actions)

        if should_query_planner and len(executed_commands) == 0:
            # Ask Planner for clarification
            execution_text = f"""I reviewed the plan but need clarification:

Plan received: {planner_output.get('plan_text', 'N/A')[:200]}...

Issues:
- Could not parse {len([a for a in actions if a.get('action_type') == 'unknown'])} actions
- Node IDs may not match grid format

Could you please provide:
1. Specific node IDs (e.g., financial_district, not FD)
2. Clear action types: increase_supply, discharge_storage, or reduce_demand
3. Target MW values for each action

Once clarified, I can execute the commands."""

            actuator_output = {
                "execution_text": execution_text,
                "commands": executed_commands,
                "actions_parsed": actions,
                "timestamp": grid_state.get("timestamp"),
                "needs_clarification": True
            }
        else:
            # Create execution summary
            execution_text = "\n".join([
                "**Executed Actions:**",
                *execution_log,
                "",
                f"**Commands executed:** {len(executed_commands)}",
                f"**Grid updated:** {'Yes' if executed_commands else 'No'}",
                "",
                "**Question for Planner:** Did these actions sufficiently address the deficit? Should we continue?"
            ])

            actuator_output = {
                "execution_text": execution_text,
                "commands": executed_commands,
                "actions_parsed": actions,
                "timestamp": grid_state.get("timestamp"),
                "reward_data": reward_data  # Include RL reward
            }

        # Check if Actuator needs clarification from Planner
        needs_clarification = actuator_output.get("needs_clarification", False)

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=execution_text, name="Actuator")],
            "actuator_output": actuator_output,
            "grid_state": updated_grid_state,  # Return updated grid
            "next_agent": "planner" if needs_clarification else "end"
        }

    def _parse_plan_to_actions(self, plan_text: str) -> list:
        """Parse natural language plan into structured actions"""
        import re

        actions = []

        # Look for ACTION PLAN section
        action_plan_match = re.search(r'===\s*ACTION PLAN\s*===(.+?)===', plan_text, re.DOTALL)

        if action_plan_match:
            action_section = action_plan_match.group(1)
        else:
            # Fallback: look for any section with "Node:" patterns
            action_section = plan_text

        # Split by "Node:" to get each action
        node_sections = re.split(r'\n\s*Node:\s*', action_section)

        for section in node_sections:
            if not section.strip():
                continue

            try:
                # Extract node ID (first line after "Node:")
                lines = section.strip().split('\n')
                if not lines:
                    continue

                node_id = lines[0].strip().lower().replace(' ', '_')

                # Extract action type
                action_match = re.search(r'Action Type:\s*([^\n]+)', section, re.IGNORECASE)
                if not action_match:
                    continue

                action_text = action_match.group(1).strip().lower()

                # Map to action types
                if 'increase_supply' in action_text or ('increase' in action_text and 'supply' in action_text):
                    action_type = "increase_supply"
                elif 'discharge_storage' in action_text or 'discharge' in action_text:
                    action_type = "discharge_storage"
                elif 'charge_storage' in action_text or 'charge' in action_text:
                    action_type = "charge_storage"
                elif 'reduce_demand' in action_text or ('reduce' in action_text and 'demand' in action_text):
                    action_type = "reduce_demand"
                else:
                    action_type = "unknown"

                # Extract MW value
                mw_match = re.search(r'Target MW Adjustment:\s*[\+\-]?\s*(\d+(?:\.\d+)?)', section, re.IGNORECASE)
                target_mw = float(mw_match.group(1)) if mw_match else 0

                actions.append({
                    "node_id": node_id,
                    "action_type": action_type,
                    "target_mw": target_mw
                })

            except Exception as e:
                print(f"Warning: Could not parse section: {str(e)}")
                continue

        return actions

    def _apply_actions_to_grid(self, grid_state: dict, commands: list) -> dict:
        """Apply executed commands to update grid state"""
        import copy
        updated_state = copy.deepcopy(grid_state)

        nodes = updated_state.get("nodes", {})

        for cmd in commands:
            node_id = cmd.get("node_id")

            # Find matching node (try exact match first, then fuzzy)
            if node_id not in nodes:
                # Try to find by partial match
                for key in nodes.keys():
                    if node_id in key or key in node_id:
                        node_id = key
                        break

            if node_id not in nodes:
                continue

            node = nodes[node_id]

            # Apply changes based on command type
            cmd_type = cmd.get("command") or cmd.get("action")  # Support both keys

            if cmd_type == "discharge_battery":
                power_mw = cmd.get("power_mw", 0)
                node["supply_mw"] += power_mw  # Increase supply
                node["storage"]["soc"] = max(0, node["storage"]["soc"] - 0.05)  # Decrease SOC

            elif cmd_type == "charge_battery":
                power_mw = cmd.get("power_mw", 0)
                node["supply_mw"] -= power_mw  # Decrease supply (used for charging)
                node["storage"]["soc"] = min(1.0, node["storage"]["soc"] + 0.05)  # Increase SOC

            elif cmd_type == "reduce_demand":
                reduction_mw = cmd.get("reduction_mw", 0)
                node["demand_mw"] -= reduction_mw

            # Recalculate balance
            node["net_load_mw"] = node["demand_mw"] - node["supply_mw"]

        # Recalculate KPIs
        updated_state["kpis"] = self._recalculate_kpis(nodes)

        return updated_state

    def _recalculate_kpis(self, nodes: dict) -> dict:
        """Recalculate city-wide KPIs after actions"""
        total_demand = sum(n["demand_mw"] for n in nodes.values())
        total_supply = sum(n["supply_mw"] for n in nodes.values())
        unserved = max(0, total_demand - total_supply)
        avg_risk = sum(n["risk"]["overload"] for n in nodes.values()) / len(nodes)

        return {
            "city_demand_mw": round(total_demand, 2),
            "city_supply_mw": round(total_supply, 2),
            "unserved_energy_proxy_mw": round(unserved, 2),
            "avg_overload_risk": round(avg_risk, 3),
            "fairness_index": 0.998  # Simplified for now
        }

    def should_continue(self, state: AgentState) -> str:
        """Router function to determine next agent"""
        next_agent = state.get("next_agent", "end")
        messages = state.get("messages", [])

        # Prevent infinite loops - force end after max_turns
        if len(messages) > self.max_turns:
            print(f"[System] Max turns ({self.max_turns}) reached. Ending conversation.")
            return END

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

    @weave.op()
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
