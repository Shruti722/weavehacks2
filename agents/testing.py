"""
Test script for SynErgi Multi-Agent System
Takes a user prompt and runs through the full agent workflow
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent_system import SynErgiMultiAgentSystem
from data.simulator import GridSimulator


def test_multi_agent_workflow(user_prompt: str = None):
    """
    Test the multi-agent system with a user prompt

    Args:
        user_prompt: User's request/question for the grid system
    """
    print("=" * 70)
    print("SynErgi Multi-Agent System - Test")
    print("=" * 70)

    # Default prompt if none provided
    if user_prompt is None:
        user_prompt = "Analyze the current grid state and optimize for equity and cost efficiency"

    print(f"\n📝 User Prompt: {user_prompt}\n")

    # Step 1: Generate realistic grid state
    print("🔄 Generating grid state data...")
    sim = GridSimulator()
    grid_state = sim.generate_tick()

    # Add user prompt to grid state
    grid_state["user_prompt"] = user_prompt

    # Print grid summary
    kpis = grid_state.get("kpis", {})
    weather = grid_state.get("drivers", {}).get("weather", {})
    print(f"   Time: {grid_state.get('drivers', {}).get('time_of_day', 'N/A')}")
    print(f"   Demand: {kpis.get('city_demand_mw', 0):.1f} MW")
    print(f"   Supply: {kpis.get('city_supply_mw', 0):.1f} MW")
    print(f"   Weather: {weather.get('temp_c', 0)}°C, Solar: {weather.get('solar_irradiance_wm2', 0)} W/m²")
    print(f"   Risk: {kpis.get('avg_overload_risk', 0):.3f}")

    # Step 2: Initialize multi-agent system
    print("\n🤖 Initializing Multi-Agent System...")
    try:
        mas = SynErgiMultiAgentSystem()
        print("   ✓ System initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return None

    # Step 3: Run the workflow
    print("\n⚙️  Running Agent Workflow...")
    print("   Digital Twin → Analyst → Planner → Actuator")
    print()

    try:
        result = mas.run(grid_state)

        # Step 4: Display results
        print("\n" + "=" * 70)
        print("📊 RESULTS")
        print("=" * 70)

        # Show initial vs final grid state
        initial_kpis = grid_state.get("kpis", {})
        final_grid = result.get("grid_state", {})
        final_kpis = final_grid.get("kpis", {})

        print("\n⚡ GRID STATE COMPARISON:")
        print("-" * 70)
        print(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change':<15}")
        print("-" * 70)

        demand_before = initial_kpis.get('city_demand_mw', 0)
        demand_after = final_kpis.get('city_demand_mw', 0)
        supply_before = initial_kpis.get('city_supply_mw', 0)
        supply_after = final_kpis.get('city_supply_mw', 0)
        deficit_before = demand_before - supply_before
        deficit_after = demand_after - supply_after

        print(f"{'Total Demand (MW)':<30} {demand_before:<15.1f} {demand_after:<15.1f} {demand_after - demand_before:+.1f}")
        print(f"{'Total Supply (MW)':<30} {supply_before:<15.1f} {supply_after:<15.1f} {supply_after - supply_before:+.1f}")
        print(f"{'Deficit (MW)':<30} {deficit_before:<15.1f} {deficit_after:<15.1f} {deficit_after - deficit_before:+.1f}")
        print(f"{'Avg Risk':<30} {initial_kpis.get('avg_overload_risk', 0):<15.3f} {final_kpis.get('avg_overload_risk', 0):<15.3f} {final_kpis.get('avg_overload_risk', 0) - initial_kpis.get('avg_overload_risk', 0):+.3f}")

        # Analyst output
        analyst_output = result.get("analyst_output", {})
        print("\n🔍 ANALYST ANALYSIS:")
        print("-" * 70)
        analyst_text = analyst_output.get("analysis_text", "No analysis available")
        print(analyst_text[:500] + ("..." if len(analyst_text) > 500 else ""))

        # Planner output
        planner_output = result.get("planner_output", {})
        print("\n📋 PLANNER ACTION PLAN:")
        print("-" * 70)
        plan_text = planner_output.get("plan_text", "No plan available")
        print(plan_text[:500] + ("..." if len(plan_text) > 500 else ""))

        # Actuator output
        actuator_output = result.get("actuator_output", {})
        print("\n⚡ ACTUATOR EXECUTION:")
        print("-" * 70)
        execution_text = actuator_output.get("execution_text", "No execution details")
        print(execution_text[:500] + ("..." if len(execution_text) > 500 else ""))

        # Conversation history - FULL TRANSCRIPT
        print("\n💬 FULL AGENT CONVERSATION:")
        print("=" * 70)
        messages = result.get("conversation_history", [])
        print(f"Total messages exchanged: {len(messages)}\n")

        for i, msg in enumerate(messages, 1):
            agent_name = getattr(msg, 'name', 'System')
            msg_type = type(msg).__name__
            content = msg.content if hasattr(msg, 'content') else str(msg)

            # Format each message with clear separation
            print(f"\n{'='*70}")
            print(f"Message {i}: {agent_name} ({msg_type})")
            print(f"{'='*70}")
            print(content)
            print()

        print("\n" + "=" * 70)
        print("✅ Test Complete!")
        print("=" * 70)

        return result

    except Exception as e:
        print(f"\n❌ Error during workflow execution:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Allow custom prompt from command line
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        # Use default test prompt
        user_prompt = "Analyze the grid and create a plan that prioritizes equity-weighted zones while minimizing costs"

    # Run the test
    result = test_multi_agent_workflow(user_prompt)

    if result is None:
        sys.exit(1)