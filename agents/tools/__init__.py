"""
Agent Tools Module - ALL ENHANCED FOR DEEP REASONING
Defines callable tools for each agent with strategic insights
"""

from .analyst_tools import (
    # Enhanced reasoning tools
    identify_spatial_clusters,
    analyze_storage_strategy,
    compare_to_baseline,
    assess_cascading_failure_risk,
    # Legacy tools
    load_forecast,
    risk_scan,
    anomaly_detection
)
from .planner_tools import (
    # Simplified tool
    get_top_deficit_nodes
)
from .actuator_tools import (
    # Enhanced execution tools
    validate_execution_sequence,
    estimate_execution_time,
    identify_execution_risks,
    # Legacy tools
    charge_battery,
    discharge_battery,
    reconfigure_lines,
    update_grid_twin
)

__all__ = [
    # Analyst tools - Enhanced
    "identify_spatial_clusters",
    "analyze_storage_strategy",
    "compare_to_baseline",
    "assess_cascading_failure_risk",
    # Analyst tools - Legacy
    "load_forecast",
    "risk_scan",
    "anomaly_detection",

    # Planner tools - Simplified
    "get_top_deficit_nodes",

    # Actuator tools - Enhanced
    "validate_execution_sequence",
    "estimate_execution_time",
    "identify_execution_risks",
    # Actuator tools - Legacy
    "charge_battery",
    "discharge_battery",
    "reconfigure_lines",
    "update_grid_twin"
]
