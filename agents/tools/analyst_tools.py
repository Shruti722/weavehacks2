"""
AnalystAgent Tools
Tools for analyzing grid state, forecasting load, and detecting anomalies
"""

import json
from typing import Dict, List


def load_forecast(nodes: List[Dict], forecast_horizon_min: int = 30) -> Dict:
    """
    Forecast how much energy should be sent to each node.

    Args:
        nodes: List of node data with current demand/supply
        forecast_horizon_min: How many minutes ahead to forecast

    Returns:
        Dict with forecasted energy allocation per node
    """
    forecast = {
        "forecast_horizon_min": forecast_horizon_min,
        "timestamp": "projected",
        "allocations": []
    }

    for node in nodes:
        node_id = node.get("id") if isinstance(node, dict) and "id" in node else "unknown"
        current_demand = node.get("demand_mw", 0)
        current_supply = node.get("supply_mw", 0)
        storage_soc = node.get("storage", {}).get("soc", 0) if isinstance(node.get("storage"), dict) else 0

        # Simple forecasting logic (will be enhanced with ML/historical data)
        # Assume 5% demand growth in next 30 min during peak hours
        forecasted_demand = current_demand * 1.05

        # Calculate needed allocation
        deficit = forecasted_demand - current_supply

        allocation = {
            "node_id": node_id,
            "current_demand_mw": current_demand,
            "forecasted_demand_mw": round(forecasted_demand, 2),
            "current_supply_mw": current_supply,
            "needed_allocation_mw": round(max(0, deficit), 2),
            "priority": "high" if deficit > 5 else "medium" if deficit > 0 else "low",
            "storage_available": storage_soc > 0.3
        }

        forecast["allocations"].append(allocation)

    # Sort by priority
    forecast["allocations"].sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])

    return forecast


def risk_scan(nodes: List[Dict], threshold: float = 0.4) -> Dict:
    """
    Scan grid for risk indicators (overload, low margins, storage issues).

    Args:
        nodes: List of node data
        threshold: Risk threshold (0-1), nodes above this are flagged

    Returns:
        Dict with risk assessment per node
    """
    risk_report = {
        "scan_timestamp": "current",
        "threshold": threshold,
        "high_risk_nodes": [],
        "medium_risk_nodes": [],
        "low_risk_nodes": [],
        "critical_count": 0
    }

    for node in nodes:
        node_id = node.get("id") if isinstance(node, dict) and "id" in node else "unknown"
        risk_data = node.get("risk", {}) if isinstance(node.get("risk"), dict) else {}
        overload_risk = risk_data.get("overload", 0)
        n1_margin = risk_data.get("n_1_margin", 1.0)

        demand = node.get("demand_mw", 0)
        supply = node.get("supply_mw", 0)
        imbalance = abs(supply - demand)

        storage = node.get("storage", {}) if isinstance(node.get("storage"), dict) else {}
        soc = storage.get("soc", 0)

        # Calculate composite risk score
        risk_factors = {
            "overload_risk": overload_risk,
            "low_n1_margin": 1.0 - n1_margin,
            "supply_demand_imbalance": min(1.0, imbalance / max(demand, 1.0)),
            "low_storage": 1.0 - soc if soc < 0.3 else 0
        }

        composite_risk = (
            risk_factors["overload_risk"] * 0.4 +
            risk_factors["low_n1_margin"] * 0.3 +
            risk_factors["supply_demand_imbalance"] * 0.2 +
            risk_factors["low_storage"] * 0.1
        )

        risk_entry = {
            "node_id": node_id,
            "composite_risk": round(composite_risk, 3),
            "risk_factors": risk_factors,
            "status": "critical" if composite_risk > 0.7 else "high" if composite_risk > threshold else "medium" if composite_risk > 0.2 else "low"
        }

        if risk_entry["status"] == "critical":
            risk_report["critical_count"] += 1
            risk_report["high_risk_nodes"].append(risk_entry)
        elif risk_entry["status"] == "high":
            risk_report["high_risk_nodes"].append(risk_entry)
        elif risk_entry["status"] == "medium":
            risk_report["medium_risk_nodes"].append(risk_entry)
        else:
            risk_report["low_risk_nodes"].append(risk_entry)

    return risk_report


def anomaly_detection(nodes: List[Dict], historical_baseline: Dict = None) -> Dict:
    """
    Detect anomalies in grid behavior (unusual demand spikes, failures, etc.).

    Args:
        nodes: List of current node data
        historical_baseline: Historical baseline data for comparison (optional)

    Returns:
        Dict with detected anomalies
    """
    anomalies = {
        "detection_timestamp": "current",
        "anomalies_found": [],
        "anomaly_count": 0
    }

    # Default baselines if not provided (typical SF grid values)
    if historical_baseline is None:
        historical_baseline = {
            "typical_demand_range": (20, 80),  # MW
            "typical_risk_range": (0.05, 0.3),
            "typical_soc_range": (0.4, 0.8)
        }

    for node in nodes:
        node_id = node.get("id") if isinstance(node, dict) and "id" in node else "unknown"
        demand = node.get("demand_mw", 0)
        supply = node.get("supply_mw", 0)

        risk_data = node.get("risk", {}) if isinstance(node.get("risk"), dict) else {}
        risk = risk_data.get("overload", 0)

        storage = node.get("storage", {}) if isinstance(node.get("storage"), dict) else {}
        soc = storage.get("soc", 0)

        node_anomalies = []

        # Check for demand anomalies
        typical_demand_min, typical_demand_max = historical_baseline["typical_demand_range"]
        if demand > typical_demand_max * 1.5:
            node_anomalies.append({
                "type": "demand_spike",
                "severity": "high",
                "value": demand,
                "expected_range": f"{typical_demand_min}-{typical_demand_max} MW",
                "description": f"Demand spike: {demand:.1f}MW (>{typical_demand_max*1.5:.1f}MW threshold)"
            })
        elif demand < typical_demand_min * 0.5:
            node_anomalies.append({
                "type": "demand_drop",
                "severity": "medium",
                "value": demand,
                "expected_range": f"{typical_demand_min}-{typical_demand_max} MW",
                "description": f"Demand drop: {demand:.1f}MW (<{typical_demand_min*0.5:.1f}MW threshold)"
            })

        # Check for supply-demand mismatch
        imbalance_ratio = abs(supply - demand) / max(demand, 1.0)
        if imbalance_ratio > 0.5:
            node_anomalies.append({
                "type": "supply_demand_mismatch",
                "severity": "high" if imbalance_ratio > 0.7 else "medium",
                "value": imbalance_ratio,
                "description": f"Large supply-demand gap: {abs(supply-demand):.1f}MW ({imbalance_ratio*100:.0f}% imbalance)"
            })

        # Check for abnormal risk
        typical_risk_min, typical_risk_max = historical_baseline["typical_risk_range"]
        if risk > typical_risk_max * 2:
            node_anomalies.append({
                "type": "elevated_risk",
                "severity": "critical",
                "value": risk,
                "expected_range": f"{typical_risk_min}-{typical_risk_max}",
                "description": f"Critically high risk: {risk:.2f} (>{typical_risk_max*2:.2f} threshold)"
            })

        # Check for storage anomalies
        typical_soc_min, typical_soc_max = historical_baseline["typical_soc_range"]
        if soc < typical_soc_min * 0.5 and soc > 0:
            node_anomalies.append({
                "type": "critically_low_storage",
                "severity": "high",
                "value": soc,
                "expected_range": f"{typical_soc_min}-{typical_soc_max}",
                "description": f"Storage critically low: {soc*100:.0f}% SOC"
            })

        # Add to report if anomalies found
        if node_anomalies:
            anomalies["anomalies_found"].append({
                "node_id": node_id,
                "anomalies": node_anomalies,
                "anomaly_count": len(node_anomalies)
            })
            anomalies["anomaly_count"] += len(node_anomalies)

    return anomalies
