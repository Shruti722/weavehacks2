"""
AnalystAgent Tools - Enhanced for Deep Reasoning
Tools that provide INSIGHTS, not just data processing
"""

import json
from typing import Dict, List


def identify_spatial_clusters(nodes: Dict) -> Dict:
    """
    Identify if problem nodes are spatially clustered (suggests transmission issues)
    vs scattered (suggests capacity issues).

    Returns insights about root causes.
    """
    # Group nodes by type and check for clustering
    problem_nodes = {
        k: v for k, v in nodes.items()
        if v.get("demand_mw", 0) > v.get("supply_mw", 0)
    }

    # Simple clustering heuristic: check if nodes share prefixes
    clusters = {}
    for node_id in problem_nodes.keys():
        parts = node_id.split("_")
        area = parts[0] if parts else node_id
        if area not in clusters:
            clusters[area] = []
        clusters[area].append(node_id)

    # Find largest cluster
    largest_cluster = max(clusters.items(), key=lambda x: len(x[1])) if clusters else (None, [])

    total_deficit = sum(
        v.get("demand_mw", 0) - v.get("supply_mw", 0)
        for v in problem_nodes.values()
    )

    return {
        "problem_node_count": len(problem_nodes),
        "total_deficit_mw": round(total_deficit, 1),
        "largest_cluster": {
            "area": largest_cluster[0],
            "node_count": len(largest_cluster[1]),
            "nodes": largest_cluster[1]
        },
        "clustering_score": len(largest_cluster[1]) / len(problem_nodes) if problem_nodes else 0,
        "insight": (
            f"HIGH CLUSTERING: {len(largest_cluster[1])}/{len(problem_nodes)} problem nodes in '{largest_cluster[0]}' area. "
            f"This suggests a TRANSMISSION BOTTLENECK, not individual capacity issues."
            if largest_cluster[1] and len(largest_cluster[1]) >= 3
            else f"LOW CLUSTERING: Problem nodes scattered across city. This suggests CAPACITY SHORTAGE, not transmission issues."
        )
    }


def analyze_storage_strategy(nodes: Dict) -> Dict:
    """
    Analyze if storage can be used, or if it's already depleted.
    Provides strategic recommendations.
    """
    soc_levels = [
        (k, v.get("storage", {}).get("soc", 0))
        for k, v in nodes.items()
    ]

    avg_soc = sum(s for _, s in soc_levels) / len(soc_levels) if soc_levels else 0
    low_soc_nodes = [(k, s) for k, s in soc_levels if s < 0.2]
    high_soc_nodes = [(k, s) for k, s in soc_levels if s > 0.5]

    return {
        "avg_soc": round(avg_soc, 3),
        "low_soc_count": len(low_soc_nodes),
        "high_soc_count": len(high_soc_nodes),
        "can_discharge": len(high_soc_nodes) > 0,
        "storage_crisis": len(low_soc_nodes) > len(nodes) * 0.5,
        "insight": (
            f"STORAGE CRISIS: {len(low_soc_nodes)}/{len(nodes)} nodes below 20% SOC. "
            f"Battery discharge is NOT viable. Must use demand response or increase generation."
            if len(low_soc_nodes) > len(nodes) * 0.5
            else f"STORAGE AVAILABLE: {len(high_soc_nodes)} nodes with SOC > 50%. "
                 f"Can discharge {len(high_soc_nodes) * 2}-{len(high_soc_nodes) * 5} MW safely."
        ),
        "strategic_recommendation": (
            "Avoid battery discharge. Focus on demand reduction and supply increase."
            if len(low_soc_nodes) > len(nodes) * 0.5
            else f"Discharge from these {min(5, len(high_soc_nodes))} nodes: {', '.join([k for k, _ in high_soc_nodes[:5]])}"
        )
    }


def compare_to_baseline(current_state: Dict, time_of_day: str) -> Dict:
    """
    Compare current state to expected baseline for this time.
    Identifies if situation is normal or anomalous.
    """
    # Expected demand by time of day (simplified)
    baselines = {
        "morning": {"demand_range": (800, 1200), "deficit_tolerance": 50},
        "midday": {"demand_range": (1200, 1600), "deficit_tolerance": 80},
        "evening": {"demand_range": (1400, 1800), "deficit_tolerance": 100},
        "night": {"demand_range": (600, 1000), "deficit_tolerance": 30}
    }

    # Determine time category
    hour = int(time_of_day.split(":")[0]) if ":" in time_of_day else 12
    if 6 <= hour < 10:
        category = "morning"
    elif 10 <= hour < 16:
        category = "midday"
    elif 16 <= hour < 22:
        category = "evening"
    else:
        category = "night"

    baseline = baselines[category]
    kpis = current_state.get("kpis", {})
    current_demand = kpis.get("city_demand_mw", 0)
    current_supply = kpis.get("city_supply_mw", 0)
    deficit = current_demand - current_supply

    is_anomalous = (
        current_demand < baseline["demand_range"][0] * 0.8 or
        current_demand > baseline["demand_range"][1] * 1.2 or
        deficit > baseline["deficit_tolerance"]
    )

    return {
        "time_category": category,
        "expected_demand_range": baseline["demand_range"],
        "actual_demand": round(current_demand, 1),
        "is_anomalous": is_anomalous,
        "deficit_vs_tolerance": round(deficit - baseline["deficit_tolerance"], 1),
        "insight": (
            f"ANOMALOUS SITUATION: Current deficit of {deficit:.1f} MW exceeds {category} tolerance of {baseline['deficit_tolerance']} MW. "
            f"This is NOT typical for {category}. Investigate potential outages or demand spikes."
            if is_anomalous
            else f"NORMAL OPERATION: {deficit:.1f} MW deficit is within expected range for {category} ({baseline['deficit_tolerance']} MW tolerance)."
        )
    }


def assess_cascading_failure_risk(nodes: Dict) -> Dict:
    """
    Assess if multiple high-risk nodes could trigger cascading failure.
    Provides urgency assessment.
    """
    high_risk_nodes = [
        (k, v.get("risk", {}).get("overload", 0))
        for k, v in nodes.items()
        if v.get("risk", {}).get("overload", 0) > 0.3
    ]

    critical_nodes = [k for k, r in high_risk_nodes if r > 0.5]
    adjacent_risk = len(high_risk_nodes) > 5  # Simple heuristic

    return {
        "high_risk_count": len(high_risk_nodes),
        "critical_count": len(critical_nodes),
        "cascading_risk_level": (
            "CRITICAL" if len(critical_nodes) > 2 and adjacent_risk
            else "HIGH" if len(high_risk_nodes) > 5
            else "MEDIUM" if len(high_risk_nodes) > 2
            else "LOW"
        ),
        "critical_nodes": critical_nodes,
        "insight": (
            f"CASCADING FAILURE RISK: {len(critical_nodes)} nodes at CRITICAL risk (>50%). "
            f"If one fails, others may follow. IMMEDIATE action required on: {', '.join(critical_nodes[:3])}"
            if len(critical_nodes) > 2
            else f"ISOLATED RISKS: {len(high_risk_nodes)} high-risk nodes, but they're independent. "
                 f"Can address them sequentially without urgency."
        )
    }


# Legacy tools kept for backwards compatibility
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
