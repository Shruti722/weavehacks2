"""
DigitalTwinEnv: Simulates the power grid environment
"""

import copy
from datetime import datetime, timedelta


class DigitalTwinEnv:
    """
    Digital Twin simulator for the power grid.

    Simulates:
    - Grid state evolution over time
    - Supply/demand dynamics
    - Storage charging/discharging
    - Environmental drivers (temperature, pricing)
    - Risk propagation
    """

    def __init__(self, initial_state=None):
        """
        Initialize the Digital Twin environment.

        Args:
            initial_state (dict, optional): Initial grid state. Uses default if None.
        """
        self.state = initial_state or self._get_default_state()
        self.history = []
        self.tick_count = 0

    def _get_default_state(self):
        """Generate a default initial grid state for San Francisco"""
        return {
            "timestamp": datetime.now().isoformat(),
            "drivers": {
                "temperature": 18.0,
                "price_per_kwh": 0.25
            },
            "nodes": [
                {
                    "id": "SF_Downtown",
                    "demand_mw": 42.3,
                    "supply_mw": 38.1,
                    "storage_level": 0.65,
                    "storage_capacity_mwh": 50.0,
                    "risk_index": 0.42
                },
                {
                    "id": "SF_Sunset",
                    "demand_mw": 31.5,
                    "supply_mw": 34.0,
                    "storage_level": 0.51,
                    "storage_capacity_mwh": 40.0,
                    "risk_index": 0.21
                },
                {
                    "id": "SF_Mission",
                    "demand_mw": 28.7,
                    "supply_mw": 29.2,
                    "storage_level": 0.73,
                    "storage_capacity_mwh": 35.0,
                    "risk_index": 0.15
                }
            ],
            "total_cost": 12100,
            "fairness_index": 0.88
        }

    def get_state(self):
        """Return the current grid state"""
        return copy.deepcopy(self.state)

    def step(self, commands):
        """
        Execute one simulation step with the given commands.

        Args:
            commands (list): List of control commands from ActuatorAgent

        Returns:
            dict: New grid state after executing commands
        """
        # TODO: Add realistic physics simulation
        # TODO: Add noise and stochastic events

        # Save current state to history
        self.history.append(copy.deepcopy(self.state))

        # Apply commands to each node
        self._apply_commands(commands)

        # Simulate natural evolution (demand changes, etc.)
        self._simulate_evolution()

        # Update aggregate metrics
        self._update_metrics()

        # Increment tick counter
        self.tick_count += 1

        # Update timestamp
        current_time = datetime.fromisoformat(self.state["timestamp"])
        new_time = current_time + timedelta(minutes=5)
        self.state["timestamp"] = new_time.isoformat()

        return self.get_state()

    def _apply_commands(self, commands):
        """Apply control commands to grid nodes"""
        for cmd in commands:
            node_id = cmd.get("node_id")
            controls = cmd.get("controls", {})

            # Find the node
            node = None
            for n in self.state["nodes"]:
                if n["id"] == node_id:
                    node = n
                    break

            if node is None:
                continue

            # Apply supply adjustments
            if "supply_adjustment_mw" in controls:
                node["supply_mw"] += controls["supply_adjustment_mw"]
                node["supply_mw"] = max(0, node["supply_mw"])  # Can't be negative

            # Apply storage charging
            if "storage_charge_mw" in controls:
                charge_amount = controls["storage_charge_mw"] / 12  # Convert MW to MWh (5 min = 1/12 hour)
                capacity = node.get("storage_capacity_mwh", 50.0)
                current_level = node["storage_level"]

                # Update storage level
                new_level = min(1.0, current_level + (charge_amount / capacity))
                node["storage_level"] = new_level

            # Apply storage discharging
            if "storage_discharge_mw" in controls:
                discharge_amount = controls["storage_discharge_mw"] / 12
                capacity = node.get("storage_capacity_mwh", 50.0)
                current_level = node["storage_level"]

                # Update storage level
                new_level = max(0.0, current_level - (discharge_amount / capacity))
                node["storage_level"] = new_level

                # Add discharged energy to supply
                actual_discharge = (current_level - new_level) * capacity
                node["supply_mw"] += actual_discharge * 12  # Convert back to MW

    def _simulate_evolution(self):
        """Simulate natural grid evolution (demand changes, etc.)"""
        import random

        # Update environmental drivers
        temp_change = random.uniform(-1.0, 1.0)
        self.state["drivers"]["temperature"] += temp_change
        self.state["drivers"]["temperature"] = max(0, min(40, self.state["drivers"]["temperature"]))

        # Slight price fluctuation
        price_change = random.uniform(-0.02, 0.02)
        self.state["drivers"]["price_per_kwh"] += price_change
        self.state["drivers"]["price_per_kwh"] = max(0.1, min(0.5, self.state["drivers"]["price_per_kwh"]))

        # Update demand for each node based on temperature and randomness
        for node in self.state["nodes"]:
            # Demand increases with extreme temperatures
            temp = self.state["drivers"]["temperature"]
            temp_factor = 1.0 + 0.01 * abs(temp - 20)  # +1% per degree from comfort zone

            # Random variation
            random_factor = random.uniform(0.95, 1.05)

            node["demand_mw"] *= temp_factor * random_factor

            # Update risk index based on supply-demand balance
            imbalance = abs(node["supply_mw"] - node["demand_mw"]) / max(node["demand_mw"], 1.0)
            node["risk_index"] = min(1.0, imbalance * 0.5 + random.uniform(0, 0.2))

    def _update_metrics(self):
        """Update aggregate grid metrics"""
        total_cost = 0
        fairness_scores = []

        for node in self.state["nodes"]:
            # Calculate cost based on supply and price
            node_cost = node["supply_mw"] * self.state["drivers"]["price_per_kwh"] * (5/60)  # 5-minute interval
            total_cost += node_cost

            # Calculate fairness score (how well demand is met)
            supply_ratio = node["supply_mw"] / max(node["demand_mw"], 1.0)
            fairness_scores.append(min(supply_ratio, 1.0))

        self.state["total_cost"] = total_cost

        # Fairness index: how uniform is service quality across nodes
        if fairness_scores:
            avg_fairness = sum(fairness_scores) / len(fairness_scores)
            variance = sum((s - avg_fairness) ** 2 for s in fairness_scores) / len(fairness_scores)
            self.state["fairness_index"] = max(0, 1.0 - variance)
        else:
            self.state["fairness_index"] = 1.0

    def reset(self):
        """Reset the environment to initial state"""
        self.state = self._get_default_state()
        self.history = []
        self.tick_count = 0
        return self.get_state()
