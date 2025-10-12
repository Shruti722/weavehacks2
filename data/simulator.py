"""
Real-time Grid Data Simulator - ENHANCED FOR RL
Generates realistic time-series data for SF power grid with:
- Transmission line constraints and power flow
- Renewable generation variability (solar/wind)
- Generator types with different costs and ramp rates
- Battery degradation
- Multi-objective cost function
"""


import json
import random
import math
from datetime import datetime, timedelta
from pathlib import Path


class GridSimulator:
    """
    Simulates realistic power grid data changes over time for San Francisco.

    STATIC (never changes):
    - Node IDs, names, zones
    - Equity weights
    - Storage/flex capacities
    - Network topology (edges)

    DYNAMIC (changes every 5 min):
    - Demand/supply (follows daily patterns + noise)
    - Weather (temp, solar, wind)
    - Prices (day-ahead, real-time)
    - Storage SOC
    - Risk indices
    - Flow on edges
    - DR activation, fatigue
    """

    def __init__(self, template_path="data/data.json"):
        """Initialize simulator with template data"""
        self.template_path = Path(template_path)
        self.base_data = self._load_template()

        # Track simulation time
        self.current_time = datetime.now()
        self.tick_count = 0

        # Track previous values for smooth transitions
        self.prev_demands = {}
        self.prev_supplies = {}
        self.prev_temp = 18.0
        self.prev_solar = 0.0
        self.prev_wind = 3.0

        # Initialize baseline values from neighborhoods
        self._initialize_baselines()

        # RL-READY ENHANCEMENTS
        self._initialize_generators()
        self._initialize_renewables()
        self._initialize_transmission_constraints()
        self._initialize_battery_degradation()

        # Track accumulated costs and emissions for episode
        self.cumulative_cost = 0.0
        self.cumulative_emissions = 0.0
        self.cumulative_unserved_energy = 0.0

    def _load_template(self):
        """Load the template JSON file"""
        with open(self.template_path, 'r') as f:
            return json.load(f)

    def _initialize_baselines(self):
        """Set baseline demand/supply for each SF neighborhood based on characteristics"""
        # SF neighborhood characteristics (MW baselines)
        self.baselines = {
            # Core business districts - high daytime demand
            "financial_district": {"demand": 85.0, "supply": 75.0, "type": "commercial"},
            "south_of_market": {"demand": 70.0, "supply": 65.0, "type": "mixed"},

            # Residential - moderate demand with evening peak
            "mission": {"demand": 45.0, "supply": 42.0, "type": "residential"},
            "castro": {"demand": 35.0, "supply": 33.0, "type": "residential"},
            "haight_ashbury": {"demand": 32.0, "supply": 30.0, "type": "residential"},
            "noe_valley": {"demand": 28.0, "supply": 27.0, "type": "residential"},
            "bernal_heights": {"demand": 26.0, "supply": 25.0, "type": "residential"},

            # High-income residential - higher consumption
            "pacific_heights": {"demand": 38.0, "supply": 36.0, "type": "residential"},
            "marina": {"demand": 35.0, "supply": 33.0, "type": "residential"},
            "russian_hill": {"demand": 32.0, "supply": 31.0, "type": "residential"},
            "nob_hill": {"demand": 30.0, "supply": 29.0, "type": "residential"},

            # Dense urban - moderate-high
            "chinatown": {"demand": 42.0, "supply": 40.0, "type": "mixed"},
            "north_beach": {"demand": 35.0, "supply": 34.0, "type": "mixed"},
            "tenderloin": {"demand": 38.0, "supply": 35.0, "type": "residential"},
            "civic_center": {"demand": 40.0, "supply": 38.0, "type": "mixed"},
            "western_addition": {"demand": 34.0, "supply": 33.0, "type": "residential"},
            "hayes_valley": {"demand": 30.0, "supply": 29.0, "type": "mixed"},
            "lower_pacific_heights": {"demand": 32.0, "supply": 31.0, "type": "residential"},

            # Richmond/Sunset - residential, solar potential
            "inner_richmond": {"demand": 36.0, "supply": 38.0, "type": "residential"},
            "outer_richmond": {"demand": 32.0, "supply": 35.0, "type": "residential"},
            "inner_sunset": {"demand": 34.0, "supply": 37.0, "type": "residential"},
            "outer_sunset": {"demand": 30.0, "supply": 34.0, "type": "residential"},
            "parkside": {"demand": 28.0, "supply": 31.0, "type": "residential"},
            "lakeshore": {"demand": 25.0, "supply": 28.0, "type": "residential"},

            # South SF - mixed industrial/residential
            "potrero_hill": {"demand": 38.0, "supply": 37.0, "type": "mixed"},
            "dogpatch": {"demand": 35.0, "supply": 34.0, "type": "mixed"},
            "mission_bay": {"demand": 55.0, "supply": 52.0, "type": "commercial"},
            "south_beach": {"demand": 42.0, "supply": 40.0, "type": "mixed"},
            "bayview_hunters_point": {"demand": 40.0, "supply": 38.0, "type": "industrial"},
            "visitacion_valley": {"demand": 28.0, "supply": 27.0, "type": "residential"},
            "excelsior": {"demand": 30.0, "supply": 29.0, "type": "residential"},
            "crocker_amazon": {"demand": 26.0, "supply": 25.0, "type": "residential"},

            # Parks and low-density
            "golden_gate_park": {"demand": 8.0, "supply": 15.0, "type": "park"},
            "presidio": {"demand": 12.0, "supply": 20.0, "type": "park"},
            "seacliff": {"demand": 22.0, "supply": 24.0, "type": "residential"},
            "twin_peaks": {"demand": 15.0, "supply": 18.0, "type": "residential"},
            "glen_park": {"demand": 24.0, "supply": 23.0, "type": "residential"},

            # Other neighborhoods
            "oceanview": {"demand": 26.0, "supply": 25.0, "type": "residential"},
            "ingleside": {"demand": 28.0, "supply": 27.0, "type": "residential"},
            "sunnyside": {"demand": 25.0, "supply": 24.0, "type": "residential"},
            "west_portal": {"demand": 27.0, "supply": 26.0, "type": "residential"},
            "forest_hill": {"demand": 24.0, "supply": 25.0, "type": "residential"},
            "balboa_park": {"demand": 22.0, "supply": 21.0, "type": "residential"},
            "westwood_park": {"demand": 20.0, "supply": 20.0, "type": "residential"},
            "yosemite_slopes": {"demand": 18.0, "supply": 19.0, "type": "residential"},
            "treasure_island": {"demand": 15.0, "supply": 18.0, "type": "mixed"},
        }

    def _initialize_generators(self):
        """Initialize generator types with different costs and characteristics"""
        self.generators = {
            "gas_peaker": {
                "marginal_cost_usd_per_mwh": 180.0,  # Expensive but fast
                "ramp_rate_mw_per_min": 10.0,
                "min_output_mw": 5.0,
                "max_output_mw": 50.0,
                "startup_cost_usd": 500.0,
                "emissions_kg_co2_per_mwh": 600.0,
                "count": 5  # 5 peaker plants across SF
            },
            "gas_combined_cycle": {
                "marginal_cost_usd_per_mwh": 60.0,  # Cheaper, slower
                "ramp_rate_mw_per_min": 2.0,
                "min_output_mw": 20.0,
                "max_output_mw": 150.0,
                "startup_cost_usd": 1500.0,
                "emissions_kg_co2_per_mwh": 400.0,
                "count": 2  # 2 large CC plants
            },
            "battery_discharge": {
                "marginal_cost_usd_per_mwh": 20.0,  # Cheap but limited by SOC
                "ramp_rate_mw_per_min": 50.0,  # Very fast
                "degradation_cost_usd_per_mwh": 15.0,  # Battery wear cost
                "efficiency": 0.95,
                "emissions_kg_co2_per_mwh": 0.0
            }
        }

    def _initialize_renewables(self):
        """Initialize renewable generation with variability"""
        # Track renewable capacity by zone
        self.renewable_capacity = {
            # Solar-rich western neighborhoods
            "outer_sunset": {"solar_mw": 12.0, "wind_mw": 0.0},
            "outer_richmond": {"solar_mw": 10.0, "wind_mw": 0.0},
            "inner_sunset": {"solar_mw": 10.0, "wind_mw": 0.0},
            "inner_richmond": {"solar_mw": 10.0, "wind_mw": 0.0},
            "parkside": {"solar_mw": 8.0, "wind_mw": 0.0},

            # Wind potential near coast/bay
            "treasure_island": {"solar_mw": 2.0, "wind_mw": 8.0},
            "presidio": {"solar_mw": 5.0, "wind_mw": 5.0},
            "golden_gate_park": {"solar_mw": 3.0, "wind_mw": 4.0},
        }

        # Forecast error parameters (mean=0, std=20% of capacity)
        self.renewable_forecast_error = 0.2

    def _initialize_transmission_constraints(self):
        """Initialize transmission line capacity limits"""
        # Main transmission corridors in SF
        self.transmission_lines = {
            # North-South backbone (high capacity)
            "financial_district|mission_bay": {"capacity_mw": 80.0, "reactance": 0.05},
            "mission_bay|potrero_hill": {"capacity_mw": 70.0, "reactance": 0.06},

            # East-West corridors (medium capacity)
            "financial_district|russian_hill": {"capacity_mw": 60.0, "reactance": 0.04},
            "mission|castro": {"capacity_mw": 55.0, "reactance": 0.05},
            "inner_sunset|haight_ashbury": {"capacity_mw": 50.0, "reactance": 0.06},

            # Local feeders (lower capacity - bottleneck potential!)
            "tenderloin|civic_center": {"capacity_mw": 35.0, "reactance": 0.08},
            "chinatown|north_beach": {"capacity_mw": 40.0, "reactance": 0.07},
            "outer_richmond|seacliff": {"capacity_mw": 30.0, "reactance": 0.09},
        }

        # Voltage limits (per-unit)
        self.voltage_min = 0.95
        self.voltage_max = 1.05

    def _initialize_battery_degradation(self):
        """Track battery health and degradation"""
        self.battery_health = {}  # SOH (state of health) per node
        for node_id in self.baselines.keys():
            self.battery_health[node_id] = {
                "soh_percent": 100.0,  # Starts at 100%
                "cycle_count": 0,
                "throughput_mwh": 0.0,
                "degradation_rate_per_cycle": 0.02  # Loses 0.02% per full cycle
            }

    def _get_hour_factor(self, hour, node_type):
        """Get demand multiplier based on time of day and node type"""
        if node_type == "commercial":
            # Peak 9am-5pm
            if 9 <= hour <= 17:
                return 1.3
            elif 6 <= hour <= 8 or 18 <= hour <= 20:
                return 0.9
            else:
                return 0.4

        elif node_type == "residential":
            # Peak 6-9am and 6-10pm
            if 6 <= hour <= 9 or 18 <= hour <= 22:
                return 1.2
            elif 10 <= hour <= 17:
                return 0.7
            else:
                return 0.5

        elif node_type == "mixed":
            # Moderate throughout day
            if 8 <= hour <= 20:
                return 1.0
            else:
                return 0.6

        elif node_type == "industrial":
            # Constant during work hours
            if 7 <= hour <= 18:
                return 1.1
            else:
                return 0.5

        elif node_type == "park":
            # Low, slight increase during day
            if 10 <= hour <= 18:
                return 1.3
            else:
                return 0.8

        return 1.0

    def _smooth_transition(self, prev_value, target_value, smoothing=0.3):
        """Smooth transition between values (exponential moving average)"""
        return prev_value * (1 - smoothing) + target_value * smoothing

    def _update_weather(self):
        """Update weather with realistic SF patterns"""
        hour = self.current_time.hour
        month = self.current_time.month

        # Temperature only changes every 5 hours (60 ticks), not every 5 minutes
        if self.tick_count % 60 == 0 or self.tick_count == 0:
            # SF temperature: cool year-round (50-70°F / 10-21°C)
            # Warmer Sep-Oct, cooler Jun-Aug (fog season)
            base_temp = {
                1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17,
                7: 17, 8: 18, 9: 19, 10: 18, 11: 15, 12: 13
            }[month]

            # Daily variation (warmer afternoon)
            daily_variation = 4 * math.sin((hour - 6) * math.pi / 12)

            # Very gradual change - small random variation
            target_temp = base_temp + daily_variation + random.uniform(-0.5, 0.5)
            # Very smooth transition (only 5% change per 5-hour period)
            self.prev_temp = self._smooth_transition(self.prev_temp, target_temp, 0.05)

        # Solar irradiance (W/m²) - SF gets good sun in fall, fog in summer
        if 6 <= hour <= 19:
            solar_peak = {1: 600, 2: 700, 3: 800, 4: 900, 5: 950, 6: 850,
                         7: 800, 8: 850, 9: 950, 10: 900, 11: 700, 12: 600}[month]
            solar_curve = math.sin((hour - 6) * math.pi / 13)
            target_solar = solar_peak * solar_curve * random.uniform(0.8, 1.0)
        else:
            target_solar = 0.0

        self.prev_solar = self._smooth_transition(self.prev_solar, target_solar, 0.3)

        # Wind (m/s) - SF is windy, especially afternoon
        base_wind = 4.5 + 2 * math.sin((hour - 10) * math.pi / 14)
        target_wind = max(0, base_wind + random.uniform(-1.5, 1.5))
        self.prev_wind = self._smooth_transition(self.prev_wind, target_wind, 0.2)

        return {
            "temp_c": round(self.prev_temp, 1),
            "solar_irradiance_wm2": round(self.prev_solar, 1),
            "wind_mps": round(self.prev_wind, 1)
        }

    def _update_prices(self):
        """Update electricity prices based on time and demand"""
        hour = self.current_time.hour

        # Day-ahead price (more stable)
        if 16 <= hour <= 21:  # Evening peak
            da_price = 0.32 + random.uniform(-0.02, 0.02)
        elif 9 <= hour <= 15:  # Midday
            da_price = 0.26 + random.uniform(-0.02, 0.02)
        else:  # Off-peak
            da_price = 0.18 + random.uniform(-0.02, 0.02)

        # Real-time price (more volatile)
        rt_price = da_price * random.uniform(0.9, 1.3)

        return {
            "da_usd_per_kwh": round(da_price, 3),
            "rt_usd_per_kwh": round(rt_price, 3)
        }

    def _calculate_renewable_output(self, node_id):
        """Calculate actual renewable generation with forecast error"""
        if node_id not in self.renewable_capacity:
            return 0.0

        capacity = self.renewable_capacity[node_id]

        # Solar output (depends on irradiance)
        solar_cf = self.prev_solar / 1000.0  # Capacity factor 0-1
        solar_forecast_error = random.gauss(0, self.renewable_forecast_error)
        solar_output = capacity["solar_mw"] * solar_cf * (1 + solar_forecast_error)
        solar_output = max(0, min(capacity["solar_mw"], solar_output))

        # Wind output (depends on wind speed, cubic relationship)
        wind_cf = min(1.0, (self.prev_wind / 12.0) ** 3)  # Cut-in to rated
        wind_forecast_error = random.gauss(0, self.renewable_forecast_error)
        wind_output = capacity["wind_mw"] * wind_cf * (1 + wind_forecast_error)
        wind_output = max(0, min(capacity["wind_mw"], wind_output))

        return solar_output + wind_output

    def _check_transmission_constraint(self, edge_id, flow_mw):
        """Check if power flow violates transmission line capacity"""
        if edge_id in self.transmission_lines:
            capacity = self.transmission_lines[edge_id]["capacity_mw"]
            loading = abs(flow_mw) / capacity if capacity > 0 else 0
            is_violated = loading > 1.0
            return {
                "capacity_mw": capacity,
                "flow_mw": flow_mw,
                "loading_percent": round(loading * 100, 1),
                "violated": is_violated,
                "margin_mw": round(capacity - abs(flow_mw), 2)
            }
        return None

    def calculate_action_cost(self, action_type, node_id, mw, duration_min=5):
        """
        Calculate cost of an action for RL reward function
        Returns dict with cost breakdown
        """
        duration_hours = duration_min / 60.0
        energy_mwh = mw * duration_hours

        cost_breakdown = {
            "generation_cost_usd": 0.0,
            "battery_degradation_cost_usd": 0.0,
            "startup_cost_usd": 0.0,
            "emissions_kg_co2": 0.0,
            "total_cost_usd": 0.0
        }

        if action_type == "increase_supply":
            # Use gas peaker (expensive but fast)
            gen = self.generators["gas_peaker"]
            cost_breakdown["generation_cost_usd"] = energy_mwh * gen["marginal_cost_usd_per_mwh"]
            cost_breakdown["emissions_kg_co2"] = energy_mwh * gen["emissions_kg_co2_per_mwh"]
            if mw > gen["min_output_mw"]:
                cost_breakdown["startup_cost_usd"] = gen["startup_cost_usd"]

        elif action_type in ["discharge_storage", "discharge_battery"]:
            # Battery discharge
            gen = self.generators["battery_discharge"]
            cost_breakdown["generation_cost_usd"] = energy_mwh * gen["marginal_cost_usd_per_mwh"]
            cost_breakdown["battery_degradation_cost_usd"] = energy_mwh * gen["degradation_cost_usd_per_mwh"]

            # Update battery health
            if node_id in self.battery_health:
                health = self.battery_health[node_id]
                health["throughput_mwh"] += energy_mwh
                # Assume 50 MWh capacity = 1 full cycle
                cycles = energy_mwh / 50.0
                health["cycle_count"] += cycles
                health["soh_percent"] -= cycles * health["degradation_rate_per_cycle"]
                health["soh_percent"] = max(50.0, health["soh_percent"])  # Floor at 50%

        elif action_type == "reduce_demand":
            # Demand response has cost (customer inconvenience)
            cost_breakdown["generation_cost_usd"] = energy_mwh * 50.0  # $50/MWh incentive

        cost_breakdown["total_cost_usd"] = (
            cost_breakdown["generation_cost_usd"] +
            cost_breakdown["battery_degradation_cost_usd"] +
            cost_breakdown["startup_cost_usd"]
        )

        return cost_breakdown

    def _update_node(self, node_id, node_data):
        """Update dynamic fields for a single node"""
        # Get baseline or use default
        baseline = self.baselines.get(node_id, {"demand": 30.0, "supply": 28.0, "type": "residential"})

        hour = self.current_time.hour
        node_type = baseline["type"]

        # Calculate target demand with time-of-day pattern
        hour_factor = self._get_hour_factor(hour, node_type)
        target_demand = baseline["demand"] * hour_factor * random.uniform(0.95, 1.05)

        # Smooth transition from previous value
        prev_demand = self.prev_demands.get(node_id, target_demand)
        new_demand = self._smooth_transition(prev_demand, target_demand, 0.25)
        self.prev_demands[node_id] = new_demand

        # Supply tries to match demand with renewable generation
        renewable_output = self._calculate_renewable_output(node_id)
        base_supply = baseline["supply"] * hour_factor * random.uniform(0.93, 1.03)
        target_supply = base_supply + renewable_output

        prev_supply = self.prev_supplies.get(node_id, target_supply)
        new_supply = self._smooth_transition(prev_supply, target_supply, 0.25)
        self.prev_supplies[node_id] = new_supply

        # Store renewable contribution for visibility
        node_data["renewable_mw"] = round(renewable_output, 2)

        # Update dynamic fields
        node_data["demand_mw"] = round(new_demand, 2)
        node_data["supply_mw"] = round(new_supply, 2)
        node_data["net_load_mw"] = round(new_demand - new_supply, 2)

        # Storage SOC changes slowly
        current_soc = node_data["storage"]["soc"]
        if new_supply > new_demand:  # Charging
            target_soc = min(0.95, current_soc + random.uniform(0.01, 0.03))
        else:  # Discharging
            target_soc = max(0.15, current_soc - random.uniform(0.01, 0.03))
        node_data["storage"]["soc"] = round(target_soc, 3)

        # Update risk based on supply-demand imbalance
        imbalance_ratio = abs(new_demand - new_supply) / max(new_demand, 1.0)
        node_data["risk"]["overload"] = round(min(1.0, imbalance_ratio * 0.6 + random.uniform(0, 0.1)), 3)
        node_data["risk"]["n_1_margin"] = round(max(0, 1.0 - imbalance_ratio * 0.8), 3)

        # Demand response fatigue increases with usage
        current_dr = node_data["flex"]["active_dr_percent"]
        if current_dr > 0:
            node_data["flex"]["fatigue"] = min(1.0, node_data["flex"]["fatigue"] + 0.02)
        else:
            node_data["flex"]["fatigue"] = max(0, node_data["flex"]["fatigue"] - 0.01)

        # Price signal based on local stress
        node_data["price_signal"] = round(imbalance_ratio * 2.0, 3)

    def _update_edges(self, edges, nodes):
        """Update power flow on transmission lines with constraint checking"""
        for edge_id, edge_data in edges.items():
            if not edge_data["active"]:
                continue

            # Parse edge nodes
            node_a, node_b = edge_id.split("|")

            if node_a in nodes and node_b in nodes:
                net_a = nodes[node_a]["net_load_mw"]
                net_b = nodes[node_b]["net_load_mw"]

                # Flow from surplus to deficit
                flow = (net_a - net_b) * random.uniform(0.3, 0.5)
                edge_data["flow_mw"] = round(flow, 2)

                # Check transmission constraint
                constraint_check = self._check_transmission_constraint(edge_id, flow)
                if constraint_check:
                    edge_data["capacity_mw"] = constraint_check["capacity_mw"]
                    edge_data["loading"] = constraint_check["loading_percent"] / 100.0
                    edge_data["constraint_violated"] = constraint_check["violated"]
                    edge_data["margin_mw"] = constraint_check["margin_mw"]
                else:
                    # No constraint defined - use default
                    capacity = edge_data.get("capacity_mw", 50.0)
                    if capacity > 0:
                        edge_data["loading"] = round(abs(flow) / capacity, 3)
                        edge_data["constraint_violated"] = abs(flow) > capacity

    def _update_kpis(self, nodes):
        """Update city-wide KPIs"""
        total_demand = sum(n["demand_mw"] for n in nodes.values())
        total_supply = sum(n["supply_mw"] for n in nodes.values())
        unserved = max(0, total_demand - total_supply)
        avg_risk = sum(n["risk"]["overload"] for n in nodes.values()) / len(nodes)

        # Fairness: how uniformly is supply meeting demand
        supply_ratios = [min(1.0, n["supply_mw"] / max(n["demand_mw"], 1.0)) for n in nodes.values()]
        avg_ratio = sum(supply_ratios) / len(supply_ratios)
        variance = sum((r - avg_ratio) ** 2 for r in supply_ratios) / len(supply_ratios)
        fairness = max(0, 1.0 - variance)

        # Count renewable generation
        total_renewable = sum(n.get("renewable_mw", 0) for n in nodes.values())

        return {
            "city_demand_mw": round(total_demand, 2),
            "city_supply_mw": round(total_supply, 2),
            "unserved_energy_proxy_mw": round(unserved, 2),
            "avg_overload_risk": round(avg_risk, 3),
            "total_curtailment_mw": 0.0,  # To be set by control actions
            "fairness_index": round(fairness, 3),
            "total_renewable_mw": round(total_renewable, 2),
            "renewable_penetration": round(total_renewable / total_supply * 100, 1) if total_supply > 0 else 0
        }

    def calculate_rl_reward(self, actions_taken, grid_state_before, grid_state_after):
        """
        Calculate RL reward for taken actions by comparing BEFORE vs AFTER
        Reward is based on:
        1. Grid improvement (reduced deficit, risk, violations)
        2. Action costs (cheaper is better)
        3. Emissions (lower is better)
        4. Penalty for taking no action when action is needed
        """
        # Check if no actions were taken - apply penalty if deficit exists
        no_action_penalty = 0.0
        if len(actions_taken) == 0:
            kpis_before = grid_state_before.get("kpis", {})
            deficit_before = kpis_before.get("city_demand_mw", 0) - kpis_before.get("city_supply_mw", 0)
            # If there's a deficit > 1 MW but no action taken, apply penalty
            if deficit_before > 1.0:
                no_action_penalty = 0.3  # 30% raw score penalty

        # Calculate costs for all actions
        total_cost = 0.0
        total_emissions = 0.0

        for action in actions_taken:
            cost_data = self.calculate_action_cost(
                action["action_type"],
                action["node_id"],
                action["target_mw"],
                duration_min=5
            )
            total_cost += cost_data["total_cost_usd"]
            total_emissions += cost_data["emissions_kg_co2"]

        # Get BEFORE metrics
        kpis_before = grid_state_before.get("kpis", {})
        unserved_before = kpis_before.get("unserved_energy_proxy_mw", 0)
        deficit_before = kpis_before.get("city_demand_mw", 0) - kpis_before.get("city_supply_mw", 0)
        risk_before = kpis_before.get("avg_overload_risk", 0)
        fairness_before = kpis_before.get("fairness_index", 0)

        edges_before = grid_state_before.get("edges", {})
        violations_before = sum(1 for e in edges_before.values() if e.get("constraint_violated", False))

        # Get AFTER metrics
        kpis_after = grid_state_after.get("kpis", {})
        unserved_after = kpis_after.get("unserved_energy_proxy_mw", 0)
        deficit_after = kpis_after.get("city_demand_mw", 0) - kpis_after.get("city_supply_mw", 0)
        risk_after = kpis_after.get("avg_overload_risk", 0)
        fairness_after = kpis_after.get("fairness_index", 0)

        edges_after = grid_state_after.get("edges", {})
        violations_after = sum(1 for e in edges_after.values() if e.get("constraint_violated", False))

        # Calculate IMPROVEMENTS (positive = better)
        deficit_improvement = deficit_before - deficit_after  # Reduced deficit is good
        unserved_improvement = unserved_before - unserved_after  # Reduced unserved is good
        risk_improvement = risk_before - risk_after  # Reduced risk is good
        fairness_improvement = fairness_after - fairness_before  # Increased fairness is good
        violations_improvement = violations_before - violations_after  # Fewer violations is good

        # Calculate normalized reward components (each 0-1 scale)

        # 1. Deficit reduction (0 to 1, where 1 = eliminated all deficit)
        # More sensitive: use percentage reduction instead of absolute
        if deficit_before > 0:
            deficit_reduction_pct = deficit_improvement / deficit_before
            deficit_score = max(0, min(1, deficit_reduction_pct))  # 0-100% reduction
        else:
            deficit_score = 1.0  # Perfect if no deficit to begin with

        # 2. Risk reduction (0 to 1, where 1 = eliminated all risk)
        # More sensitive: use percentage reduction
        if risk_before > 0.01:  # If there was meaningful risk
            risk_reduction_pct = risk_improvement / risk_before
            risk_score = max(0, min(1, risk_reduction_pct * 2))  # 50% reduction = 1.0 score
        else:
            risk_score = 0.8  # Good baseline if risk was already low

        # 3. Cost efficiency (0 to 1, where 1 = no cost, 0 = very expensive)
        # More sensitive: quadratic penalty (makes cost differences more impactful)
        normalized_cost = total_cost / 2000.0  # Normalize to 0-1 (assuming max $2000)
        cost_score = max(0, min(1, (1 - normalized_cost) ** 2))  # Quadratic: better scores for low costs

        # 4. Fairness improvement (0 to 1)
        # More sensitive: wider range
        fairness_score = max(0, min(1, 0.5 + fairness_improvement * 2)) if fairness_before > 0 else 0.5

        # 5. Violations penalty (0 to 1, where 1 = no violations)
        violation_score = 1.0 if violations_after == 0 else max(0, 1 - violations_after / 5.0)  # Harsher penalty

        # Weighted average (normalized to 0-1) - this is the "raw score"
        raw_score = (
            deficit_score * 0.40 +      # 40% weight on deficit reduction
            cost_score * 0.20 +          # 20% weight on cost efficiency
            risk_score * 0.20 +          # 20% weight on risk reduction
            fairness_score * 0.10 +      # 10% weight on fairness
            violation_score * 0.10       # 10% weight on violations
        )

        # Ensure raw score is bounded [0, 1]
        raw_score = max(0.0, min(1.0, raw_score))

        # SHAPED REWARD: Use raw score directly for PPO training
        # This provides gradient signal even for sub-optimal actions
        # Raw score is already bounded [0, 1] and combines all objectives
        THRESHOLD = 0.7  # Keep for logging/analysis
        reward = raw_score  # Use continuous reward for RL

        # Track cumulative metrics
        self.cumulative_cost += total_cost
        self.cumulative_emissions += total_emissions
        self.cumulative_unserved_energy += unserved_after

        return {
            "reward": reward,  # BINARY: 0 or 1 (for RL training)
            "raw_score": round(raw_score, 3),  # Continuous score before threshold
            "threshold": THRESHOLD,
            "cost_usd": round(total_cost, 2),
            "emissions_kg_co2": round(total_emissions, 2),

            # Reward component scores (for debugging)
            "deficit_score": round(deficit_score, 3),
            "cost_score": round(cost_score, 3),
            "risk_score": round(risk_score, 3),
            "fairness_score": round(fairness_score, 3),
            "violation_score": round(violation_score, 3),

            # BEFORE metrics
            "deficit_before_mw": round(deficit_before, 2),
            "unserved_before_mw": round(unserved_before, 2),
            "risk_before": round(risk_before, 3),
            "fairness_before": round(fairness_before, 3),
            "violations_before": violations_before,

            # AFTER metrics
            "deficit_after_mw": round(deficit_after, 2),
            "unserved_after_mw": round(unserved_after, 2),
            "risk_after": round(risk_after, 3),
            "fairness_after": round(fairness_after, 3),
            "violations_after": violations_after,

            # IMPROVEMENTS
            "deficit_improvement_mw": round(deficit_improvement, 2),
            "unserved_improvement_mw": round(unserved_improvement, 2),
            "risk_improvement": round(risk_improvement, 3),
            "fairness_improvement": round(fairness_improvement, 3),
            "violations_improvement": violations_improvement,

            # Cumulative
            "cumulative_cost_usd": round(self.cumulative_cost, 2),
            "cumulative_emissions_kg": round(self.cumulative_emissions, 2),
            "cumulative_unserved_mwh": round(self.cumulative_unserved_energy, 2)
        }

    def generate_tick(self):
        """Generate one 5-minute tick of data"""
        # Deep copy template
        data = json.loads(json.dumps(self.base_data))

        # Update simulation metadata
        data["sim"]["frame_id"] = f"sf_{self.tick_count:06d}"
        data["sim"]["sim_time"] = self.current_time.strftime("%Y-%m-%dT%H:%M:%S")
        data["sim"]["generated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Update drivers
        data["drivers"]["time_of_day"] = self.current_time.strftime("%H:%M")
        data["drivers"]["weekday"] = self.current_time.weekday() < 5
        data["drivers"]["weather"] = self._update_weather()
        data["drivers"]["price"] = self._update_prices()

        # Update all nodes
        for node_id, node_data in data["nodes"].items():
            self._update_node(node_id, node_data)

        # Update edges
        self._update_edges(data["edges"], data["nodes"])

        # Update KPIs
        data["kpis"] = self._update_kpis(data["nodes"])

        # Advance time
        self.current_time += timedelta(minutes=5)
        self.tick_count += 1

        return data

    def save_tick(self, data, output_path="data/current_state.json"):
        """Save the current tick to file"""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def reset(self, start_time=None):
        """Reset simulator to initial state - for RL episode reset"""
        self.current_time = start_time or datetime.now()
        self.tick_count = 0
        self.prev_demands = {}
        self.prev_supplies = {}
        self.prev_temp = 18.0
        self.prev_solar = 0.0
        self.prev_wind = 3.0

        # Reset cumulative RL metrics
        self.cumulative_cost = 0.0
        self.cumulative_emissions = 0.0
        self.cumulative_unserved_energy = 0.0

        # Reset battery health
        for node_id in self.battery_health.keys():
            self.battery_health[node_id] = {
                "soh_percent": 100.0,
                "cycle_count": 0,
                "throughput_mwh": 0.0,
                "degradation_rate_per_cycle": 0.02
            }


if __name__ == "__main__":
    print("=" * 60)
    print("SF Grid Simulator - 5-minute tick generator")
    print("=" * 60)

    # Initialize simulator
    sim = GridSimulator("data/data.json")

    # Generate and save 3 ticks as demo
    for i in range(3):
        print(f"\n[Tick {i}] Generating data for {sim.current_time.strftime('%Y-%m-%d %H:%M')}")

        data = sim.generate_tick()

        # Print summary
        kpis = data["kpis"]
        weather = data["drivers"]["weather"]
        price = data["drivers"]["price"]

        print(f"  Weather: {weather['temp_c']}°C, Solar: {weather['solar_irradiance_wm2']}W/m², Wind: {weather['wind_mps']}m/s")
        print(f"  Price: DA=${price['da_usd_per_kwh']}/kWh, RT=${price['rt_usd_per_kwh']}/kWh")
        print(f"  Demand: {kpis['city_demand_mw']:.1f}MW, Supply: {kpis['city_supply_mw']:.1f}MW")
        print(f"  Risk: {kpis['avg_overload_risk']:.3f}, Fairness: {kpis['fairness_index']:.3f}")

        # Save current state
        sim.save_tick(data)
        print(f"  → Saved to data/current_state.json")

    print(f"\n{'=' * 60}")
    print("Demo complete!")
