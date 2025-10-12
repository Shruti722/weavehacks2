"""
Real-time Grid Data Simulator
Generates realistic time-series data for SF power grid with smooth 5-minute updates
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

        # Supply tries to match demand with some lag and renewable variation
        solar_bonus = self.prev_solar / 1000.0 * 5  # Solar adds up to 5 MW
        target_supply = baseline["supply"] * hour_factor * random.uniform(0.93, 1.03) + solar_bonus

        prev_supply = self.prev_supplies.get(node_id, target_supply)
        new_supply = self._smooth_transition(prev_supply, target_supply, 0.25)
        self.prev_supplies[node_id] = new_supply

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
        """Update power flow on transmission lines"""
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

                capacity = edge_data.get("capacity_mw", 50.0)
                if capacity > 0:
                    edge_data["loading"] = round(abs(flow) / capacity, 3)

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

        return {
            "city_demand_mw": round(total_demand, 2),
            "city_supply_mw": round(total_supply, 2),
            "unserved_energy_proxy_mw": round(unserved, 2),
            "avg_overload_risk": round(avg_risk, 3),
            "total_curtailment_mw": 0.0,  # To be set by control actions
            "fairness_index": round(fairness, 3)
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
        """Reset simulator to initial state"""
        self.current_time = start_time or datetime.now()
        self.tick_count = 0
        self.prev_demands = {}
        self.prev_supplies = {}
        self.prev_temp = 18.0
        self.prev_solar = 0.0
        self.prev_wind = 3.0


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
