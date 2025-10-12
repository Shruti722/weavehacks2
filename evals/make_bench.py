import json, copy, random, datetime as dt
from pathlib import Path

# Resolve project root = folder that contains this "weavehacks2" package
ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "data.json"  # you have this file
OUT_PATH = ROOT / "data" / "bench" / "scenarios.json"

def main():
    print(f"[make_bench] Project root: {ROOT}")
    print(f"[make_bench] Base scenario: {BASE_PATH}")

    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Base scenario not found at {BASE_PATH}")

    base = json.loads(BASE_PATH.read_text())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    sim_time = base.get("sim", {}).get("sim_time")
    try:
        t0 = dt.datetime.fromisoformat(sim_time) if sim_time else None
    except Exception:
        t0 = None

    def tweak(b, minutes, da, rt, temp, solar, wind, outages=()):
        s = copy.deepcopy(b)

        # timestamp + HH:MM
        if t0:
            ts = (t0 + dt.timedelta(minutes=minutes))
            s.setdefault("sim", {})["sim_time"] = ts.isoformat()
            s.setdefault("drivers", {})["time_of_day"] = ts.strftime("%H:%M")

        # price & weather
        s.setdefault("drivers", {}).setdefault("price", {})
        s["drivers"]["price"]["da_usd_per_kwh"] = da
        s["drivers"]["price"]["rt_usd_per_kwh"] = rt
        s.setdefault("drivers", {}).setdefault("weather", {})
        s["drivers"]["weather"] = {"temp_c": temp, "solar_irradiance_wm2": solar, "wind_mps": wind}

        # perturb nodes
        nodes = s.get("nodes", {})
        for name, node in nodes.items():
            nl = rng.uniform(-0.2, 0.6)
            node["net_load_mw"] = round(nl, 2)
            node["demand_mw"] = max(nl, 0.0)
            node["supply_mw"] = max(-nl, 0.0)

            node.setdefault("risk", {})
            node["risk"]["overload"] = max(0.0, round(rng.uniform(0, 0.35), 2))
            node["risk"]["n_1_margin"] = max(0.0, round(rng.uniform(0, 0.35), 2))

            node.setdefault("flags", {})
            node["flags"]["outage"] = name in outages

            node.setdefault("storage", {"soc": 0, "power_mw_limit": 0, "energy_mwh_cap": 0})
            node.setdefault("flex", {"available_mw": 0, "active_dr_percent": 0, "fatigue": 0, "cooldown_steps": 0})

        # KPIs
        total_demand = sum(n.get("demand_mw", 0) for n in nodes.values())
        total_supply = sum(n.get("supply_mw", 0) for n in nodes.values())
        s.setdefault("kpis", {})
        s["kpis"]["city_demand_mw"] = round(total_demand, 2)
        s["kpis"]["city_supply_mw"] = round(total_supply, 2)
        s["kpis"]["unserved_energy_proxy_mw"] = max(0.0, round(total_demand - total_supply, 2))
        s["kpis"]["avg_overload_risk"] = round(
            sum(n["risk"]["overload"] for n in nodes.values()) / max(1, len(nodes)), 3
        )
        s["kpis"].setdefault("fairness_index", 0.0)
        return s

    scenarios = [
        tweak(base,   0, 0.18,0.20, 18,200,3.0),
        tweak(base,  30, 0.20,0.25, 26,800,2.0),
        tweak(base,  60, 0.22,0.28, 33,900,1.0, outages=("mission",)),
        tweak(base,  90, 0.19,0.24, 12, 50,5.0),
        tweak(base, 120, 0.16,0.18, 10,  0,7.0),
        tweak(base, 150, 0.21,0.27, 30,600,4.0, outages=("south_of_market","dogpatch")),
        tweak(base, 180, 0.23,0.30, 35,950,2.0),
        tweak(base, 210, 0.17,0.19, 15,150,6.0),
        tweak(base, 240, 0.24,0.32, 38,980,1.0, outages=("bayview_hunters_point",)),
        tweak(base, 270, 0.15,0.17,  8,  0,8.0),
    ]

    OUT_PATH.write_text(json.dumps(scenarios, indent=2))
    print(f"[make_bench] Wrote {OUT_PATH.resolve()} with {len(scenarios)} scenarios.")
    print(f"[make_bench] File size: {OUT_PATH.stat().st_size} bytes")

if __name__ == "__main__":
    main()
