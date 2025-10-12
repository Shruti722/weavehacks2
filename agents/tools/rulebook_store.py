# weavehacks2/agents/tools/rulebook_store.py
DEFAULT_RULEBOOKS = [
    {
        "name": "v1_safe_batteries_first",
        "safety": {"max_line_loading": 0.90, "soc_min": 0.20, "soc_max": 0.90, "max_mw_step": 10.0},
        "priorities": ["battery", "dr", "peaker"],
        "thresholds": {"use_battery_if_soc_gt": 0.65, "use_peaker_if_unserved_gt_mw": 10.0, "trigger_dr_if_risk_gt": 0.80},
        "caps": {"battery_discharge_cap_mw": 8.0, "peaker_cap_mw": 12.0, "dr_cap_fraction": 0.08}
    },
    {
        "name": "v2_cost_sensitive",
        "safety": {"max_line_loading": 0.88, "soc_min": 0.22, "soc_max": 0.88, "max_mw_step": 8.0},
        "priorities": ["battery", "peaker", "dr"],
        "thresholds": {"use_battery_if_soc_gt": 0.60, "use_peaker_if_unserved_gt_mw": 8.0, "trigger_dr_if_risk_gt": 0.85},
        "caps": {"battery_discharge_cap_mw": 6.0, "peaker_cap_mw": 10.0, "dr_cap_fraction": 0.06}
    },
    {
        "name": "v3_grid_safety_first",
        "safety": {"max_line_loading": 0.85, "soc_min": 0.25, "soc_max": 0.90, "max_mw_step": 7.0},
        "priorities": ["dr", "battery", "peaker"],
        "thresholds": {"use_battery_if_soc_gt": 0.70, "use_peaker_if_unserved_gt_mw": 12.0, "trigger_dr_if_risk_gt": 0.75},
        "caps": {"battery_discharge_cap_mw": 5.0, "peaker_cap_mw": 8.0, "dr_cap_fraction": 0.10}
    }
]
