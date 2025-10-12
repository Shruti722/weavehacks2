# weavehacks2/eval/critique.py
import json
from typing import Dict, Any

def critique(raw_text: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw_text)
    except Exception as e:
        return {"verdict": "REJECT", "reason": f"json_parse_fail: {e}"}

    actions = data.get("actions")
    if not isinstance(actions, list) or len(actions) == 0:
        return {"verdict": "REJECT", "reason": "no_actions"}

    for a in actions:
        if not isinstance(a, dict):
            return {"verdict": "REJECT", "reason": "action_not_object"}
        if "node_id" not in a or "adjustments" not in a:
            return {"verdict": "REJECT", "reason": "bad_action_shape"}
        adj = a["adjustments"]
        if not isinstance(adj, dict) or not adj:
            return {"verdict": "REJECT", "reason": "empty_adjustments"}
        for k, v in adj.items():
            if not isinstance(v, (int, float)):
                return {"verdict": "REJECT", "reason": f"non_numeric_adjustment:{k}"}

    return {"verdict": "ACCEPT", "reason": "ok"}
