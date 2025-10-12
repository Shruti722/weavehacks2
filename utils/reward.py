"""
Reward computation for the RL system
Evaluates grid performance based on cost, risk, and fairness
"""


def compute_reward(state, weights=None):
    """
    Calculate reward for the current grid state.

    Reward = -(α * normalized_cost + β * risk) + γ * fairness

    Args:
        state (dict): Current grid state
        weights (dict, optional): Custom weight parameters
            - alpha: weight for cost (default: 1.0)
            - beta: weight for risk (default: 1.5)
            - gamma: weight for fairness (default: 0.5)

    Returns:
        dict: Reward breakdown including total reward and components
    """
    # Default weights
    if weights is None:
        weights = {
            "alpha": 1.0,
            "beta": 1.5,
            "gamma": 0.5
        }

    alpha = weights.get("alpha", 1.0)
    beta = weights.get("beta", 1.5)
    gamma = weights.get("gamma", 0.5)

    # Extract metrics from state
    total_cost = state.get("total_cost", 0)
    fairness_index = state.get("fairness_index", 0)

    # Calculate aggregate risk
    nodes = state.get("nodes", [])
    if nodes:
        avg_risk = sum(node.get("risk_index", 0) for node in nodes) / len(nodes)
    else:
        avg_risk = 0

    # Normalize cost (assuming typical range 5000-20000)
    normalized_cost = total_cost / 15000.0

    # Calculate reward components
    cost_penalty = alpha * normalized_cost
    risk_penalty = beta * avg_risk
    fairness_bonus = gamma * fairness_index

    # Total reward
    total_reward = -(cost_penalty + risk_penalty) + fairness_bonus

    # Return detailed breakdown
    return {
        "total_reward": total_reward,
        "components": {
            "cost_penalty": -cost_penalty,
            "risk_penalty": -risk_penalty,
            "fairness_bonus": fairness_bonus
        },
        "metrics": {
            "total_cost": total_cost,
            "normalized_cost": normalized_cost,
            "avg_risk": avg_risk,
            "fairness_index": fairness_index
        }
    }


def compute_reward_delta(prev_state, current_state, weights=None):
    """
    Calculate the change in reward between two states.

    Args:
        prev_state (dict): Previous grid state
        current_state (dict): Current grid state
        weights (dict, optional): Custom weight parameters

    Returns:
        dict: Reward delta and comparison
    """
    prev_reward = compute_reward(prev_state, weights)
    current_reward = compute_reward(current_state, weights)

    delta = current_reward["total_reward"] - prev_reward["total_reward"]

    return {
        "reward_delta": delta,
        "prev_reward": prev_reward["total_reward"],
        "current_reward": current_reward["total_reward"],
        "improvement": delta > 0,
        "prev_breakdown": prev_reward,
        "current_breakdown": current_reward
    }


def evaluate_performance(state_history, weights=None):
    """
    Evaluate overall performance across multiple states.

    Args:
        state_history (list): List of grid states over time
        weights (dict, optional): Custom weight parameters

    Returns:
        dict: Performance summary statistics
    """
    if not state_history:
        return {"error": "No state history provided"}

    rewards = [compute_reward(state, weights)["total_reward"] for state in state_history]

    return {
        "total_steps": len(state_history),
        "cumulative_reward": sum(rewards),
        "avg_reward": sum(rewards) / len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "final_reward": rewards[-1],
        "trend": "improving" if len(rewards) > 1 and rewards[-1] > rewards[0] else "declining"
    }
