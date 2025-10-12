"""
Export Training Data from Weave Traces
Queries collected episodes and exports training data using both:
1. Threshold-based selection (reward=1)
2. Top K% selection (by raw_score)

Usage:
    python scripts/export_training_data.py --batch iteration_0 --top-k 0.30
"""

import sys
import os
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weave


@weave.op()
def analyze_episode_distribution(episodes):
    """
    Analyze the distribution of episodes by raw_score.

    Args:
        episodes: List of episode summaries

    Returns:
        dict: Distribution statistics
    """
    if not episodes:
        return {"error": "No episodes provided"}

    raw_scores = [e.get("raw_score", 0) for e in episodes]
    rewards = [e.get("reward", 0) for e in episodes]

    # Sort by raw score
    sorted_episodes = sorted(episodes, key=lambda x: x.get("raw_score", 0), reverse=True)

    # Calculate percentiles
    n = len(raw_scores)
    percentiles = {
        "p90": sorted_episodes[int(n * 0.10)]["raw_score"] if n > 10 else 0,
        "p75": sorted_episodes[int(n * 0.25)]["raw_score"] if n > 4 else 0,
        "p50": sorted_episodes[int(n * 0.50)]["raw_score"] if n > 2 else 0,
        "p25": sorted_episodes[int(n * 0.75)]["raw_score"] if n > 1 else 0,
        "p10": sorted_episodes[int(n * 0.90)]["raw_score"] if n > 1 else 0,
    }

    # Threshold-based stats
    threshold = episodes[0].get("threshold", 0.7) if episodes else 0.7
    successful = [e for e in episodes if e.get("reward", 0) == 1]
    success_rate = len(successful) / len(episodes) if episodes else 0

    # Score distribution
    score_ranges = defaultdict(int)
    for score in raw_scores:
        if score >= 0.9:
            score_ranges["0.9-1.0"] += 1
        elif score >= 0.8:
            score_ranges["0.8-0.9"] += 1
        elif score >= 0.7:
            score_ranges["0.7-0.8"] += 1
        elif score >= 0.6:
            score_ranges["0.6-0.7"] += 1
        elif score >= 0.5:
            score_ranges["0.5-0.6"] += 1
        elif score >= 0.4:
            score_ranges["0.4-0.5"] += 1
        else:
            score_ranges["<0.4"] += 1

    return {
        "total_episodes": len(episodes),
        "threshold": threshold,
        "success_rate": success_rate,
        "successful_count": len(successful),
        "failed_count": len(episodes) - len(successful),
        "percentiles": percentiles,
        "score_ranges": dict(score_ranges),
        "min_score": min(raw_scores),
        "max_score": max(raw_scores),
        "avg_score": sum(raw_scores) / len(raw_scores),
        "sorted_episodes": sorted_episodes
    }


@weave.op()
def select_training_examples_threshold(episodes, threshold=0.7):
    """
    Select training examples using threshold-based approach.

    Args:
        episodes: List of episode summaries
        threshold: Reward threshold (default 0.7)

    Returns:
        dict: Selected examples and metadata
    """
    successful = [e for e in episodes if e.get("reward", 0) == 1]

    return {
        "method": "threshold",
        "threshold": threshold,
        "selected_count": len(successful),
        "selected_episodes": successful,
        "selection_rate": len(successful) / len(episodes) if episodes else 0,
        "min_score": min([e["raw_score"] for e in successful]) if successful else 0,
        "max_score": max([e["raw_score"] for e in successful]) if successful else 0,
        "avg_score": sum([e["raw_score"] for e in successful]) / len(successful) if successful else 0
    }


@weave.op()
def select_training_examples_topk(episodes, top_k_percent=0.30):
    """
    Select training examples using Top K% approach.

    Args:
        episodes: List of episode summaries
        top_k_percent: Percentage of top episodes to select (0.0 to 1.0)

    Returns:
        dict: Selected examples and metadata
    """
    # Sort by raw score
    sorted_episodes = sorted(episodes, key=lambda x: x.get("raw_score", 0), reverse=True)

    # Select top K%
    k = int(len(episodes) * top_k_percent)
    selected = sorted_episodes[:k]

    # Calculate cutoff score
    cutoff_score = selected[-1]["raw_score"] if selected else 0

    # How many would have passed threshold?
    threshold_count = sum(1 for e in selected if e.get("reward", 0) == 1)

    return {
        "method": "top_k_percent",
        "top_k_percent": top_k_percent,
        "selected_count": len(selected),
        "selected_episodes": selected,
        "cutoff_score": cutoff_score,
        "selection_rate": top_k_percent,
        "threshold_overlap": threshold_count,
        "threshold_overlap_rate": threshold_count / len(selected) if selected else 0,
        "min_score": selected[-1]["raw_score"] if selected else 0,
        "max_score": selected[0]["raw_score"] if selected else 0,
        "avg_score": sum([e["raw_score"] for e in selected]) / len(selected) if selected else 0
    }


@weave.op()
def export_training_dataset(batch_name: str, top_k_percent: float = 0.30,
                           use_threshold: bool = True, use_topk: bool = True):
    """
    Export training data from Weave traces using both selection methods.

    Args:
        batch_name: Name of the episode batch to export
        top_k_percent: Percentage for Top K selection
        use_threshold: Whether to export threshold-based selection
        use_topk: Whether to export top-k based selection

    Returns:
        dict: Export summary with both datasets
    """
    print("=" * 70)
    print("📦 Exporting Training Data from Weave")
    print("=" * 70)
    print(f"Batch: {batch_name}")
    print(f"Top K%: {top_k_percent * 100}%")
    print()

    # Load episodes from Weave
    print("Loading episodes from Weave...")
    client = weave.get_client()

    try:
        # Get the dataset
        dataset_ref = f"episodes_{batch_name}"
        dataset = weave.ref(dataset_ref).get()

        if not dataset:
            print(f"❌ Dataset '{dataset_ref}' not found")
            return None

        episodes = dataset.get("episodes", [])
        print(f"✓ Loaded {len(episodes)} episodes\n")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

    # Analyze distribution
    print("📊 Analyzing episode distribution...")
    analysis = analyze_episode_distribution(episodes)

    print(f"\nTotal episodes: {analysis['total_episodes']}")
    print(f"Threshold: {analysis['threshold']}")
    print(f"Success rate: {analysis['success_rate']:.1%} ({analysis['successful_count']} episodes)")
    print(f"Score range: {analysis['min_score']:.3f} to {analysis['max_score']:.3f}")
    print(f"Average score: {analysis['avg_score']:.3f}")
    print()

    print("Score distribution:")
    for range_name, count in sorted(analysis['score_ranges'].items(), reverse=True):
        pct = count / analysis['total_episodes'] * 100
        bar = "█" * int(pct / 2)
        print(f"  {range_name}: {count:4d} ({pct:5.1f}%) {bar}")
    print()

    print("Percentiles:")
    for p_name, p_value in analysis['percentiles'].items():
        print(f"  {p_name}: {p_value:.3f}")
    print()

    # Selection Method 1: Threshold-based
    results = {}

    if use_threshold:
        print("=" * 70)
        print("🎯 Method 1: Threshold-Based Selection")
        print("=" * 70)

        threshold_selection = select_training_examples_threshold(
            episodes,
            threshold=analysis['threshold']
        )

        print(f"Selected: {threshold_selection['selected_count']} episodes")
        print(f"Selection rate: {threshold_selection['selection_rate']:.1%}")
        print(f"Score range: {threshold_selection['min_score']:.3f} to {threshold_selection['max_score']:.3f}")
        print(f"Average score: {threshold_selection['avg_score']:.3f}")
        print()

        # Publish as Weave Dataset
        dataset_name_threshold = f"training_data_{batch_name}_threshold"
        weave.publish(threshold_selection, name=dataset_name_threshold)
        print(f"💾 Published: '{dataset_name_threshold}'")
        print()

        results['threshold'] = threshold_selection

    # Selection Method 2: Top K%
    if use_topk:
        print("=" * 70)
        print(f"🏆 Method 2: Top {top_k_percent*100}% Selection")
        print("=" * 70)

        topk_selection = select_training_examples_topk(
            episodes,
            top_k_percent=top_k_percent
        )

        print(f"Selected: {topk_selection['selected_count']} episodes")
        print(f"Cutoff score: {topk_selection['cutoff_score']:.3f}")
        print(f"Score range: {topk_selection['min_score']:.3f} to {topk_selection['max_score']:.3f}")
        print(f"Average score: {topk_selection['avg_score']:.3f}")
        print(f"Overlap with threshold: {topk_selection['threshold_overlap']} ({topk_selection['threshold_overlap_rate']:.1%})")
        print()

        # Publish as Weave Dataset
        dataset_name_topk = f"training_data_{batch_name}_topk{int(top_k_percent*100)}"
        weave.publish(topk_selection, name=dataset_name_topk)
        print(f"💾 Published: '{dataset_name_topk}'")
        print()

        results['topk'] = topk_selection

    # Comparison
    if use_threshold and use_topk:
        print("=" * 70)
        print("📊 Comparison")
        print("=" * 70)

        thresh_count = threshold_selection['selected_count']
        topk_count = topk_selection['selected_count']

        print(f"{'Method':<20} {'Count':<10} {'Avg Score':<12} {'Min Score':<12}")
        print("-" * 70)
        print(f"{'Threshold':<20} {thresh_count:<10} {threshold_selection['avg_score']:<12.3f} {threshold_selection['min_score']:<12.3f}")
        print(f"{'Top K%':<20} {topk_count:<10} {topk_selection['avg_score']:<12.3f} {topk_selection['min_score']:<12.3f}")
        print()

        print("Recommendation:")
        if thresh_count >= 200:
            print("  ✓ Use THRESHOLD method - enough high-quality examples")
        elif topk_count >= 200:
            print("  ✓ Use TOP K% method - more training data available")
        else:
            print("  ⚠ Collect more episodes (need at least 200 training examples)")
        print()

    # Create final export summary
    export_summary = {
        "batch_name": batch_name,
        "timestamp": datetime.now().isoformat(),
        "total_episodes": len(episodes),
        "analysis": analysis,
        "threshold_selection": results.get('threshold'),
        "topk_selection": results.get('topk'),
        "top_k_percent": top_k_percent
    }

    # Publish combined summary
    summary_name = f"export_summary_{batch_name}"
    weave.publish(export_summary, name=summary_name)

    print("=" * 70)
    print("✅ Export Complete!")
    print("=" * 70)
    print(f"📊 Summary published: '{summary_name}'")
    print(f"🔗 View at: https://wandb.ai/ruchib-northwestern-university/synergi-rl-training/weave")
    print()
    print("Next steps:")
    print("1. Choose selection method (threshold or topk)")
    print("2. Run: python scripts/fine_tune.py --dataset <dataset_name>")
    print("=" * 70)

    return export_summary


def main():
    parser = argparse.ArgumentParser(description="Export training data from Weave traces")
    parser.add_argument("--batch", type=str, required=True,
                       help="Name of episode batch (e.g., iteration_0)")
    parser.add_argument("--top-k", type=float, default=0.30,
                       help="Top K percent to select (0.0-1.0)")
    parser.add_argument("--threshold-only", action="store_true",
                       help="Only use threshold-based selection")
    parser.add_argument("--topk-only", action="store_true",
                       help="Only use top-k selection")

    args = parser.parse_args()

    # Initialize Weave
    weave.init("synergi-rl-training")

    # Determine which methods to use
    use_threshold = not args.topk_only
    use_topk = not args.threshold_only

    # Export
    export_training_dataset(
        batch_name=args.batch,
        top_k_percent=args.top_k,
        use_threshold=use_threshold,
        use_topk=use_topk
    )


if __name__ == "__main__":
    main()
