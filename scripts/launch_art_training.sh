#!/bin/bash
# Launch OpenPipe ServerlessRL (ART) training for SynErgi Grid Multi-Agent System
#
# Usage:
#   ./scripts/launch_art_training.sh
#
# Prerequisites:
#   - .env configured with OPENPIPE_API_KEY, WANDB_API_KEY, etc.
#   - openpipe CLI installed: pip install openpipe
#   - Authenticated: openpipe login

set -e  # Exit on error

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Loaded environment variables from .env"
else
    echo "❌ .env file not found. Copy .env.example to .env and configure it."
    exit 1
fi

# Validate required variables
REQUIRED_VARS=("OPENPIPE_API_KEY" "WANDB_API_KEY" "WEAVE_PROJECT" "WANDB_PROJECT" "BASE_MODEL")
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "❌ Missing required environment variable: $VAR"
        exit 1
    fi
done

echo ""
echo "========================================================================"
echo "OpenPipe ServerlessRL (ART) Training Launch"
echo "========================================================================"
echo "  Model: $BASE_MODEL"
echo "  W&B Project: $WANDB_PROJECT"
echo "  Weave Project: $WEAVE_PROJECT"
echo "  Dataset: ${DATASET_REF:-none (synthetic only)}"
echo "  Horizon: ${HORIZON:-12} steps"
echo "  Temperature: ${TEMPERATURE:-0.3}"
echo "  Top-p: ${TOP_P:-0.9}"
echo "========================================================================"
echo ""

# Confirm before launching
read -p "Launch training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Launch ART training
echo ""
echo "🚀 Launching ART training..."
echo ""

openpipe art train \
  --env training.train_serverless_rl:make_env \
  --project weavehacks-grid \
  --method ppo \
  --episodes ${EPISODES:-300} \
  --model $BASE_MODEL

echo ""
echo "========================================================================"
echo "✅ Training job submitted!"
echo "========================================================================"
echo ""
echo "Monitor progress:"
echo "  - Weave traces: https://wandb.ai/$WEAVE_PROJECT/weave"
echo "  - Training metrics: https://wandb.ai/$WANDB_PROJECT"
echo "  - Live logs: openpipe art logs"
echo ""
echo "LoRA checkpoints will be saved as W&B Artifacts in: $WANDB_PROJECT"
echo "========================================================================"
