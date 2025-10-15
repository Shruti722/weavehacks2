# SynErgi - Multi-Agent Grid Optimization System Presentation: [Link](https://synergi.lovable.app/)

A multi-agent reinforcement learning system for optimizing power grid operations in San Francisco, using LangGraph for agent orchestration and W&B Inference for LLM-powered reasoning.

## Overview

SynErgi uses three specialized AI agents that collaborate to analyze, plan, and execute grid optimization decisions:

- **Analyst Agent**: Analyzes grid state, identifies supply-demand imbalances, and detects critical issues
- **Planner Agent**: Creates action plans to address grid issues while prioritizing equity-weighted zones
- **Actuator Agent**: Translates plans into specific execution instructions with equipment, timing, and safety checks

## Features

- 🤖 **Multi-Agent Workflow**: LangGraph-based agent collaboration with Digital Twin → Analyst → Planner → Actuator flow
- 🧠 **LLM-Powered Reasoning**: Uses Qwen2.5 14B Instruct via W&B Inference (optimized for future RL fine-tuning)
- ⚡ **Real-Time Grid Simulation**: 45 San Francisco neighborhoods with realistic demand/supply patterns
- 📊 **Comprehensive Analytics**: Risk assessment, fairness metrics, and equity-weighted prioritization
- 🔄 **Full Conversation Tracking**: Monitor agent-to-agent communication and reasoning

## Architecture

```
                    ┌─────────────────┐
                    │  Digital Twin   │
                    │  (Grid State)   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐         ┌─────────────────┐
     │ Analyst Agent   │◄───────►│ Planner Agent   │
     │ (Analyze State) │         │ (Create Plans)  │
     └────────┬────────┘         └────────┬────────┘
              │                           │
              └───────────┬───────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Actuator Agent  │
                 │ (Execute Plans) │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Digital Twin   │
                 │    (Update)     │
                 └─────────────────┘
```

**Agent Communication Flow:**
- Digital Twin provides initial state to both Analyst and Planner
- Analyst ↔ Planner communicate back-and-forth to refine analysis and plans
- Planner ↔ Actuator can query each other for clarification
- Actuator can loop back to Planner if plan needs adjustment
- Actuator updates Digital Twin with executed changes

## Installation

### Prerequisites

- Python 3.10+
- W&B account with API key ([get here](https://wandb.ai/authorize))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/imruchi/weavehacks2.git
   cd weavehacks2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your WANDB_API_KEY
   ```

## Usage

### Run the Multi-Agent System

**Default test:**
```bash
python agents/testing.py
```

**Custom prompt:**
```bash
python agents/testing.py "Optimize battery usage in Tenderloin and Bayview"
```

**Example prompts:**
- `"Find high-risk nodes and reduce overload"`
- `"Prioritize equity zones while minimizing costs"`
- `"Balance supply-demand across all districts"`

### Expected Output

The system will:
1. Generate realistic grid state (1500-1600 MW demand across 45 nodes)
2. Run agents through the workflow
3. Display detailed analysis, plan, and execution steps
4. Show full agent conversation transcript

Example output:
```
======================================================================
SynErgi Multi-Agent System - Test
======================================================================

📝 User Prompt: Analyze the grid and create a plan...

🔄 Generating grid state data...
   Time: 19:57
   Demand: 1593.0 MW
   Supply: 1524.9 MW
   Risk: 0.105

🤖 Initializing Multi-Agent System...
   ✓ System initialized

⚙️  Running Agent Workflow...
   Digital Twin → Analyst → Planner → Actuator

======================================================================
📊 RESULTS
======================================================================

🔍 ANALYST ANALYSIS:
The current grid state indicates a significant city-wide supply-demand
imbalance, with a total demand of 1593.0 MW exceeding supply by 68.1 MW...

📋 PLANNER ACTION PLAN:
Based on the analysis, here is the action plan:
- Financial District: +10 MW supply increase (gas turbine)
- Tenderloin: +2 MW discharge storage (battery)
- Russian Hill: -2 MW demand reduction (demand response)...

⚡ ACTUATOR EXECUTION:
Execution instructions:
- Start gas turbine in Financial District (0.5 MW/min ramp rate)
- Discharge battery in Tenderloin (5 min operation)
- Activate demand response in Russian Hill...
```

## Configuration

### Model Options

Edit `multi_agent_system.py` to change the LLM model:

```python
# Default: Qwen2.5 14B (recommended for RL)
mas = SynErgiMultiAgentSystem(model_name="Qwen/Qwen2.5-14B-Instruct")

# Alternatives:
# - meta-llama/Llama-3.1-8B-Instruct (lighter, 8B params)
# - microsoft/Phi-4-mini-instruct (very light, 3.8B params)
```

### Environment Variables

Create a `.env` file with:
```bash
WANDB_API_KEY=your_api_key_here
WEAVE_PROJECT=synergi-grid-optimization
```

---

## Run on ServerlessRL (ART)

**Fine-tune Qwen2.5-14B with PPO using OpenPipe's ServerlessRL (ART)**

### Prerequisites

1. **OpenPipe Account**: Sign up at [openpipe.ai](https://openpipe.ai)
2. **API Keys**: Get keys from:
   - OpenPipe: [app.openpipe.ai/settings](https://app.openpipe.ai/settings)
   - W&B: [wandb.ai/authorize](https://wandb.ai/authorize)

### Environment Setup

Configure environment variables in `.env`:

```bash
# Required
OPENPIPE_API_KEY=op-xxxxxxxx
WANDB_API_KEY=xxxxxxxx
WANDB_PROJECT=synergi-grid-optimization
WEAVE_PROJECT=your-username/synergi-grid

# Training config
BASE_MODEL=qwen2.5-14b           # Model: qwen2.5-14b, qwen2.5-7b, qwen2-72b
HORIZON=12                        # Episode length (12 steps = 1 hour)

# Optional
DATASET_REF=weave-traces-ds:latest  # Pre-collected trajectories
```

### Install OpenPipe CLI

```bash
pip install openpipe
openpipe login  # Follow prompts to authenticate
```

### Launch Training

**Option 1: Quick start (bash script)**
```bash
./scripts/launch_art_training.sh
```

**Option 2: Direct CLI command**
```bash
# Run RL training with OpenPipe ServerlessRL (ART)
openpipe art train \
  --env training.train_serverless_rl:make_env \
  --project weavehacks-grid \
  --method ppo \
  --episodes 300 \
  --model qwen2.5-14b
```

**Optional flags:**
```bash
# Advanced tuning
openpipe art train \
  --env training.train_serverless_rl:make_env \
  --project weavehacks-grid \
  --method ppo \
  --episodes 300 \
  --model qwen2.5-14b \
  --ppo-epochs 4 \
  --batch-size 32 \
  --learning-rate 3e-5 \
  --lora-rank 64 \
  --gamma 0.99 \
  --clip-epsilon 0.2
```

**What happens:**
- ART creates parallel environments and runs PPO training
- `make_env()` loads W&B Artifact dataset (`weave-traces-ds:latest`) for seeding
- Each `reset()` seeds initial context from random dataset row (or synthetic fallback)
- Model calls route through OpenPipe's API (`https://api.openpipe.ai/v1`)
- Policy samples with `TEMPERATURE` and `TOP_P` from env (ART can override for KL control)
- LoRA checkpoints saved as W&B Artifacts (project: `WANDB_PROJECT`)
- Weave logs all LLM traces (no separate W&B runs per episode)
- Returns scalar `reward: float` from `step()` for PPO optimizer

### Monitor Training

**Weave Dashboard** (LLM traces):
```bash
https://wandb.ai/your-username/synergi-grid/weave
```

**W&B Dashboard** (training metrics):
```bash
https://wandb.ai/your-username/synergi-grid-optimization
```

**View live logs**:
```bash
openpipe art logs
```

### Use Fine-Tuned Model

After training completes, use the fine-tuned checkpoint:

```python
from multi_agent_system import SynErgiMultiAgentSystem

# Load fine-tuned model from W&B artifact
mas = SynErgiMultiAgentSystem(
    model_name="openpipe:ft-qwen2.5-14b-xxxxx"  # From ART output
)
```

### Test Locally

Test environment before submitting to ART:

```bash
python training/train_serverless_rl.py
```

This runs a single episode locally to verify setup.

### Key Features

- ✅ **No training loops**: ART handles reset()/step() orchestration
- ✅ **Automatic PPO**: Hyperparameter tuning included
- ✅ **LoRA fine-tuning**: Efficient 14B parameter model training
- ✅ **W&B Artifacts**: Checkpoints saved automatically
- ✅ **Weave tracing**: All LLM calls logged for debugging
- ✅ **Fail-fast validation**: Missing env vars caught immediately

## Project Structure

```
weavehacks2/
├── agents/
│   ├── analyst_agent.py      # Grid analysis agent
│   ├── planner_agent.py      # Action planning agent
│   ├── actuator_agent.py     # Execution agent
│   ├── testing.py            # Test script
│   └── tools/                # Agent tools (future use)
├── data/
│   ├── simulator.py          # Grid state simulator
│   ├── data.json             # SF grid template
│   └── current_state.json    # Latest simulation
├── multi_agent_system.py     # LangGraph orchestration
├── requirements.txt          # Dependencies
└── README.md
```

## Technical Details

### Grid Simulator

Generates realistic power grid data for 45 SF neighborhoods:
- Dynamic demand/supply following time-of-day patterns
- Weather simulation (temperature, solar, wind)
- Real-time pricing (day-ahead and real-time)
- Storage state of charge (SOC)
- Risk indicators (overload, N-1 margin)
- Fairness metrics across equity zones

### Agent Communication

Agents communicate through LangGraph state:
- **Digital Twin** provides initial grid state
- **Analyst** receives state, outputs analysis
- **Planner** receives analysis, outputs action plan
- **Actuator** receives plan, outputs execution instructions

Each agent sees the full conversation history.

### Model Selection

**Qwen2.5 14B Instruct** is used because:
- ✅ 14B parameters (manageable for RL fine-tuning with LoRA/QLoRA)
- ✅ Strong reasoning capabilities
- ✅ Supports tool use and multi-turn conversations
- ✅ Free tier on W&B Inference
- ✅ Good balance of performance vs. trainability

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

## License

MIT License

## Acknowledgments

- Built for Weave Hacks 2025
- Uses [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Powered by [W&B Inference](https://wandb.ai/site/inference) for LLM hosting
- Grid data inspired by San Francisco's power infrastructure

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Status**: 🚧 Active Development | **Model**: Qwen2.5 14B | **Last Updated**: October 2025
